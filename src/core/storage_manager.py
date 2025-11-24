import json
import uuid
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Dict, List, Any
import hashlib

from .config import settings
if TYPE_CHECKING:
    from .comparator import DocumentComparisonResult


class ComparisonStorageManager:
    """
    Manages storage of comparison results with compressed diff maps.
    
    Directory structure:
        output_dir/
        └── {timestamp}/
            └── {document_id}/
                ├── metadata.json
                ├── summary.json
                └── page_{N}/
                    ├── segments.json
                    ├── overlay.png
                    └── diffMaps/
                        ├── seg_{id}.npz
                        └── index.json
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        compression_level: int = 9  # Max compression (1-9)
    ):
        """
        Initialize storage manager.
        
        Args:
            output_dir: Base output directory (defaults to config)
            compression_level: NPZ compression level (1-9, higher = smaller files)
        """
        self.output_dir = Path(output_dir or settings.STORAGE_OUTPUT_DIR)
        self.compression_level = compression_level
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_document_workspace(
        self,
        document_name: Optional[str] = None,
        baseline_path: Optional[str] = None,
        test_path: Optional[str] = None
    ) -> Path:
        """
        Create a unique workspace for a document comparison.
        
        Args:
            document_name: Optional document name
            baseline_path: Path to baseline file (used for ID generation)
            test_path: Path to test file (used for ID generation)
            
        Returns:
            Path to document workspace directory
        """
        # Create timestamp directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        timestamp_dir = self.output_dir / timestamp
        timestamp_dir.mkdir(exist_ok=True)
        
        # Generate unique document ID
        doc_id = self._generate_document_id(
            document_name,
            baseline_path,
            test_path
        )
        
        # Create document directory
        doc_dir = timestamp_dir / doc_id
        doc_dir.mkdir(exist_ok=True)
        
        return doc_dir
    
    def _generate_document_id(
        self,
        document_name: Optional[str],
        baseline_path: Optional[str],
        test_path: Optional[str]
    ) -> str:
        """
        Generate unique document ID.
        
        Strategy:
        1. Use document_name if provided
        2. Hash baseline + test paths if available
        3. Fallback to UUID
        
        Args:
            document_name: Optional document name
            baseline_path: Path to baseline file
            test_path: Path to test file
            
        Returns:
            Unique document ID string
        """
        if document_name:
            # Sanitize document name for filesystem
            safe_name = "".join(
                c if c.isalnum() or c in ('-', '_') else '_'
                for c in document_name
            )[:50]  # Limit length
            # Add short hash for uniqueness
            hash_suffix = hashlib.md5(
                f"{baseline_path}{test_path}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:8]
            return f"{safe_name}_{hash_suffix}"
        
        elif baseline_path and test_path:
            # Hash file paths for deterministic ID
            combined = f"{baseline_path}|{test_path}"
            hash_value = hashlib.md5(combined.encode()).hexdigest()[:12]
            return f"doc_{hash_value}"
        
        else:
            # Fallback to UUID
            return f"doc_{uuid.uuid4().hex[:12]}"
    
    def create_page_workspace(
        self,
        doc_workspace: Path,
        page_number: int
    ) -> Path:
        """
        Create workspace for a single page/image.
        
        Args:
            doc_workspace: Document workspace directory
            page_number: Page number (1-indexed)
            
        Returns:
            Path to page workspace directory
        """
        page_dir = doc_workspace / f"page_{page_number:03d}"
        page_dir.mkdir(exist_ok=True)
        
        # Create diffMaps subdirectory
        diffmaps_dir = page_dir / "diffMaps"
        diffmaps_dir.mkdir(exist_ok=True)
        
        return page_dir
    
    def save_diff_map(
        self,
        page_workspace: Path,
        segment_id: str,
        diff_map: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a segment's diff map with heavy compression.
        
        Args:
            page_workspace: Page workspace directory
            segment_id: Unique segment identifier (e.g., "seg_0042")
            diff_map: Binary diff map (H, W) uint8 array
            metadata: Optional metadata to store with diff map
            
        Returns:
            Relative path to saved diff map file
        """
        diffmaps_dir = page_workspace / "diffMaps"
        
        # Validate diff map
        if diff_map.dtype != np.uint8:
            diff_map = diff_map.astype(np.uint8)
        
        # Compress diff map
        # Strategy: Use NPZ with maximum compression
        filename = f"{segment_id}.npz"
        filepath = diffmaps_dir / filename
        
        # Additional compression: Run-length encode or sparse storage
        # For binary masks, store only changed pixel coordinates (even smaller)
        if self._should_use_sparse_storage(diff_map):
            changed_coords = np.argwhere(diff_map > 0)
            np.savez_compressed(
                filepath,
                coords=changed_coords,
                shape=diff_map.shape,
                storage_type='sparse',
                metadata=metadata or {}
            )
        else:
            # Standard compressed storage
            np.savez_compressed(
                filepath,
                mask=diff_map,
                storage_type='dense',
                metadata=metadata or {}
            )
        
        # Update index
        self._update_diffmap_index(
            page_workspace,
            segment_id,
            filename,
            diff_map.shape,
            metadata
        )
        
        return f"diffMaps/{filename}"
    
    def _should_use_sparse_storage(self, diff_map: np.ndarray) -> bool:
        """
        Decide if sparse storage is more efficient.
        
        Sparse storage stores only changed pixel coordinates.
        Efficient when < 10% of pixels are changed.
        
        Args:
            diff_map: Binary diff map
            
        Returns:
            True if sparse storage should be used
        """
        sparsity = np.count_nonzero(diff_map) / diff_map.size
        return sparsity < 0.10  # Use sparse if < 10% changed
    
    def _update_diffmap_index(
        self,
        page_workspace: Path,
        segment_id: str,
        filename: str,
        shape: tuple,
        metadata: Optional[Dict[str, Any]]
    ) -> None:
        """
        Update the diffMaps/index.json file.
        
        Args:
            page_workspace: Page workspace directory
            segment_id: Segment identifier
            filename: Diff map filename
            shape: Diff map shape
            metadata: Optional metadata
        """
        index_path = page_workspace / "diffMaps" / "index.json"
        
        # Load existing index or create new
        if index_path.exists():
            with open(index_path, 'r') as f:
                index = json.load(f)
        else:
            index = {"segments": {}}
        
        # Add/update entry
        index["segments"][segment_id] = {
            "filename": filename,
            "shape": list(shape),
            "metadata": metadata or {}
        }
        
        # Save index
        with open(index_path, 'w') as f:
            json.dump(index, f, indent=2)
    
    def load_diff_map(
        self,
        page_workspace: Path,
        segment_id: str
    ) -> np.ndarray:
        """
        Load a segment's diff map from disk.
        
        Args:
            page_workspace: Page workspace directory
            segment_id: Segment identifier
            
        Returns:
            Decompressed diff map array
        """
        # Read index to find file
        index_path = page_workspace / "diffMaps" / "index.json"
        with open(index_path, 'r') as f:
            index = json.load(f)
        
        segment_info = index["segments"][segment_id]
        filepath = page_workspace / "diffMaps" / segment_info["filename"]
        
        # Load compressed data
        data = np.load(filepath)
        
        # Reconstruct based on storage type
        if data['storage_type'] == 'sparse':
            # Reconstruct from sparse coordinates
            shape = tuple(data['shape'])
            diff_map = np.zeros(shape, dtype=np.uint8)
            coords = data['coords']
            if len(coords) > 0:
                diff_map[coords[:, 0], coords[:, 1]] = 255
            return diff_map
        else:
            # Dense storage
            return data['mask']
    
    def save_page_metadata(
        self,
        page_workspace: Path,
        segments_metadata: List[Dict[str, Any]],
        page_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save page-level metadata (segments info, statistics).
        
        Args:
            page_workspace: Page workspace directory
            segments_metadata: List of segment metadata dicts
            page_metadata: Optional page-level metadata
        """
        segments_path = page_workspace / "segments.json"
        
        data = {
            "page_metadata": page_metadata or {},
            "segments": segments_metadata,
            "total_segments": len(segments_metadata),
            "changed_segments": sum(
                1 for s in segments_metadata 
                if s.get('has_changes', False)
            )
        }
        
        with open(segments_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save_document_metadata(
        self,
        doc_workspace: Path,
        result: 'DocumentComparisonResult'
    ) -> None:
        """
        Save document-level metadata and summary.
        
        Args:
            doc_workspace: Document workspace directory
            result: DocumentComparisonResult instance
        """
        # Full metadata
        metadata_path = doc_workspace / "metadata.json"
        with open(metadata_path, 'w') as f:
            # Exclude large arrays, keep only references
            metadata = result.model_dump(
                exclude={'segment_results': {'__all__': {'pixel_diff_map'}}}
            )
            json.dump(metadata, f, indent=2)
        
        # Quick summary for listing
        summary_path = doc_workspace / "summary.json"
        with open(summary_path, 'w') as f:
            summary = {
                "document_id": doc_workspace.name,
                "timestamp": result.metadata.get("timestamp", ""),
                "total_segments": len(result.segment_results),
                "changed_segments": result.segments_with_changes,
                "change_percentage": result.change_percentage,
                "processing_time": result.processing_time,
                "status": "completed"
            }
            json.dump(summary, f, indent=2)
    
    def get_comparison_info(
        self,
        doc_workspace: Path
    ) -> Dict[str, Any]:
        """
        Get summary info for a comparison without loading full data.
        
        Args:
            doc_workspace: Document workspace directory
            
        Returns:
            Summary dictionary
        """
        summary_path = doc_workspace / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary not found: {summary_path}")
        
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    def list_comparisons(
        self,
        timestamp: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all comparisons, optionally filtered by timestamp.
        
        Args:
            timestamp: Optional timestamp directory to filter
            
        Returns:
            List of comparison summaries
        """
        comparisons = []
        
        # Get timestamp directories
        if timestamp:
            timestamp_dirs = [self.output_dir / timestamp]
        else:
            timestamp_dirs = sorted(
                [d for d in self.output_dir.iterdir() if d.is_dir()],
                reverse=True  # Most recent first
            )
        
        # Collect summaries
        for ts_dir in timestamp_dirs:
            for doc_dir in ts_dir.iterdir():
                if doc_dir.is_dir():
                    try:
                        summary = self.get_comparison_info(doc_dir)
                        summary['workspace_path'] = str(doc_dir)
                        comparisons.append(summary)
                    except FileNotFoundError:
                        continue
        
        return comparisons


# ============================================================================
# Integration with SegmentComparisonResult
# ============================================================================

# Add these methods to SegmentComparisonResult class in comparator.py

def save_diff_map_to_workspace(
    self,
    storage_manager: ComparisonStorageManager,
    page_workspace: Path
) -> Optional[str]:
    """
    Save this segment's diff map to workspace.
    
    Args:
        storage_manager: ComparisonStorageManager instance
        page_workspace: Page workspace directory
        
    Returns:
        Relative path to saved diff map, or None if no diff map
    """
    if self.pixel_diff_map is None:
        return None
    
    # Save with metadata
    metadata = {
        "segment_id": self.segment_id,
        "change_type": self.change_type.value,
        "diff_percentage": self.diff_percentage,
        "diff_pixel_count": self.diff_pixel_count
    }
    
    path = storage_manager.save_diff_map(
        page_workspace,
        self.segment_id,
        self.pixel_diff_map,
        metadata
    )
    
    # Clear from memory after saving
    self.pixel_diff_map = None
    
    return path


def load_diff_map_from_workspace(
    self,
    storage_manager: ComparisonStorageManager,
    page_workspace: Path
) -> np.ndarray:
    """
    Load this segment's diff map from workspace.
    
    Args:
        storage_manager: ComparisonStorageManager instance
        page_workspace: Page workspace directory
        
    Returns:
        Decompressed diff map array
    """
    return storage_manager.load_diff_map(page_workspace, self.segment_id)