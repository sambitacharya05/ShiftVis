"""
Output Manager for Frontend-Ready Structure

Organizes comparison results into a frontend-friendly structure with dual views:
1. "with_shift" - Shows original comparison with global shift detection
2. "aligned" - Shows comparison after applying shift correction

Directory structure:
    output_dir/
        metadata.json                    # Master control file
        assets/
            original/
                baseline.png
                test.png
            with_shift/
                overlay.png
                side_by_side.png
            aligned/
                test_aligned.png
                overlay.png
                side_by_side.png
            thumbnails/
                baseline_thumb.png
                test_thumb.png
        data/
            shift_analysis.json
            segments.json
            content_changes.json
            statistics.json
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2


class OutputManager:
    """Manages frontend-ready output structure for comparison results."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize OutputManager.
        
        Args:
            output_dir: Root directory for organized output
        """
        self.output_dir = Path(output_dir)
        self.assets_dir = self.output_dir / "assets"
        self.data_dir = self.output_dir / "data"
        
        # Create directory structure
        self._create_directory_structure()
    
    def _create_directory_structure(self) -> None:
        """Create the output directory structure."""
        dirs = [
            self.assets_dir / "original",
            self.assets_dir / "with_shift",
            self.assets_dir / "aligned",
            self.assets_dir / "thumbnails",
            self.data_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_original_images(
        self,
        baseline_img: np.ndarray,
        test_img: np.ndarray,
        thumbnail_size: Tuple[int, int] = (200, 200)
    ) -> None:
        """
        Save original baseline and test images.
        
        Args:
            baseline_img: Baseline image array
            test_img: Test image array
            thumbnail_size: Size for thumbnails (width, height)
        """
        # Save full images
        cv2.imwrite(
            str(self.assets_dir / "original" / "baseline.png"),
            baseline_img
        )
        cv2.imwrite(
            str(self.assets_dir / "original" / "test.png"),
            test_img
        )
        
        # Generate and save thumbnails
        baseline_thumb = self._create_thumbnail(baseline_img, thumbnail_size)
        test_thumb = self._create_thumbnail(test_img, thumbnail_size)
        
        cv2.imwrite(
            str(self.assets_dir / "thumbnails" / "baseline_thumb.png"),
            baseline_thumb
        )
        cv2.imwrite(
            str(self.assets_dir / "thumbnails" / "test_thumb.png"),
            test_thumb
        )
    
    def save_shift_view(
        self,
        overlay_img: np.ndarray,
        side_by_side_img: np.ndarray
    ) -> None:
        """
        Save "with shift" view images (default view showing global shift).
        
        Args:
            overlay_img: Color-coded overlay image
            side_by_side_img: Side-by-side comparison image
        """
        cv2.imwrite(
            str(self.assets_dir / "with_shift" / "overlay.png"),
            overlay_img
        )
        cv2.imwrite(
            str(self.assets_dir / "with_shift" / "side_by_side.png"),
            side_by_side_img
        )
    
    def save_aligned_view(
        self,
        test_aligned_img: np.ndarray,
        overlay_img: np.ndarray,
        side_by_side_img: np.ndarray
    ) -> None:
        """
        Save "aligned" view images (toggle view after shift correction).
        
        Args:
            test_aligned_img: Test image after applying shift correction
            overlay_img: Color-coded overlay after alignment
            side_by_side_img: Side-by-side comparison after alignment
        """
        cv2.imwrite(
            str(self.assets_dir / "aligned" / "test_aligned.png"),
            test_aligned_img
        )
        cv2.imwrite(
            str(self.assets_dir / "aligned" / "overlay.png"),
            overlay_img
        )
        cv2.imwrite(
            str(self.assets_dir / "aligned" / "side_by_side.png"),
            side_by_side_img
        )
    
    def save_shift_analysis_data(self, shift_data: Dict[str, Any]) -> None:
        """
        Save global shift analysis data.
        
        Args:
            shift_data: Dictionary containing shift analysis results
                {
                    "detected": bool,
                    "shift_vector": [dx, dy],
                    "confidence": float,
                    "direction": str,
                    "magnitude": float,
                    "inlier_count": int,
                    "total_segments": int
                }
        """
        output_path = self.data_dir / "shift_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(shift_data, f, indent=2)
    
    def save_segments_data(self, segments: List[Dict[str, Any]]) -> None:
        """
        Save detailed segment information.
        
        Args:
            segments: List of segment dictionaries with metadata
        """
        output_path = self.data_dir / "segments.json"
        with open(output_path, 'w') as f:
            json.dump({"segments": segments}, f, indent=2)
    
    def save_content_changes_data(self, content_changes: List[Dict[str, Any]]) -> None:
        """
        Save filtered list of segments with actual content changes.
        
        Args:
            content_changes: List of segments classified as CONTENT_CHANGE
        """
        output_path = self.data_dir / "content_changes.json"
        with open(output_path, 'w') as f:
            json.dump({"content_changes": content_changes}, f, indent=2)
    
    def save_statistics_data(self, statistics: Dict[str, Any]) -> None:
        """
        Save comparison statistics.
        
        Args:
            statistics: Dictionary with summary statistics
        """
        output_path = self.data_dir / "statistics.json"
        with open(output_path, 'w') as f:
            json.dump(statistics, f, indent=2)
    
    def save_metadata(
        self,
        baseline_path: str,
        test_path: str,
        comparison_id: str,
        timestamp: str,
        has_global_shift: bool,
        shift_vector: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Save master metadata file for frontend.
        
        Args:
            baseline_path: Path to baseline image
            test_path: Path to test image
            comparison_id: Unique comparison identifier
            timestamp: Comparison timestamp
            has_global_shift: Whether global shift was detected
            shift_vector: Global shift vector if detected
        """
        metadata = {
            "comparison_id": comparison_id,
            "timestamp": timestamp,
            "baseline_image": baseline_path,
            "test_image": test_path,
            "has_global_shift": has_global_shift,
            "shift_vector": shift_vector,
            "views": {
                "with_shift": {
                    "description": "Original comparison showing global shift",
                    "overlay": "assets/with_shift/overlay.png",
                    "side_by_side": "assets/with_shift/side_by_side.png"
                },
                "aligned": {
                    "description": "Comparison after shift correction",
                    "test_aligned": "assets/aligned/test_aligned.png",
                    "overlay": "assets/aligned/overlay.png",
                    "side_by_side": "assets/aligned/side_by_side.png"
                }
            },
            "data_files": {
                "shift_analysis": "data/shift_analysis.json",
                "segments": "data/segments.json",
                "content_changes": "data/content_changes.json",
                "statistics": "data/statistics.json"
            }
        }
        
        output_path = self.output_dir / "metadata.json"
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def apply_shift_correction(
        self,
        test_img: np.ndarray,
        shift_vector: Tuple[int, int],
        fill_color: Tuple[int, int, int] = (255, 255, 255)
    ) -> np.ndarray:
        """
        Apply shift correction to test image.
        
        Args:
            test_img: Test image to correct
            shift_vector: (dx, dy) shift to apply
            fill_color: Color for filled areas (default white)
        
        Returns:
            Shift-corrected test image
        """
        dx, dy = shift_vector
        h, w = test_img.shape[:2]
        
        # Create translation matrix (negate shift to correct it)
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        
        # Apply translation with white background fill
        corrected = cv2.warpAffine(
            test_img,
            M,
            (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_color
        )
        
        return corrected
    
    def _create_thumbnail(
        self,
        img: np.ndarray,
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create thumbnail maintaining aspect ratio.
        
        Args:
            img: Input image
            size: Maximum (width, height)
        
        Returns:
            Thumbnail image
        """
        h, w = img.shape[:2]
        max_w, max_h = size
        
        # Calculate scaling factor maintaining aspect ratio
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        thumbnail = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas with white background
        canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
        
        # Center thumbnail on canvas
        y_offset = (max_h - new_h) // 2
        x_offset = (max_w - new_w) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail
        
        return canvas
