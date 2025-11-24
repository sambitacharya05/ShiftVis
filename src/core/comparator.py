from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, model_validator, ConfigDict
import cv2
import logging

from .segmentor import Segment, ImageSegmentor
from .aligner import AlignmentResult, AlignmentType, ImageAligner
from .validator import ValidationResult, ConsistencyValidator
from .preprocessor import Preprocessor
from .config import settings
from .storage_manager import ComparisonStorageManager

class ChangeType(str, Enum):
    """
    Enumeration of possible change types detected in segment comparison.
    
    Inherits from str for better JSON serialization and API compatibility.
    
    Values:
        NONE: No changes detected (identical content)
        VISUAL_CHANGE: Content has changed (pixels differ)
        POSITION_SHIFT: Content moved but unchanged (spatial displacement)
        NO_MATCH: Segment not found in test image (added/deleted content)
        SKIPPED: Segment not processed (low importance/confidence)
    """
    NONE = "none"
    VISUAL_CHANGE = "visual_change"
    POSITION_SHIFT = "position_shift"
    NO_MATCH = "no_match"
    SKIPPED = "skipped"


class BoundingBox(BaseModel):
    """
    Represents a rectangular region where changes were detected.
    
    Attributes:
        x: Top-left x-coordinate in full image space
        y: Top-left y-coordinate in full image space
        width: Width of bounding box in pixels
        height: Height of bounding box in pixels
        confidence: Confidence score (0-1) for this detection
        label: Optional text label describing the change
    
    Examples:
        >>> bbox = BoundingBox(x=100, y=200, width=50, height=30, confidence=0.95)
        >>> bbox.area()
        1500
        >>> other_bbox = BoundingBox(x=120, y=210, width=40, height=25, confidence=0.90)
        >>> bbox.overlaps(other_bbox)
        True
        >>> bbox.iou(other_bbox)
        0.524
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=False
    )
    
    x: int = Field(..., ge=0, description="Top-left x-coordinate")
    y: int = Field(..., ge=0, description="Top-left y-coordinate")
    width: int = Field(..., gt=0, description="Width in pixels")
    height: int = Field(..., gt=0, description="Height in pixels")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    label: Optional[str] = Field(None, description="Optional text label")
    
    def area(self) -> int:
        """Calculate area of bounding box."""
        return self.width * self.height
    
    def center(self) -> Tuple[float, float]:
        """Calculate center point of bounding box."""
        return (self.x + self.width / 2.0, self.y + self.height / 2.0)
    
    def overlaps(self, other: 'BoundingBox') -> bool:
        """Check if this box overlaps with another."""
        return not (
            self.x + self.width <= other.x or
            other.x + other.width <= self.x or
            self.y + self.height <= other.y or
            other.y + other.height <= self.y
        )
    
    def iou(self, other: 'BoundingBox') -> float:
        """
        Calculate Intersection over Union (IoU) with another box.
        
        Returns:
            float: IoU score (0-1), where 1 = perfect overlap
        """
        if not self.overlaps(other):
            return 0.0
        
        # Calculate intersection area
        x_left = max(self.x, other.x)
        y_top = max(self.y, other.y)
        x_right = min(self.x + self.width, other.x + other.width)
        y_bottom = min(self.y + self.height, other.y + other.height)
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def to_dict_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        data = self.model_dump()
        data['center'] = self.center()
        data['area'] = self.area()
        return data


class SegmentComparisonResult(BaseModel):
    """
    Results of comparing a single segment pair (baseline vs test).
    
    Attributes:
        segment_id: Unique identifier for this segment (e.g., "seg_r2_c3")
        has_changes: Boolean indicating if changes were detected
        change_type: Classification of change type
        pixel_diff_map: Binary mask showing changed pixels (Optional for memory)
        diff_percentage: Percentage of pixels that differ (0-100)
        diff_pixel_count: Absolute count of changed pixels
        change_regions: List of bounding boxes around detected changes
        alignment_info: Alignment result from the aligner
        baseline_segment: Reference to the baseline segment
        metadata: Additional information (SSIM, processing time, etc.)
    
    Examples:
        >>> result = SegmentComparisonResult(
        ...     segment_id="seg_r2_c3",
        ...     has_changes=True,
        ...     change_type=ChangeType.VISUAL_CHANGE,
        ...     diff_percentage=2.5,
        ...     diff_pixel_count=1600
        ... )
        >>> result.is_significant()
        True
        >>> result.get_change_summary()
        'Visual change: 2.50% different (1600 pixels)'
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=False,
        use_enum_values=False  # Keep enum objects for type safety
    )
    
    segment_id: str = Field(..., min_length=1, description="Unique segment identifier")
    has_changes: bool = Field(..., description="Whether changes were detected")
    change_type: ChangeType = Field(..., description="Type of change detected")
    pixel_diff_map: Optional[np.ndarray] = Field(
        None,
        description="Binary mask of changed pixels (optional for memory efficiency)",
        exclude=True  # Exclude from JSON serialization
    )
    diff_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of pixels that differ (0-100)"
    )
    diff_pixel_count: int = Field(
        ...,
        ge=0,
        description="Absolute count of changed pixels"
    )
    change_regions: List[BoundingBox] = Field(
        default_factory=list,
        description="Bounding boxes around detected changes"
    )
    alignment_info: Optional[AlignmentResult] = Field(
        None,
        description="Alignment result from the aligner"
    )
    baseline_segment: Optional[Segment] = Field(
        None,
        description="Reference to baseline segment"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (SSIM, processing time, etc.)"
    )
    
    @model_validator(mode='after')
    def validate_consistency(self) -> 'SegmentComparisonResult':
        """
        Validate logical consistency between fields.
        
        Enforces strict mapping between has_changes and allowed ChangeType values:
        - has_changes=False: Only NONE or SKIPPED allowed
        - has_changes=True: Only VISUAL_CHANGE, POSITION_SHIFT, or NO_MATCH allowed
        """
        # Define allowed change types for each has_changes state
        ALLOWED_WHEN_NO_CHANGES = {ChangeType.NONE, ChangeType.SKIPPED}
        ALLOWED_WHEN_HAS_CHANGES = {
            ChangeType.VISUAL_CHANGE,
            ChangeType.POSITION_SHIFT,
            ChangeType.NO_MATCH
        }
        
        # Validate has_changes=False constraints
        if not self.has_changes and self.change_type not in ALLOWED_WHEN_NO_CHANGES:
            raise ValueError(
                f"Inconsistent state: has_changes=False but change_type={self.change_type.value}. "
                f"When has_changes=False, change_type must be one of: "
                f"{', '.join(ct.value for ct in ALLOWED_WHEN_NO_CHANGES)}"
            )
        
        # Validate has_changes=True constraints
        if self.has_changes and self.change_type not in ALLOWED_WHEN_HAS_CHANGES:
            raise ValueError(
                f"Inconsistent state: has_changes=True but change_type={self.change_type.value}. "
                f"When has_changes=True, change_type must be one of: "
                f"{', '.join(ct.value for ct in ALLOWED_WHEN_HAS_CHANGES)}"
            )
        
        # Validate diff_percentage vs diff_pixel_count consistency
        if self.diff_percentage == 0.0 and self.diff_pixel_count > 0:
            raise ValueError(
                "Inconsistent: diff_percentage=0 but diff_pixel_count>0"
            )
        
        return self
    

    def is_significant(self, threshold: float = settings.COMPARATOR_SIGNIFICANCE_THRESHOLD) -> bool:
        """
        Check if changes exceed significance threshold.
        
        Args:
            threshold: Minimum diff_percentage to consider significant (default from settings)
            
        Returns:
            bool: True if changes are significant
        """
        return self.has_changes and self.diff_percentage >= threshold
    
    def get_change_summary(self) -> str:
        """
        Get human-readable summary of changes.
        
        Returns:
            str: Description of the change
        """
        if self.change_type == ChangeType.NONE:
            return "No changes detected"
        elif self.change_type == ChangeType.VISUAL_CHANGE:
            return f"Visual change: {self.diff_percentage:.2f}% different ({self.diff_pixel_count} pixels)"
        elif self.change_type == ChangeType.POSITION_SHIFT:
            shift = self.alignment_info.shift if self.alignment_info else (0, 0)
            magnitude = np.sqrt(shift[0]**2 + shift[1]**2)
            return f"Position shift: {shift} ({magnitude:.1f}px)"
        elif self.change_type == ChangeType.NO_MATCH:
            return "Content not found in test image"
        elif self.change_type == ChangeType.SKIPPED:
            return "Segment skipped (low importance)"
        return "Unknown change type"
    
    def get_change_density(self) -> float:
        """
        Calculate change density (changed pixels per 1000 pixels of area).
        
        Returns:
            float: Changes per 1000 pixels (0 if no baseline segment)
        """
        if not self.baseline_segment:
            return 0.0
        area = self.baseline_segment.area()
        return (self.diff_pixel_count / area) * 1000 if area > 0 else 0.0
    
    def get_spatial_info(self) -> Dict[str, Any]:
        """
        Get spatial information about this segment.
        
        Returns:
            dict: Position, size, and location info
        """
        if not self.baseline_segment:
            return {}
        
        return {
            "position": (self.baseline_segment.x, self.baseline_segment.y),
            "size": (self.baseline_segment.width, self.baseline_segment.height),
            "center": self.baseline_segment.center(),
            "row_col": self.baseline_segment.get_row_col(),
            "area": self.baseline_segment.area()
        }
    
    def has_alignment(self) -> bool:
        """
        Check if alignment was performed and successful.
        
        Returns:
            bool: True if valid alignment exists
        """
        return (
            self.alignment_info is not None and
            self.alignment_info.alignment_type not in [
                AlignmentType.NO_MATCH,
                AlignmentType.LOW_CONFIDENCE
            ]
        )
    
    def to_dict_serializable(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable dictionary (excludes numpy arrays).
        
        Returns:
            dict: Serializable representation
        """
        data = self.model_dump(
            exclude={'pixel_diff_map', 'alignment_info', 'baseline_segment'}
        )
        
        # Add summary info for excluded fields
        data['has_pixel_diff_map'] = self.pixel_diff_map is not None
        data['has_alignment_info'] = self.alignment_info is not None
        data['has_baseline_segment'] = self.baseline_segment is not None
        
        # Add computed fields
        data['change_summary'] = self.get_change_summary()
        data['change_density'] = self.get_change_density()
        data['spatial_info'] = self.get_spatial_info()
        
        # Add alignment info if available
        if self.alignment_info:
            data['alignment_summary'] = {
                'shift': self.alignment_info.shift,
                'similarity_score': self.alignment_info.similarity_score,
                'alignment_type': self.alignment_info.alignment_type.value
            }
        
        return data
    
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


class DocumentComparisonResult(BaseModel):
    """
    Aggregated results of comparing two complete documents.
    
    Attributes:
        overall_similarity: Overall similarity score (0-1, where 1=identical)
        total_segments: Total number of segments processed
        segments_with_changes: Count of segments with detected changes
        change_percentage: Percentage of segments with changes (0-100)
        segment_results: List of per-segment comparison results
        validation_result: Result from consistency validator
        global_shift_applied: Global shift correction applied (dx, dy) or None
        merged_change_regions: Combined bounding boxes from all segments
        visualization_data: Data prepared for visualization/reporting
        processing_time: Total processing time in seconds
        metadata: Additional information (paths, timestamps, config, etc.)
    
    Examples:
        >>> result = DocumentComparisonResult(...)
        >>> if result.is_mostly_identical():
        ...     print("Documents are very similar")
        >>> stats = result.get_summary_stats()
        >>> print(f"Visual changes: {stats['visual_changes']}")
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=False
    )
    
    overall_similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall similarity score (0=different, 1=identical)"
    )
    total_segments: int = Field(
        ...,
        gt=0,
        description="Total number of segments processed"
    )
    segments_with_changes: int = Field(
        ...,
        ge=0,
        description="Count of segments with detected changes"
    )
    change_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of segments with changes (0-100)"
    )
    segment_results: List[SegmentComparisonResult] = Field(
        ...,
        min_length=1,
        description="Per-segment comparison results"
    )
    validation_result: ValidationResult = Field(
        ...,
        description="Result from consistency validator"
    )
    global_shift_applied: Optional[Tuple[int, int]] = Field(
        None,
        description="Global shift correction applied (dx, dy)"
    )
    merged_change_regions: List[BoundingBox] = Field(
        default_factory=list,
        description="Combined bounding boxes from all segments"
    )
    visualization_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Data prepared for visualization/reporting"
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Total processing time in seconds"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (paths, timestamps, config, etc.)"
    )
    
    @model_validator(mode='after')
    def validate_consistency(self) -> 'DocumentComparisonResult':
        """Validate logical consistency between fields."""
        # Check segment count consistency
        if len(self.segment_results) != self.total_segments:
            raise ValueError(
                f"Inconsistent segment count: total_segments={self.total_segments} "
                f"but got {len(self.segment_results)} segment_results"
            )
        
        # Check segments_with_changes consistency
        actual_changes = sum(1 for seg in self.segment_results if seg.has_changes)
        if actual_changes != self.segments_with_changes:
            raise ValueError(
                f"Inconsistent change count: segments_with_changes={self.segments_with_changes} "
                f"but {actual_changes} segments actually have changes"
            )
        
        # Validate change_percentage calculation
        expected_change_pct = (self.segments_with_changes / self.total_segments) * 100
        if abs(self.change_percentage - expected_change_pct) > 0.1:
            raise ValueError(
                f"Inconsistent change_percentage: {self.change_percentage}% "
                f"but calculated {expected_change_pct:.1f}%"
            )
        
        return self
    
    def get_changed_segments(self) -> List[SegmentComparisonResult]:
        """Get only segments with detected changes."""
        return [seg for seg in self.segment_results if seg.has_changes]
    
    def get_segments_by_type(self, change_type: ChangeType) -> List[SegmentComparisonResult]:
        """Get segments of a specific change type."""
        return [seg for seg in self.segment_results if seg.change_type == change_type]
    
    def is_mostly_identical(self, threshold: float = settings.COMPARATOR_MATCH_THRESHOLD) -> bool:
        """
        Check if documents are mostly identical.
        
        Args:
            threshold: Minimum similarity score (default from settings)
            
        Returns:
            bool: True if similarity exceeds threshold
        """
        return self.overall_similarity >= threshold
    
    def has_significant_changes(self, threshold: float = settings.COMPARATOR_CHANGE_THRESHOLD) -> bool:
        """
        Check if document has significant changes.
        
        Args:
            threshold: Minimum change_percentage (default from settings)
            
        Returns:
            bool: True if changes exceed threshold
        """
        return self.change_percentage >= threshold
    
    def has_position_shifts(self) -> bool:
        """Check if any segments have position shifts."""
        return len(self.get_segments_by_type(ChangeType.POSITION_SHIFT)) > 0
    
    def has_content_missing(self) -> bool:
        """Check if any segments are missing in test image."""
        return len(self.get_segments_by_type(ChangeType.NO_MATCH)) > 0
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for reporting.
        
        Returns:
            dict: Human-readable statistics
        """
        return {
            "total_segments": self.total_segments,
            "unchanged": len(self.get_segments_by_type(ChangeType.NONE)),
            "visual_changes": len(self.get_segments_by_type(ChangeType.VISUAL_CHANGE)),
            "position_shifts": len(self.get_segments_by_type(ChangeType.POSITION_SHIFT)),
            "no_matches": len(self.get_segments_by_type(ChangeType.NO_MATCH)),
            "skipped": len(self.get_segments_by_type(ChangeType.SKIPPED)),
            "overall_similarity": f"{self.overall_similarity * 100:.1f}%",
            "change_percentage": f"{self.change_percentage:.1f}%",
            "processing_time": f"{self.processing_time:.2f}s",
            "global_shift": str(self.global_shift_applied) if self.global_shift_applied else "None",
            "consistency_score": f"{self.validation_result.consistency_score * 100:.1f}%"
        }
    
    def get_detailed_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for analysis.
        
        Returns:
            dict: Detailed stats including per-type breakdowns
        """
        stats = self.get_summary_stats()
        
        # Add per-type statistics
        visual_changes = self.get_segments_by_type(ChangeType.VISUAL_CHANGE)
        if visual_changes:
            stats['visual_changes_details'] = {
                'count': len(visual_changes),
                'avg_diff_percentage': float(np.mean([s.diff_percentage for s in visual_changes])),
                'max_diff_percentage': float(max(s.diff_percentage for s in visual_changes)),
                'total_changed_pixels': sum(s.diff_pixel_count for s in visual_changes),
                'avg_change_density': float(np.mean([s.get_change_density() for s in visual_changes]))
            }
        
        position_shifts = self.get_segments_by_type(ChangeType.POSITION_SHIFT)
        if position_shifts:
            shifts = [
                s.alignment_info.shift 
                for s in position_shifts 
                if s.alignment_info
            ]
            if shifts:
                stats['position_shifts_details'] = {
                    'count': len(shifts),
                    'avg_shift_x': float(np.mean([s[0] for s in shifts])),
                    'avg_shift_y': float(np.mean([s[1] for s in shifts])),
                    'max_shift_magnitude': float(max(
                        np.sqrt(s[0]**2 + s[1]**2) for s in shifts
                    ))
                }
        
        # Add validation info
        stats['validation_details'] = {
            'is_consistent': self.validation_result.is_consistent,
            'consistency_score': self.validation_result.consistency_score,
            'outlier_count': self.validation_result.outlier_count,
            'outlier_percentage': self.validation_result.get_outlier_percentage(),
            'is_highly_consistent': self.validation_result.is_highly_consistent()
        }
        
        return stats
    
    def get_change_heatmap_data(self) -> List[Dict[str, Any]]:
        """
        Get data formatted for heatmap visualization.
        
        Returns:
            list: List of dicts with position and intensity data
        """
        heatmap_data = []
        for seg in self.segment_results:
            if seg.baseline_segment and seg.has_changes:
                heatmap_data.append({
                    'x': seg.baseline_segment.x,
                    'y': seg.baseline_segment.y,
                    'width': seg.baseline_segment.width,
                    'height': seg.baseline_segment.height,
                    'intensity': seg.diff_percentage / 100.0,  # Normalize to 0-1
                    'change_type': seg.change_type.value,
                    'segment_id': seg.segment_id
                })
        return heatmap_data
    
    def to_json(self, **kwargs) -> str:
        """
        Export to JSON string.
        
        Args:
            **kwargs: Additional arguments for model_dump_json
            
        Returns:
            str: JSON representation
        """
        # Exclude non-serializable fields
        return self.model_dump_json(
            exclude={
                'segment_results': {
                    '__all__': {
                        'pixel_diff_map',
                        'alignment_info',
                        'baseline_segment'
                    }
                }
            },
            **kwargs
        )
    
    def save_to_file(self, filepath: str, indent: int = 2) -> None:
        """
        Save comparison results to JSON file.
        
        Args:
            filepath: Path to save JSON file
            indent: JSON indentation level (default 2)
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json(indent=indent))
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'DocumentComparisonResult':
        """
        Load comparison results from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            DocumentComparisonResult instance
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls.model_validate_json(f.read())

class ImageComparator:

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.segmentor = ImageSegmentor(
            segment_size=settings.SEGMENT_SIZE,
            overlap_percentage=settings.SEGMENT_OVERLAP_PERCENTAGE
        )
        self.aligner = ImageAligner(
            local_search_radius=settings.ALIGNER_LOCAL_SEARCH_RADIUS,
            large_search_radius=settings.ALIGNER_LARGE_SEARCH_RADIUS,
            tier1_similarity_threshold=settings.ALIGNER_TIER1_THRESHOLD,
            tier2_similarity_threshold=settings.ALIGNER_TIER2_THRESHOLD,
            entropy_threshold=settings.ALIGNER_ENTROPY_THRESHOLD,
            variance_threshold=settings.ALIGNER_VARIANCE_THRESHOLD
        )
        self.validator = ConsistencyValidator(
            consistency_threshold=settings.VALIDATOR_CONSISTENCY_THRESHOLD,
            outlier_threshold=settings.VALIDATOR_OUTLIER_THRESHOLD
        )
        self.settings = settings

        self.pixel_diff_threshold = settings.COMPARATOR_PIXEL_DIFF_THRESHOLD
        self.min_change_pixels = settings.COMPARATOR_MIN_CHANGE_PIXELS
        self.min_contour_area = settings.COMPARATOR_MIN_CONTOUR_AREA
        self.morphology_kernel_size = settings.COMPARATOR_MORPHOLOGY_KERNEL_SIZE
        self.merge_distance = settings.COMPARATOR_MERGE_DISTANCE
        self.change_classification_delta = settings.COMPARATOR_CHANGE_CLASSIFICATION_DELTA

        #legacy
        self.match_threshold = settings.COMPARATOR_MATCH_THRESHOLD
        self.change_threshold = settings.COMPARATOR_CHANGE_THRESHOLD
        self.significance_threshold = settings.COMPARATOR_SIGNIFICANCE_THRESHOLD

        # visualization
        self.addition_color = settings.get_addition_color_rgb()
        self.deletion_color = settings.get_deletion_color_rgb()
        self.modification_color = settings.get_modification_color_rgb()
        self.bbox_thickness = settings.VISUALIZATION_BBOX_THICKNESS
        self.overlay_alpha = settings.VISUALIZATION_OVERLAY_ALPHA
        self.heatmap_colormap = settings.VISUALIZATION_HEATMAP_COLORMAP

        # performance
        self.enable_parallel_alignment = settings.PERFORMANCE_ENABLE_PARALLEL_ALIGNMENT
        self.enable_parallel_comparison = settings.PERFORMANCE_ENABLE_PARALLEL_COMPARISON
        self.max_workers = settings.PERFORMANCE_MAX_WORKERS
        self.chunk_size = settings.PERFORMANCE_CHUNK_SIZE

        # storage
        self.output_dir = Path(settings.STORAGE_OUTPUT_DIR)
        self.save_pixel_masks = settings.STORAGE_SAVE_PIXEL_MASKS
        self.mask_storage_format = settings.STORAGE_MASK_STORAGE_FORMAT

        # logging/debug
        self.debug_mode = settings.DEBUG_MODE
        self.save_intermediate_results = settings.LOGGING_SAVE_INTERMEDIATE_RESULTS
        self.intermediate_output_dir = Path(settings.LOGGING_INTERMEDIATE_OUTPUT_DIR)
        self.timing_enabled = settings.LOGGING_TIMING_ENABLED

        # runtime
        self._timing_data: Dict[str, float] = {}  # Phase timing tracking
        self._cache: Dict[str, Any] = {}  # Optional result caching
        self._processing_metadata: Dict[str, Any] = {}  # Current comparison metadata

        # Initialize logger FIRST before using it
        import logging
        self.logger = logging.getLogger(__name__)

        # output dir
        if self.save_intermediate_results:
            self.intermediate_output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # debug logging init
        if self.debug_mode:
            self._log_initialization()


    # ============================================================================
    # INTERNAL UTILITIES
    # ============================================================================
    def _log_initialization(self) -> None:
        """Log initialization configuration for debugging."""        
        self.logger.info("=" * 70)
        self.logger.info("ImageComparator Initialized")
        self.logger.info("=" * 70)
        self.logger.info(f"Segment size: {self.segmentor.segment_size}px")
        self.logger.info(f"Segment overlap: {self.segmentor.overlap_pixels}px ({settings.SEGMENT_OVERLAP_PERCENTAGE*100:.0f}%)")
        self.logger.info(f"Tier-1 threshold: {self.aligner.tier1_similarity_threshold}")
        self.logger.info(f"Tier-2 threshold: {self.aligner.tier2_similarity_threshold}")
        self.logger.info(f"Pixel diff threshold: {self.pixel_diff_threshold:.4f}")
        self.logger.info(f"Min change pixels: {self.min_change_pixels}")
        self.logger.info(f"Grid step: {self.aligner.grid_step}")
        self.logger.info(f"SSIM window size: {self.aligner.ssim_win_size}")
        self.logger.info(f"Early termination: {self.aligner.early_termination}")
        self.logger.info(f"Parallel alignment: {self.enable_parallel_alignment}")
        self.logger.info(f"Parallel comparison: {self.enable_parallel_comparison}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 70)
    
    def _start_timer(self, phase: str) -> None:
        """Start timing a processing phase."""
        if self.timing_enabled:
            import time
            self._timing_data[f"{phase}_start"] = time.time()
    
    def _end_timer(self, phase: str) -> float:
        """End timing a processing phase and return duration."""
        if self.timing_enabled:
            import time
            start_key = f"{phase}_start"
            if start_key in self._timing_data:
                duration = time.time() - self._timing_data[start_key]
                self._timing_data[phase] = duration
                return duration
        return 0.0
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get timing summary for all phases."""
        return {
            k: v for k, v in self._timing_data.items()
            if not k.endswith('_start')
        }
    
    def reset(self) -> None:
        """Reset internal state for new comparison."""
        self._timing_data.clear()
        self._cache.clear()
        self._processing_metadata.clear()
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return {
            "segmentation": {
                "segment_size": self.segmentor.segment_size,
                "overlap_pixels": self.segmentor.overlap_pixels,
                "overlap_percentage": settings.SEGMENT_OVERLAP_PERCENTAGE
            },
            "alignment": {
                "local_search_radius": self.aligner.local_search_radius,
                "large_search_radius": self.aligner.large_search_radius,
                "tier1_threshold": self.aligner.tier1_similarity_threshold,
                "tier2_threshold": self.aligner.tier2_similarity_threshold,
                "entropy_threshold": self.aligner.entropy_threshold,
                "variance_threshold": self.aligner.variance_threshold
            },
            "comparison": {
                "pixel_diff_threshold": self.pixel_diff_threshold,
                "min_change_pixels": self.min_change_pixels,
                "min_contour_area": self.min_contour_area,
                "morphology_kernel_size": self.morphology_kernel_size,
                "merge_distance": self.merge_distance
            },
            "validation": {
                "outlier_threshold": self.validator.outlier_threshold,
                "consistency_threshold": self.validator.consistency_threshold
            },
            "performance": {
                "parallel_alignment": self.enable_parallel_alignment,
                "parallel_comparison": self.enable_parallel_comparison,
                "max_workers": self.max_workers
            },
            "storage": {
                "output_dir": str(self.output_dir),
                "save_pixel_masks": self.save_pixel_masks,
                "mask_format": self.mask_storage_format
            }
        }
    
    def _calculate_pixel_diff(
            self,
            baseline_segment: np.ndarray,
            test_segment: np.ndarray
    ) -> Tuple[np.ndarray, int, float]:
        """
        Calculate pixel-level differences between two segments.
        
        Converts to grayscale for diff calculation (preprocessor outputs BGR by default).
        
        Args:
            baseline_segment: Reference segment data
            test_segment: Test segment data to compare
            
        Returns:
            Tuple of (diff_map, changed_pixel_count, diff_percentage)
        """
        # Convert BGR to grayscale for difference calculation
        # Preprocessor outputs BGR by default (see preprocessor.py line 142)
        if baseline_segment.ndim == 3:
            baseline_gray = cv2.cvtColor(baseline_segment, cv2.COLOR_BGR2GRAY)
            test_gray = cv2.cvtColor(test_segment, cv2.COLOR_BGR2GRAY)
        else:
            baseline_gray = baseline_segment
            test_gray = test_segment
        
        # Calculate absolute difference
        abs_diff = cv2.absdiff(baseline_gray, test_gray)

        # Apply threshold to get binary diff map
        threshold_value = int(self.pixel_diff_threshold * 255)
        _, diff_map = cv2.threshold(
            abs_diff,
            threshold_value,
            255,
            cv2.THRESH_BINARY
        )

        # Calculate statistics
        changed_pixel_count = np.count_nonzero(diff_map)
        total_pixels = diff_map.size
        diff_percentage = (changed_pixel_count / total_pixels) * 100.0 if total_pixels > 0 else 0.0

        return diff_map, changed_pixel_count, diff_percentage
    
    def _classify_change_type(
            self,
            alignment_result: AlignmentResult,
            diff_percentage: float,
            diff_pixel_count: int
    ) -> ChangeType:
        """
        Classify the type of change detected between segments.
        
        Decision logic:
        1. NO_MATCH/LOW_CONFIDENCE → return early
        2. Below min_change_pixels → NONE
        3. High similarity (tier1) + minimal shift → NONE
        4. LARGE_SHIFT type + high similarity (tier2) → POSITION_SHIFT
        5. Large shift magnitude + high similarity (tier2) → POSITION_SHIFT  
        6. Everything else → VISUAL_CHANGE
        
        Args:
            alignment_result: Result from alignment
            diff_percentage: Percentage of pixels that differ
            diff_pixel_count: Absolute count of changed pixels
            
        Returns:
            ChangeType classification
        """
        # Handle no match or low confidence
        if alignment_result.alignment_type == AlignmentType.NO_MATCH:
            return ChangeType.NO_MATCH
        
        if alignment_result.alignment_type == AlignmentType.LOW_CONFIDENCE:
            return ChangeType.SKIPPED
        
        # Below minimum threshold → no change
        if diff_pixel_count < self.min_change_pixels:
            return ChangeType.NONE
        
        similarity_score = alignment_result.similarity_score
        shift = alignment_result.shift
        shift_magnitude = np.sqrt(shift[0]**2 + shift[1]**2)

        # High similarity with minimal/no shift → no change
        if similarity_score >= self.aligner.tier1_similarity_threshold and shift_magnitude <= 2:
            if diff_percentage < 1.0:
                return ChangeType.NONE
        
        # POSITION_SHIFT: Content moved but structurally similar
        # Case 1: Explicit LARGE_SHIFT type with good tier-2 similarity
        if (alignment_result.alignment_type == AlignmentType.LARGE_SHIFT and
            similarity_score >= self.aligner.tier2_similarity_threshold):
            return ChangeType.POSITION_SHIFT
        
        # Case 2: Significant shift with good tier-2 similarity (implicit position shift)
        if (similarity_score >= self.aligner.tier2_similarity_threshold and 
            shift_magnitude > 5):
            return ChangeType.POSITION_SHIFT
        
        # Default: visual change
        return ChangeType.VISUAL_CHANGE
    
    def _detect_change_regions(
            self,
            diff_map: np.ndarray,
            segment: Segment
    ) -> List[BoundingBox]:
        """
        Detect bounding boxes around changed regions using morphological operations.
        
        Args:
            diff_map: Binary mask of changed pixels
            segment: The segment being analyzed (for global coordinate conversion)
            
        Returns:
            List of BoundingBox objects in global image coordinates
        """
        # Apply morphological closing to connect nearby changes
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )
        closed = cv2.morphologyEx(diff_map, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Convert contours to bounding boxes
        bboxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Convert to global coordinates
            global_x = segment.x + x
            global_y = segment.y + y
            
            # Calculate confidence based on area
            confidence = min(1.0, area / (w * h) if (w * h) > 0 else 0.0)
            
            bbox = BoundingBox(
                x=global_x,
                y=global_y,
                width=w,
                height=h,
                confidence=confidence,
                label=f"change_{segment.segment_id}"
            )
            bboxes.append(bbox)
        
        return bboxes
    
    def _compare_segments(
            self,
            baseline_segments: List[Segment],
            alignment_results: List[AlignmentResult],
            baseline_image: np.ndarray
    ) -> List[SegmentComparisonResult]:
        """
        Compare all aligned segments and generate comparison results.
        
        Args:
            baseline_segments: List of baseline segments
            alignment_results: List of alignment results (parallel to baseline_segments)
            baseline_image: Full baseline image for extracting segment data
            
        Returns:
            List of SegmentComparisonResult objects
        """
        results = []
        
        for baseline_seg, alignment_result in zip(baseline_segments, alignment_results):
            try:
                # Extract baseline data
                baseline_data = self.segmentor.extract_segment_data(baseline_image, baseline_seg)
                
                # Extract test data from alignment result
                if alignment_result.aligned_data is None:
                    # No match or low confidence - determine change type first
                    change_type = (
                        ChangeType.NO_MATCH 
                        if alignment_result.alignment_type == AlignmentType.NO_MATCH
                        else ChangeType.SKIPPED
                    )
                    
                    # SKIPPED (low confidence/blank) = no changes
                    # NO_MATCH (completely different) = has changes
                    has_changes = (change_type == ChangeType.NO_MATCH)
                    
                    result = SegmentComparisonResult(
                        segment_id=baseline_seg.segment_id,
                        has_changes=has_changes,
                        change_type=change_type,
                        diff_percentage=100.0 if has_changes else 0.0,
                        diff_pixel_count=baseline_seg.area() if has_changes else 0,
                        alignment_info=alignment_result,
                        baseline_segment=baseline_seg
                    )
                    results.append(result)
                    continue
                
                test_data = alignment_result.aligned_data
                
                # Calculate pixel differences
                diff_map, diff_pixel_count, diff_percentage = self._calculate_pixel_diff(
                    baseline_data,
                    test_data
                )
                
                # Detect change regions
                change_regions = self._detect_change_regions(diff_map, baseline_seg)
                
                # Classify change type
                change_type = self._classify_change_type(
                    alignment_result,
                    diff_percentage,
                    diff_pixel_count
                )
                
                # Create result
                result = SegmentComparisonResult(
                    segment_id=baseline_seg.segment_id,
                    has_changes=(change_type not in [ChangeType.NONE, ChangeType.SKIPPED]),
                    change_type=change_type,
                    pixel_diff_map=diff_map,
                    diff_percentage=diff_percentage,
                    diff_pixel_count=diff_pixel_count,
                    change_regions=change_regions,
                    alignment_info=alignment_result,
                    baseline_segment=baseline_seg,
                    metadata={
                        "similarity_score": alignment_result.similarity_score,
                        "shift": alignment_result.shift,
                        "alignment_type": alignment_result.alignment_type.value
                    }
                )
                results.append(result)
                
            except Exception as e:
                # Create error result
                result = SegmentComparisonResult(
                    segment_id=baseline_seg.segment_id,
                    has_changes=False,
                    change_type=ChangeType.SKIPPED,
                    diff_percentage=0.0,
                    diff_pixel_count=0,
                    baseline_segment=baseline_seg,
                    metadata={"error": str(e)}
                )
                results.append(result)
                
                if self.debug_mode:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error comparing segment {baseline_seg.segment_id}: {e}")
        
        return results
    
    def _align_all_segments(
            self,
            baseline_segments: List[Segment],
            test_image: np.ndarray,
            baseline_image: np.ndarray
    ) -> List[AlignmentResult]:
        """
        Align all baseline segments with the test image.
        
        Args:
            baseline_segments: List of segments from baseline image
            test_image: Full test image
            baseline_image: Full baseline image (needed for extracting segment data)
            
        Returns:
            List of AlignmentResult objects
        """
        alignment_results = []
        total_segments = len(baseline_segments)
        
        self.logger.info(f"Starting alignment of {total_segments} segments...")
        
        for i, segment in enumerate(baseline_segments, 1):
            # Perform alignment
            alignment_result = self.aligner.find_best_alignment(
                test_image=test_image,
                baseline_segment=segment,
                baseline_image=baseline_image
            )
            alignment_results.append(alignment_result)
            
            # Log progress every 10 segments or at end
            if i % 10 == 0 or i == total_segments:
                self.logger.info(f"Aligned {i}/{total_segments} segments ({i/total_segments*100:.0f}%)")
        
        self.logger.info(f"Alignment complete: {total_segments} segments processed")
        return alignment_results
    
    def _save_comparison_results(
            self,
            comparison_results: List[SegmentComparisonResult],
            doc_id: str,
            page_num: int
    ) -> Path:
        """
        Save comparison results to workspace using storage manager.
        
        Args:
            comparison_results: List of segment comparison results
            doc_id: Document identifier
            page_num: Page number
            
        Returns:
            Path to page workspace directory
        """
        # Create storage manager
        storage_manager = ComparisonStorageManager(
            output_dir=self.output_dir,
            compression_level=1
        )
        
        # Create document workspace
        doc_workspace = storage_manager.create_document_workspace(doc_id)
        
        # Create page workspace
        page_workspace = doc_workspace / f"page_{page_num:03d}"
        page_workspace.mkdir(exist_ok=True)
        
        # Save diff maps
        if self.save_pixel_masks:
            for result in comparison_results:
                if result.pixel_diff_map is not None:
                    result.save_diff_map_to_workspace(storage_manager, page_workspace)
        
        return page_workspace
    
    def _aggregate_results(
            self,
            segment_results: List[SegmentComparisonResult],
            validation_result: ValidationResult,
            processing_time: float,
            baseline_path: str,
            test_path: str
    ) -> DocumentComparisonResult:
        """
        Aggregate segment results into document-level comparison result.
        
        Args:
            segment_results: List of segment comparison results
            validation_result: Validation result from consistency check
            processing_time: Total processing time
            baseline_path: Path to baseline image
            test_path: Path to test image
            
        Returns:
            DocumentComparisonResult with aggregated statistics
        """
        # Calculate statistics
        total_segments = len(segment_results)
        segments_with_changes = sum(1 for r in segment_results if r.has_changes)
        change_percentage = (segments_with_changes / total_segments * 100.0) if total_segments > 0 else 0.0
        
        # Calculate overall similarity
        similarity_scores = [
            r.alignment_info.similarity_score 
            for r in segment_results 
            if r.alignment_info is not None
        ]
        overall_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Collect all change regions
        all_change_regions = []
        for result in segment_results:
            all_change_regions.extend(result.change_regions)
        
        # Prepare visualization data
        visualization_data = {
            "heatmap": self._prepare_heatmap_data(segment_results),
            "change_regions_count": len(all_change_regions),
            "change_type_distribution": self._calculate_change_distribution(segment_results)
        }
        
        # Create document result
        result = DocumentComparisonResult(
            overall_similarity=float(overall_similarity),
            total_segments=total_segments,
            segments_with_changes=segments_with_changes,
            change_percentage=change_percentage,
            segment_results=segment_results,
            validation_result=validation_result,
            global_shift_applied=validation_result.dominant_shift if validation_result.is_consistent else None,
            merged_change_regions=all_change_regions,
            visualization_data=visualization_data,
            processing_time=processing_time,
            metadata={
                "baseline_path": baseline_path,
                "test_path": test_path,
                "config": self.get_config_summary(),
                "timing": self.get_timing_summary()
            }
        )
        
        return result
    
    def _prepare_heatmap_data(self, segment_results: List[SegmentComparisonResult]) -> List[Dict[str, Any]]:
        """Prepare heatmap data for visualization."""
        heatmap = []
        for result in segment_results:
            if result.baseline_segment:
                heatmap.append({
                    "x": result.baseline_segment.x,
                    "y": result.baseline_segment.y,
                    "width": result.baseline_segment.width,
                    "height": result.baseline_segment.height,
                    "intensity": result.diff_percentage,
                    "change_type": result.change_type.value
                })
        return heatmap
    
    def _calculate_change_distribution(self, segment_results: List[SegmentComparisonResult]) -> Dict[str, int]:
        """Calculate distribution of change types."""
        distribution = {ct.value: 0 for ct in ChangeType}
        for result in segment_results:
            distribution[result.change_type.value] += 1
        return distribution
    
    def compare_documents(
            self,
            baseline_image_path: str,
            test_image_path: str,
            doc_id: str = "comparison",
            page_num: int = 1
    ) -> DocumentComparisonResult:
        """
        Complete document comparison pipeline.
        
        Pipeline phases:
        1-2: Preprocessing (load and normalize images)
        3-4: Segmentation (create tiles with overlap)
        5-6: Alignment (multi-tier search for each segment)
        7: Validation (consistency check with MAD-based outlier detection)
        8-10: Comparison (pixel-level diff, contour detection, classification)
        11: Storage (save compressed diff maps, metadata)
        12-15: Aggregation (statistics, heatmap data, DocumentComparisonResult)
        
        Args:
            baseline_image_path: Path to baseline/reference image
            test_image_path: Path to test/comparison image
            doc_id: Document identifier for workspace organization
            page_num: Page number for multi-page documents
            
        Returns:
            DocumentComparisonResult with complete analysis
        """
        import time
        
        start_time = time.time()
        self.reset()
        
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING COMPARISON: {doc_id} (page {page_num})")
        self.logger.info(f"Baseline: {baseline_image_path}")
        self.logger.info(f"Test: {test_image_path}")
        self.logger.info("=" * 80)
        
        # Phase 1-2: Preprocessing
        self._start_timer("preprocessing")
        self.logger.info("[PHASE 1-2] Loading and preprocessing images...")
        baseline_image = cv2.imread(baseline_image_path)
        test_image = cv2.imread(test_image_path)
        
        if baseline_image is None:
            raise ValueError(f"Could not load baseline image: {baseline_image_path}")
        if test_image is None:
            raise ValueError(f"Could not load test image: {test_image_path}")
        
        baseline_image = self.preprocessor.prepare_single_image(baseline_image, "baseline")
        test_image = self.preprocessor.prepare_single_image(test_image, "test")
        prep_time = self._end_timer("preprocessing")
        self.logger.info(f"[PHASE 1-2] ✓ Preprocessing complete in {prep_time:.2f}s")
        self.logger.info(f"  Image dimensions: {baseline_image.shape[1]}x{baseline_image.shape[0]}")
        
        # Phase 3-4: Segmentation
        self._start_timer("segmentation")
        self.logger.info("[PHASE 3-4] Segmenting baseline image...")
        baseline_segments = self.segmentor.segment_image(baseline_image)
        seg_time = self._end_timer("segmentation")
        self.logger.info(f"[PHASE 3-4] ✓ Segmentation complete in {seg_time:.2f}s")
        self.logger.info(f"  Created {len(baseline_segments)} segments")
        
        # Phase 5-6: Alignment
        self._start_timer("alignment")
        self.logger.info(f"[PHASE 5-6] Aligning {len(baseline_segments)} segments...")
        alignment_results = self._align_all_segments(baseline_segments, test_image, baseline_image)
        align_time = self._end_timer("alignment")
        self.logger.info(f"[PHASE 5-6] ✓ Alignment complete in {align_time:.2f}s")
        self.logger.info(f"  Average time per segment: {align_time/len(baseline_segments):.3f}s")
        
        # Phase 7: Validation
        self._start_timer("validation")
        self.logger.info("[PHASE 7] Validating alignment consistency...")
        validation_result = self.validator.validate_vector_field(alignment_results)
        val_time = self._end_timer("validation")
        self.logger.info(f"[PHASE 7] ✓ Validation complete in {val_time:.2f}s")
        self.logger.info(
            f"  Consistency: {validation_result.consistency_score:.2%}, "
            f"Outliers: {validation_result.outlier_count}/{validation_result.valid_segments}"
        )
        
        # Phase 8-10: Comparison
        self._start_timer("comparison")
        self.logger.info(f"[PHASE 8-10] Comparing {len(baseline_segments)} segments...")
        segment_results = self._compare_segments(baseline_segments, alignment_results, baseline_image)
        comp_time = self._end_timer("comparison")
        changes = sum(1 for r in segment_results if r.has_changes)
        self.logger.info(f"[PHASE 8-10] ✓ Comparison complete in {comp_time:.2f}s")
        self.logger.info(f"  Segments with changes: {changes}/{len(segment_results)}")
        
        # Phase 11: Storage
        self._start_timer("storage")
        if self.save_pixel_masks:
            self.logger.info("[PHASE 11] Saving comparison results...")
            page_workspace = self._save_comparison_results(segment_results, doc_id, page_num)
            stor_time = self._end_timer("storage")
            self.logger.info(f"[PHASE 11] ✓ Storage complete in {stor_time:.2f}s")
            self.logger.info(f"  Saved to: {page_workspace}")
        else:
            self._end_timer("storage")
            self.logger.info("[PHASE 11] Storage skipped (save_pixel_masks=False)")
        
        # Phase 12-15: Aggregation
        self._start_timer("aggregation")
        self.logger.info("[PHASE 12-15] Aggregating results...")
        total_time = time.time() - start_time
        document_result = self._aggregate_results(
            segment_results,
            validation_result,
            total_time,
            baseline_image_path,
            test_image_path
        )
        agg_time = self._end_timer("aggregation")
        
        self.logger.info("=" * 80)
        self.logger.info("COMPARISON COMPLETE")
        self.logger.info(f"Total time: {total_time:.2f}s")
        self.logger.info(f"Overall similarity: {document_result.overall_similarity:.2%}")
        self.logger.info(f"Change percentage: {document_result.change_percentage:.1f}%")
        self.logger.info("=" * 80)
        self.logger.info("Timing Breakdown:")
        for phase, duration in self.get_timing_summary().items():
            self.logger.info(f"  {phase}: {duration:.2f}s ({duration/total_time*100:.1f}%)")
        self.logger.info("=" * 80)
        
        return document_result

if __name__ == "__main__":
    comparator = ImageComparator()
    bl = cv2.imread("data/test-docs/integration test/baseline_text shift.png")
    test = cv2.imread("data/test-docs/integration test/test_text shift.png")
    result = comparator.compare_documents(bl, test)
    str = result.get_detailed_statistics()
    file = "data/test-docs/integration test/output/integration_test.json"
    file.mkdir(parents=True, exist_ok=True)
    jsonObj = result.save_to_file(filepath=file)
    