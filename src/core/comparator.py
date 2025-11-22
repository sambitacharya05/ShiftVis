from typing import Optional, List, Dict, Tuple, Any
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .segmentor import Segment
from aligner import AlignmentResult, AlignmentType
from validator import ValidationResult


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
        """Validate logical consistency between fields."""
        # Check has_changes consistency with change_type
        if self.has_changes and self.change_type == ChangeType.NONE:
            raise ValueError(
                "Inconsistent state: has_changes=True but change_type=NONE"
            )
        
        if not self.has_changes and self.change_type in [
            ChangeType.VISUAL_CHANGE,
            ChangeType.NO_MATCH
        ]:
            raise ValueError(
                f"Inconsistent state: has_changes=False but change_type={self.change_type}"
            )
        
        # Validate diff_percentage vs diff_pixel_count consistency
        if self.diff_percentage == 0.0 and self.diff_pixel_count > 0:
            raise ValueError(
                "Inconsistent: diff_percentage=0 but diff_pixel_count>0"
            )
        
        return self
    
    def is_significant(self, threshold: float = 0.2) -> bool:
        """
        Check if changes exceed significance threshold.
        
        Args:
            threshold: Minimum diff_percentage to consider significant (default 0.2%)
            
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
    
    def is_mostly_identical(self, threshold: float = 0.95) -> bool:
        """
        Check if documents are mostly identical.
        
        Args:
            threshold: Minimum similarity score (default 95%)
            
        Returns:
            bool: True if similarity exceeds threshold
        """
        return self.overall_similarity >= threshold
    
    def has_significant_changes(self, threshold: float = 1.0) -> bool:
        """
        Check if document has significant changes.
        
        Args:
            threshold: Minimum change_percentage (default 1%)
            
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


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of Pydantic-based comparison data structures.
    """
    
    # Example 1: Create a BoundingBox with validation
    print("=" * 70)
    print("Example 1: BoundingBox Creation & Validation")
    print("=" * 70)
    
    try:
        bbox = BoundingBox(
            x=100, y=200, width=50, height=30, 
            confidence=0.95, label="Text change"
        )
        print(f"✅ Valid BoundingBox created")
        print(f"   Area: {bbox.area()} pixels")
        print(f"   Center: {bbox.center()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    try:
        # This will fail validation (negative width)
        invalid_bbox = BoundingBox(
            x=100, y=200, width=-50, height=30, confidence=0.95
        )
    except Exception as e:
        print(f"✅ Validation caught invalid width: {type(e).__name__}")
    
    # Example 2: Create SegmentComparisonResult
    print("\n" + "=" * 70)
    print("Example 2: SegmentComparisonResult")
    print("=" * 70)
    
    try:
        seg_result = SegmentComparisonResult(
            segment_id="seg_r2_c3",
            has_changes=True,
            change_type=ChangeType.VISUAL_CHANGE,
            diff_percentage=2.5,
            diff_pixel_count=1600,
            change_regions=[bbox],
            metadata={"ssim": 0.94, "processing_time_ms": 150}
        )
        print(f"✅ Created: {seg_result.segment_id}")
        print(f"   Summary: {seg_result.get_change_summary()}")
        print(f"   Significant: {seg_result.is_significant()}")
        print(f"   Has alignment: {seg_result.has_alignment()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Example 3: Test validation (inconsistent state)
    print("\n" + "=" * 70)
    print("Example 3: Validation - Inconsistent State")
    print("=" * 70)
    
    try:
        # This will fail: has_changes=True but change_type=NONE
        invalid_result = SegmentComparisonResult(
            segment_id="seg_invalid",
            has_changes=True,
            change_type=ChangeType.NONE,  # ❌ Inconsistent!
            diff_percentage=0.0,
            diff_pixel_count=0
        )
    except Exception as e:
        print(f"✅ Validation caught inconsistency: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    # Example 4: JSON Serialization
    print("\n" + "=" * 70)
    print("Example 4: JSON Serialization")
    print("=" * 70)
    
    json_data = seg_result.to_dict_serializable()
    print(f"✅ Serialized to dict")
    print(f"   Keys: {list(json_data.keys())}")
    print(f"   Change summary: {json_data['change_summary']}")
    print(f"   Change density: {json_data['change_density']:.2f}")
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)