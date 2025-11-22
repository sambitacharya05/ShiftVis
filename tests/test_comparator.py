"""
Tests for the Comparator data structures.
"""

import pytest
import numpy as np
from src.core.comparator import (
    BoundingBox, SegmentComparisonResult, DocumentComparisonResult, ChangeType
)
from src.core.validator import ValidationResult
from src.core.aligner import AlignmentResult, AlignmentType

class TestComparatorStructures:
    """Tests for Comparator Pydantic models."""

    def test_bounding_box(self):
        """Test BoundingBox creation and methods."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50, confidence=0.9)
        
        assert bbox.area() == 5000
        assert bbox.center() == (60.0, 45.0)
        
        # Test overlap
        bbox2 = BoundingBox(x=60, y=45, width=100, height=50, confidence=0.8)
        assert bbox.overlaps(bbox2)
        
        # Test IoU
        iou = bbox.iou(bbox2)
        assert 0.0 < iou < 1.0

    def test_bounding_box_validation(self):
        """Test BoundingBox validation."""
        with pytest.raises(ValueError):
            BoundingBox(x=-10, y=20, width=100, height=50, confidence=0.9)
        
        with pytest.raises(ValueError):
            BoundingBox(x=10, y=20, width=0, height=50, confidence=0.9)

    def test_segment_comparison_result(self):
        """Test SegmentComparisonResult creation and validation."""
        res = SegmentComparisonResult(
            segment_id="seg_r0_c0",
            has_changes=True,
            change_type=ChangeType.VISUAL_CHANGE,
            diff_percentage=5.0,
            diff_pixel_count=100,
            metadata={"test": "data"}
        )
        
        assert res.is_significant(threshold=1.0)
        assert not res.is_significant(threshold=10.0)
        assert "Visual change" in res.get_change_summary()

    def test_segment_comparison_consistency(self):
        """Test SegmentComparisonResult consistency checks."""
        # Inconsistent: has_changes=True but change_type=NONE
        with pytest.raises(ValueError, match="Inconsistent state"):
            SegmentComparisonResult(
                segment_id="seg_r0_c0",
                has_changes=True,
                change_type=ChangeType.NONE,
                diff_percentage=0.0,
                diff_pixel_count=0
            )
            
        # Inconsistent: diff_percentage=0 but diff_pixel_count>0
        with pytest.raises(ValueError, match="Inconsistent: diff_percentage=0"):
            SegmentComparisonResult(
                segment_id="seg_r0_c0",
                has_changes=True,
                change_type=ChangeType.VISUAL_CHANGE,
                diff_percentage=0.0,
                diff_pixel_count=100
            )

    def test_document_comparison_result(self):
        """Test DocumentComparisonResult creation and methods."""
        # Mock segment results
        seg_results = [
            SegmentComparisonResult(
                segment_id="seg_1",
                has_changes=False,
                change_type=ChangeType.NONE,
                diff_percentage=0.0,
                diff_pixel_count=0
            ),
            SegmentComparisonResult(
                segment_id="seg_2",
                has_changes=True,
                change_type=ChangeType.VISUAL_CHANGE,
                diff_percentage=10.0,
                diff_pixel_count=500
            )
        ]
        
        # Mock validation result
        val_result = ValidationResult(
            is_consistent=True,
            dominant_shift=(0, 0),
            outlier_count=0,
            consistency_score=1.0,
            total_segments=2,
            valid_segments=2
        )
        
        doc_res = DocumentComparisonResult(
            overall_similarity=0.95,
            total_segments=2,
            segments_with_changes=1,
            change_percentage=50.0,
            segment_results=seg_results,
            validation_result=val_result,
            processing_time=1.5
        )
        
        assert doc_res.is_mostly_identical(threshold=0.9)
        assert doc_res.has_significant_changes(threshold=10.0)
        assert len(doc_res.get_changed_segments()) == 1
        
        stats = doc_res.get_summary_stats()
        assert stats['total_segments'] == 2
        assert stats['visual_changes'] == 1

    def test_document_comparison_consistency(self):
        """Test DocumentComparisonResult consistency checks."""
        seg_results = [
            SegmentComparisonResult(
                segment_id="seg_1",
                has_changes=False,
                change_type=ChangeType.NONE,
                diff_percentage=0.0,
                diff_pixel_count=0
            )
        ]
        
        val_result = ValidationResult(
            is_consistent=True,
            dominant_shift=(0, 0),
            outlier_count=0,
            consistency_score=1.0,
            total_segments=1,
            valid_segments=1
        )
        
        # Inconsistent segment count
        with pytest.raises(ValueError, match="Inconsistent segment count"):
            DocumentComparisonResult(
                overall_similarity=1.0,
                total_segments=5,  # Mismatch with len(seg_results)
                segments_with_changes=0,
                change_percentage=0.0,
                segment_results=seg_results,
                validation_result=val_result,
                processing_time=1.0
            )
