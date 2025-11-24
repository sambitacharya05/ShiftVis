import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.comparator import ImageComparator, ChangeType
from core.aligner import AlignmentResult, AlignmentType


@pytest.fixture
def comparator():
    """Create ImageComparator instance for testing."""
    return ImageComparator()


def create_alignment_result(
    alignment_type: AlignmentType,
    similarity_score: float,
    shift: tuple,
    search_radius_used: int,
    aligned_data: np.ndarray = None
) -> AlignmentResult:
    """
    Helper to create AlignmentResult with all required fields.
    
    This handles Pydantic validation by providing all necessary fields.
    """
    # Create minimal valid aligned_data if None
    if aligned_data is None and alignment_type != AlignmentType.NO_MATCH:
        aligned_data = np.zeros((256, 256), dtype=np.uint8)
    
    return AlignmentResult(
        alignment_type=alignment_type,
        similarity_score=similarity_score,
        shift=shift,
        search_radius_used=search_radius_used,
        aligned_data=aligned_data,
        # Add any other required fields from your AlignmentResult model
        # For example, if you have quality_metrics, add:
        # quality_metrics={}
    )


class TestChangeTypeClassification:
    """Test suite for _classify_change_type method."""
    
    def test_no_match_classification(self, comparator):
        """Test NO_MATCH classification when alignment fails."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.NO_MATCH,
            similarity_score=0.0,
            shift=(0, 0),
            search_radius_used=50,
            aligned_data=None
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=50.0,
            diff_pixel_count=32768
        )
        
        # Assert
        assert result == ChangeType.NO_MATCH, \
            f"Expected NO_MATCH but got {result.value}"
    
    def test_skipped_low_confidence(self, comparator):
        """Test SKIPPED classification for low confidence regions."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOW_CONFIDENCE,
            similarity_score=0.5,
            shift=(0, 0),
            search_radius_used=20,
            aligned_data=None
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=0.1,
            diff_pixel_count=65
        )
        
        # Assert
        assert result == ChangeType.SKIPPED, \
            f"Expected SKIPPED but got {result.value}"
    
    def test_none_perfect_match(self, comparator):
        """Test NONE classification for identical content."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.99,
            shift=(0, 0),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=0.05,
            diff_pixel_count=32
        )
        
        # Assert
        assert result == ChangeType.NONE, \
            f"Expected NONE for perfect match but got {result.value}"
    
    def test_none_below_minimum_threshold(self, comparator):
        """Test NONE when changes below min_change_pixels threshold."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.95,
            shift=(1, 1),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=0.15,
            diff_pixel_count=100
        )
        
        # Assert
        assert result == ChangeType.NONE, \
            f"Expected NONE for changes below threshold but got {result.value}"
    
    def test_position_shift_large_displacement(self, comparator):
        """Test POSITION_SHIFT for moved content."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LARGE_SHIFT,
            similarity_score=0.97,
            shift=(15, 20),
            search_radius_used=50,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=1.5,
            diff_pixel_count=1000
        )
        
        # Assert
        assert result == ChangeType.POSITION_SHIFT, \
            f"Expected POSITION_SHIFT for large displacement but got {result.value}"
    
    def test_position_shift_tier2_good_similarity(self, comparator):
        """Test POSITION_SHIFT for tier2 alignment with good similarity."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LARGE_SHIFT,
            similarity_score=0.93,
            shift=(25, 10),
            search_radius_used=50,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=2.5,
            diff_pixel_count=1600
        )
        
        # Assert
        assert result == ChangeType.POSITION_SHIFT, \
            f"Expected POSITION_SHIFT for tier2 with good similarity but got {result.value}"
    
    def test_visual_change_low_similarity(self, comparator):
        """Test VISUAL_CHANGE for different content (low similarity)."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.85,
            shift=(2, 1),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=5.5,
            diff_pixel_count=3500
        )
        
        # Assert
        assert result == ChangeType.VISUAL_CHANGE, \
            f"Expected VISUAL_CHANGE for low similarity but got {result.value}"
    
    def test_visual_change_high_diff_percentage(self, comparator):
        """Test VISUAL_CHANGE for high pixel differences."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.92,
            shift=(1, 0),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=10.0,
            diff_pixel_count=6500
        )
        
        # Assert
        assert result == ChangeType.VISUAL_CHANGE, \
            f"Expected VISUAL_CHANGE for high diff percentage but got {result.value}"
    
    def test_visual_change_shift_with_content_change(self, comparator):
        """Test VISUAL_CHANGE when content shifted AND changed."""
        # Arrange
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LARGE_SHIFT,
            similarity_score=0.88,
            shift=(12, 15),
            search_radius_used=50,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        # Act
        result = comparator._classify_change_type(
            alignment_result=alignment,
            diff_percentage=4.0,
            diff_pixel_count=2600
        )
        
        # Assert
        assert result == ChangeType.VISUAL_CHANGE, \
            f"Expected VISUAL_CHANGE for shifted+changed content but got {result.value}"
    
    def test_boundary_cases(self, comparator):
        """Test edge cases at threshold boundaries."""
        
        # Case 1: Exactly at tier1 threshold with minimal shift
        print("\n  Testing Case 1: Tier1 threshold boundary")
        alignment1 = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.98,
            shift=(0, 0),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        result1 = comparator._classify_change_type(alignment1, 0.5, 300)
        print(f"    Result: {result1.value} (expected: none)")
        assert result1 == ChangeType.NONE, \
            f"Case 1 failed: Expected NONE but got {result1.value}"
        
        # Case 2: Just above min_change_pixels
        print("  Testing Case 2: Just above min_change_pixels")
        alignment2 = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.90,
            shift=(3, 2),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        result2 = comparator._classify_change_type(alignment2, 2.0, 129)
        print(f"    Result: {result2.value} (expected: visual_change)")
        assert result2 == ChangeType.VISUAL_CHANGE, \
            f"Case 2 failed: Expected VISUAL_CHANGE but got {result2.value}"
        
        # Case 3: Exactly at shift magnitude threshold (5 pixels)
        print("  Testing Case 3: Shift magnitude threshold")
        alignment3 = create_alignment_result(
            alignment_type=AlignmentType.LARGE_SHIFT,
            similarity_score=0.95,
            shift=(3, 4),
            search_radius_used=50,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        result3 = comparator._classify_change_type(alignment3, 1.8, 1200)
        print(f"    Result: {result3.value} (expected: position_shift)")
        assert result3 == ChangeType.POSITION_SHIFT, \
            f"Case 3 failed: Expected POSITION_SHIFT but got {result3.value}"


class TestChangeClassificationIntegration:
    """Integration tests with actual image processing."""
    
    def test_identical_segments(self, comparator):
        """Test classification of identical image segments."""
        baseline = np.ones((256, 256, 3), dtype=np.uint8) * 128
        test = baseline.copy()
        
        diff_map, count, pct = comparator._calculate_pixel_diff(baseline, test)
        
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=1.0,
            shift=(0, 0),
            search_radius_used=20,
            aligned_data=test[:, :, 0] if test.ndim == 3 else test
        )
        
        change_type = comparator._classify_change_type(alignment, pct, count)
        
        print(f"\n  Identical segments: {count} changed pixels ({pct:.2f}%)")
        print(f"  Classification: {change_type.value}")
        
        assert change_type == ChangeType.NONE
        assert count == 0
        assert pct == 0.0
    
    def test_shifted_content(self, comparator):
        """Test classification of shifted but unchanged content."""
        baseline = np.zeros((256, 256, 3), dtype=np.uint8)
        baseline[50:100, 50:100] = 255
        
        test = np.zeros((256, 256, 3), dtype=np.uint8)
        test[65:115, 65:115] = 255
        
        diff_map, count, pct = comparator._calculate_pixel_diff(baseline, test)
        
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LARGE_SHIFT,
            similarity_score=0.98,
            shift=(15, 15),
            search_radius_used=50,
            aligned_data=test[:, :, 0] if test.ndim == 3 else test
        )
        
        change_type = comparator._classify_change_type(alignment, 1.5, 1000)
        
        print(f"\n  Shifted content: {count} changed pixels ({pct:.2f}%)")
        print(f"  Shift: (15, 15)")
        print(f"  Classification: {change_type.value}")
        
        assert change_type == ChangeType.POSITION_SHIFT
    
    def test_content_change(self, comparator):
        """Test classification of actual content changes."""
        baseline = np.zeros((256, 256, 3), dtype=np.uint8)
        baseline[50:100, 50:100] = [255, 0, 0]
        
        test = np.zeros((256, 256, 3), dtype=np.uint8)
        test[50:100, 50:100] = [0, 0, 255]
        
        diff_map, count, pct = comparator._calculate_pixel_diff(baseline, test)
        
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.80,
            shift=(0, 0),
            search_radius_used=20,
            aligned_data=test[:, :, 0] if test.ndim == 3 else test
        )
        
        change_type = comparator._classify_change_type(alignment, pct, count)
        
        print(f"\n  Content change: {count} changed pixels ({pct:.2f}%)")
        print(f"  Similarity: 0.80")
        print(f"  Classification: {change_type.value}")
        
        assert change_type == ChangeType.VISUAL_CHANGE
        assert pct > 5.0


class TestParameterValidation:
    """Test parameter validation and edge cases."""
    
    def test_zero_diff_percentage(self, comparator):
        """Test handling of zero diff percentage."""
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.99,
            shift=(0, 0),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        result = comparator._classify_change_type(alignment, 0.0, 0)
        assert result == ChangeType.NONE
    
    def test_exact_min_change_pixels(self, comparator):
        """Test behavior at exact min_change_pixels threshold."""
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.90,
            shift=(2, 1),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        min_threshold = comparator.min_change_pixels
        result = comparator._classify_change_type(alignment, 2.0, min_threshold)
        
        assert result == ChangeType.VISUAL_CHANGE
    
    def test_high_similarity_high_diff(self, comparator):
        """Test conflict: high similarity but high pixel diff."""
        alignment = create_alignment_result(
            alignment_type=AlignmentType.LOCAL_SHIFT,
            similarity_score=0.99,
            shift=(0, 0),
            search_radius_used=20,
            aligned_data=np.zeros((256, 256), dtype=np.uint8)
        )
        
        result = comparator._classify_change_type(alignment, 15.0, 10000)
        
        assert result == ChangeType.VISUAL_CHANGE


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING CHANGE TYPE CLASSIFICATION TESTS")
    print("=" * 70)
    
    pytest.main([
        __file__, 
        "-v",
        "-s",
        "--tb=short",
        "--color=yes"
    ])