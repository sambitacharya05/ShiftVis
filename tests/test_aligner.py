"""
Tests for the ImageAligner class.
"""

import pytest
import numpy as np
from src.core.aligner import ImageAligner, AlignmentResult, AlignmentType
from src.core.segmentor import Segment

class TestAligner:
    """Tests for ImageAligner class."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default
        aligner = ImageAligner()
        assert aligner.local_search_radius == 20
        assert aligner.large_search_radius == 50
        assert aligner.tier1_similarity_threshold == 0.98

        # Custom
        aligner = ImageAligner(
            local_search_radius=10,
            large_search_radius=30,
            tier1_similarity_threshold=0.95
        )
        assert aligner.local_search_radius == 10
        assert aligner.large_search_radius == 30
        assert aligner.tier1_similarity_threshold == 0.95

    def test_initialization_validation(self):
        """Test validation of initialization parameters."""
        with pytest.raises(ValueError, match="local_search_radius must be positive"):
            ImageAligner(local_search_radius=0)
            
        with pytest.raises(ValueError, match="large_search_radius .* must be >"):
            ImageAligner(local_search_radius=20, large_search_radius=20)
            
        with pytest.raises(ValueError, match="tier1_similarity_threshold must be 0-1"):
            ImageAligner(tier1_similarity_threshold=1.5)

    def test_calculate_similarity_identical(self):
        """Test similarity calculation for identical images."""
        aligner = ImageAligner()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        score = aligner._calculate_similarity(img, img)
        assert score == 1.0

    def test_calculate_similarity_different(self):
        """Test similarity calculation for different images."""
        aligner = ImageAligner()
        img1 = np.zeros((100, 100, 3))
        img2 = np.ones((100, 100, 3))
        score = aligner._calculate_similarity(img1, img2)
        assert score < 0.1

    def test_find_best_alignment_exact_match(self):
        """Test finding exact match."""
        aligner = ImageAligner()
        
        # Create baseline image and segment
        baseline = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        seg = Segment(
            segment_id="seg_r0_c0",
            x=100, y=100,
            width=100, height=100,
            bbox=(100, 100, 200, 200),
            entropy=5.0, variance=1000.0
        )
        
        # Test image is identical
        test = baseline.copy()
        
        result = aligner.find_best_alignment(test, seg, baseline)
        
        assert result.alignment_type == AlignmentType.EXACT_MATCH
        assert result.shift == (0, 0)
        assert result.similarity_score == 1.0
        assert result.is_good_match()

    def test_find_best_alignment_local_shift(self):
        """Test finding a small shift (Tier 1)."""
        aligner = ImageAligner(local_search_radius=10)
        
        # Create baseline
        baseline = np.zeros((500, 500, 3), dtype=np.uint8)
        # Add pattern
        baseline[100:200, 100:200] = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        seg = Segment(
            segment_id="seg_r0_c0",
            x=100, y=100,
            width=100, height=100,
            bbox=(100, 100, 200, 200),
            entropy=5.0, variance=1000.0
        )
        
        # Create test with shift
        shift_x, shift_y = 5, -3
        test = np.zeros((500, 500, 3), dtype=np.uint8)
        test[100+shift_y:200+shift_y, 100+shift_x:200+shift_x] = baseline[100:200, 100:200]
        
        result = aligner.find_best_alignment(test, seg, baseline)
        
        assert result.alignment_type == AlignmentType.LOCAL_SHIFT
        assert result.shift == (shift_x, shift_y)
        assert result.similarity_score > 0.99

    def test_find_best_alignment_large_shift(self):
        """Test finding a large shift (Tier 2)."""
        aligner = ImageAligner(
            local_search_radius=5,
            large_search_radius=30,
            tier2_similarity_threshold=0.9
        )
        
        # Create baseline
        baseline = np.zeros((500, 500, 3), dtype=np.uint8)
        baseline[100:200, 100:200] = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        seg = Segment(
            segment_id="seg_r0_c0",
            x=100, y=100,
            width=100, height=100,
            bbox=(100, 100, 200, 200),
            entropy=5.0, variance=1000.0
        )
        
        # Create test with large shift (outside local radius)
        shift_x, shift_y = 20, 15
        test = np.zeros((500, 500, 3), dtype=np.uint8)
        test[100+shift_y:200+shift_y, 100+shift_x:200+shift_x] = baseline[100:200, 100:200]
        
        result = aligner.find_best_alignment(test, seg, baseline)
        
        assert result.alignment_type == AlignmentType.LARGE_SHIFT
        assert result.shift == (shift_x, shift_y)
        assert result.search_level == 2

    def test_find_best_alignment_no_match(self):
        """Test when no match is found."""
        aligner = ImageAligner()
        
        baseline = np.zeros((500, 500, 3), dtype=np.uint8)
        baseline[100:200, 100:200] = 255  # White square
        
        seg = Segment(
            segment_id="seg_r0_c0",
            x=100, y=100,
            width=100, height=100,
            bbox=(100, 100, 200, 200),
            entropy=5.0, variance=1000.0
        )
        
        # Test image is completely black (no match)
        test = np.zeros((500, 500, 3), dtype=np.uint8)
        
        result = aligner.find_best_alignment(test, seg, baseline)
        
        assert result.alignment_type == AlignmentType.NO_MATCH
        assert result.similarity_score < 0.1
        assert result.confidence == 0.0

    def test_low_confidence_segment(self):
        """Test skipping low confidence (blank) segments."""
        aligner = ImageAligner(entropy_threshold=1.0, variance_threshold=10.0)
        
        baseline = np.zeros((500, 500, 3), dtype=np.uint8)
        
        # Blank segment
        seg = Segment(
            segment_id="seg_r0_c0",
            x=100, y=100,
            width=100, height=100,
            bbox=(100, 100, 200, 200),
            entropy=0.0, variance=0.0
        )
        
        test = baseline.copy()
        
        result = aligner.find_best_alignment(test, seg, baseline)
        
        assert result.alignment_type == AlignmentType.LOW_CONFIDENCE
        assert result.search_level == 0
