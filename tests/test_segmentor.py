"""
Comprehensive tests for the ImageSegmentor class.

Tests cover:
- Grid calculation logic
- Segment metadata generation
- Pixel data extraction
- Edge case handling
- RGB and grayscale image support
- Overlap behavior
"""

import numpy as np
import pytest
from src.core.segmentor import ImageSegmentor, Segment


class TestSegmentCalculation:
    """Test grid calculation logic."""
    
    def test_grid_calculation_512x512(self):
        """Test 512x512 image creates 3x3 grid."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        num_rows, num_cols = segmenter._calculate_grid(512, 512)
        assert num_rows == 3
        assert num_cols == 3
    
    def test_grid_calculation_1000x800(self):
        """Test 1000x800 image grid."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        num_rows, num_cols = segmenter._calculate_grid(800, 1000)
        assert num_rows == 4
        assert num_cols == 5
    
    def test_grid_calculation_small_image(self):
        """Test image smaller than segment size."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        num_rows, num_cols = segmenter._calculate_grid(200, 200)
        assert num_rows == 1
        assert num_cols == 1
    
    def test_grid_calculation_invalid_dimensions(self):
        """Test error handling for invalid dimensions."""
        segmenter = ImageSegmentor()
        with pytest.raises(ValueError, match=r"Image dimensions must be positive"):
            segmenter._calculate_grid(0, 512)
        with pytest.raises(ValueError, match=r"Image dimensions must be positive"):
            segmenter._calculate_grid(512, -10)
    
    def test_grid_calculation_no_overlap(self):
        """Test grid calculation with 0% overlap."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.0)
        num_rows, num_cols = segmenter._calculate_grid(512, 512)
        assert num_rows == 2
        assert num_cols == 2


class TestSegmentImage:
    """Test image segmentation logic."""
    
    def test_segment_count_512x512(self):
        """Test correct number of segments created."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        assert len(segments) == 9  # 3x3 grid
    
    def test_first_segment_metadata(self):
        """Test first segment has correct metadata."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        seg = segments[0]
        assert seg.segment_id == "seg_r0_c0"
        assert seg.x == 0
        assert seg.y == 0
        assert seg.width == 256
        assert seg.height == 256
        assert seg.bbox == (0, 0, 256, 256)
        assert seg.data is None
    
    def test_second_segment_position(self):
        """Test second segment has correct stride-based position."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # Second segment in first row
        seg = segments[1]
        assert seg.segment_id == "seg_r0_c1"
        assert seg.x == 231  # stride = 256 - 25 = 231
        assert seg.y == 0
        assert seg.width == 256
        assert seg.height == 256
        assert seg.bbox == (231, 0, 487, 256)
    
    def test_edge_segment_partial_width(self):
        """Test edge segment has partial width."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # Last segment in first row (seg_r0_c2)
        seg = segments[2]
        assert seg.segment_id == "seg_r0_c2"
        assert seg.x == 462  # 2 * 231
        assert seg.y == 0
        assert seg.width == 50  # min(256, 512-462) = 50
        assert seg.height == 256
        assert seg.bbox == (462, 0, 512, 256)
    
    def test_corner_segment_both_partial(self):
        """Test corner segment has partial width and height."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # Bottom-right corner (seg_r2_c2)
        seg = segments[8]
        assert seg.segment_id == "seg_r2_c2"
        assert seg.x == 462
        assert seg.y == 462
        assert seg.width == 50   # Partial!
        assert seg.height == 50  # Partial!
        assert seg.bbox == (462, 462, 512, 512)
    
    def test_grayscale_image(self):
        """Test segmentation works with grayscale images."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        assert len(segments) == 9
        
        # Verify first segment
        seg = segments[0]
        assert seg.segment_id == "seg_r0_c0"
        assert seg.x == 0
        assert seg.y == 0
    
    def test_segment_ids_unique(self):
        """Test that all segment IDs are unique."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        segment_ids = [seg.segment_id for seg in segments]
        assert len(segment_ids) == len(set(segment_ids))  # All unique
    
    def test_segment_ids_sequential(self):
        """Test that segment IDs follow expected row-column pattern."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        expected_ids = [
            "seg_r0_c0", "seg_r0_c1", "seg_r0_c2",
            "seg_r1_c0", "seg_r1_c1", "seg_r1_c2",
            "seg_r2_c0", "seg_r2_c1", "seg_r2_c2"
        ]
        actual_ids = [seg.segment_id for seg in segments]
        assert actual_ids == expected_ids


class TestExtractSegmentData:
    """Test segment data extraction."""
    
    def test_extract_correct_shape(self):
        """Test extracted data has correct shape."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        data = segmenter.extract_segment_data(image, segments[0])
        assert data.shape == (256, 256, 3)
    
    def test_extract_correct_region(self):
        """Test that extraction gets the correct image region."""
        segmenter = ImageSegmentor(segment_size=100, overlap_percentage=0.0)
        
        # Create image with unique values in each quadrant
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[0:100, 0:100, :] = [255, 0, 0]      # Top-left: RED
        image[0:100, 100:200, :] = [0, 255, 0]    # Top-right: GREEN
        image[100:200, 0:100, :] = [0, 0, 255]    # Bottom-left: BLUE
        image[100:200, 100:200, :] = [255, 255, 0]  # Bottom-right: YELLOW
        
        segments = segmenter.segment_image(image)
        
        # Verify each quadrant
        data_tl = segmenter.extract_segment_data(image, segments[0])
        assert np.array_equal(data_tl[0, 0], [255, 0, 0])  # RED
        
        data_tr = segmenter.extract_segment_data(image, segments[1])
        assert np.array_equal(data_tr[0, 0], [0, 255, 0])  # GREEN
        
        data_bl = segmenter.extract_segment_data(image, segments[2])
        assert np.array_equal(data_bl[0, 0], [0, 0, 255])  # BLUE
        
        data_br = segmenter.extract_segment_data(image, segments[3])
        assert np.array_equal(data_br[0, 0], [255, 255, 0])  # YELLOW
    
    def test_extract_partial_segment(self):
        """Test extraction of partial edge segment."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # Extract edge segment (partial width)
        data = segmenter.extract_segment_data(image, segments[2])
        assert data.shape == (256, 50, 3)  # Height full, width partial
    
    def test_extract_corner_segment(self):
        """Test extraction of corner segment (both dimensions partial)."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # Extract corner segment
        data = segmenter.extract_segment_data(image, segments[8])
        assert data.shape == (50, 50, 3)  # Both partial
    
    def test_extract_grayscale(self):
        """Test extraction works with grayscale images."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        data = segmenter.extract_segment_data(image, segments[0])
        assert data.shape == (256, 256)  # No channel dimension
    
    def test_extract_matches_original(self):
        """Test extracted data matches original image region."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
        segments = segmenter.segment_image(image)
        
        # Extract first segment
        data = segmenter.extract_segment_data(image, segments[0])
        
        # Should match original region
        expected = image[0:256, 0:256, :]
        assert np.array_equal(data, expected)
    
    def test_extract_middle_segment(self):
        """Test extraction of middle segment (tests correct indexing)."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.arange(512 * 512 * 3, dtype=np.uint8).reshape(512, 512, 3)
        segments = segmenter.segment_image(image)
        
        # Extract middle segment (seg_r1_c1)
        seg = segments[4]
        data = segmenter.extract_segment_data(image, seg)
        
        # Should match original region at (231, 231)
        expected = image[231:487, 231:487, :]
        assert np.array_equal(data, expected)


class TestOverlapBehavior:
    """Test segment overlap is working correctly."""
    
    def test_overlap_calculation(self):
        """Test overlap pixels and stride are calculated correctly."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        assert segmenter.overlap_pixels == 25
        assert segmenter.stride == 231
    
    def test_segments_overlap_horizontally(self):
        """Test that adjacent horizontal segments overlap by correct amount."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # First two segments in row
        seg0 = segments[0]  # x=0
        seg1 = segments[1]  # x=231
        
        # Check overlap
        seg0_end = seg0.x + seg0.width  # 0 + 256 = 256
        seg1_start = seg1.x              # 231
        overlap = seg0_end - seg1_start  # 256 - 231 = 25
        
        assert overlap == 25  # 10% of 256
    
    def test_segments_overlap_vertically(self):
        """Test that adjacent vertical segments overlap by correct amount."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # First two segments in column
        seg0 = segments[0]  # y=0 (row 0, col 0)
        seg3 = segments[3]  # y=231 (row 1, col 0)
        
        # Check overlap
        seg0_end = seg0.y + seg0.height  # 0 + 256 = 256
        seg3_start = seg3.y              # 231
        overlap = seg0_end - seg3_start  # 256 - 231 = 25
        
        assert overlap == 25  # 10% of 256
    
    def test_no_overlap(self):
        """Test segments with 0% overlap are adjacent without overlap."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.0)
        assert segmenter.overlap_pixels == 0
        assert segmenter.stride == 256
        
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # First two segments should be adjacent
        seg0 = segments[0]
        seg1 = segments[1]
        
        seg0_end = seg0.x + seg0.width
        seg1_start = seg1.x
        
        assert seg0_end == seg1_start  # No gap, no overlap
    
    def test_large_overlap(self):
        """Test segments with larger overlap percentage."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.25)
        assert segmenter.overlap_pixels == 64
        assert segmenter.stride == 192
        
        image = np.zeros((512, 512, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        # Check overlap
        seg0 = segments[0]
        seg1 = segments[1]
        
        seg0_end = seg0.x + seg0.width
        seg1_start = seg1.x
        overlap = seg0_end - seg1_start
        
        assert overlap == 64  # 25% of 256


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_segment_image(self):
        """Test image that fits in single segment."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        assert len(segments) == 1
        seg = segments[0]
        assert seg.width == 200
        assert seg.height == 200
    
    def test_exact_segment_size(self):
        """Test image that is exactly segment_size."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        assert len(segments) == 1
        seg = segments[0]
        assert seg.width == 256
        assert seg.height == 256
    
    def test_rectangular_image(self):
        """Test non-square rectangular image."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((300, 800, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        num_rows, num_cols = segmenter._calculate_grid(300, 800)
        assert len(segments) == num_rows * num_cols
    
    def test_very_small_image(self):
        """Test very small image."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        segments = segmenter.segment_image(image)
        
        assert len(segments) == 1
        seg = segments[0]
        assert seg.width == 50
        assert seg.height == 50
    
    def test_different_segment_sizes(self):
        """Test with different segment sizes."""
        for size in [64, 128, 256, 512]:
            segmenter = ImageSegmentor(segment_size=size, overlap_percentage=0.10)
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            segments = segmenter.segment_image(image)
            
            # All segments should exist
            assert len(segments) > 0
            
            # First segment should have correct size (or image size if smaller)
            seg = segments[0]
            assert seg.width == min(size, 512)
            assert seg.height == min(size, 512)


class TestInitialization:
    """Test ImageSegmentor initialization."""
    
    def test_default_parameters(self):
        """Test default initialization parameters."""
        segmenter = ImageSegmentor()
        assert segmenter.segment_size == 256
        assert segmenter.overlap_percentage == 0.10
        assert segmenter.overlap_pixels == 25
        assert segmenter.stride == 231
    
    def test_custom_parameters(self):
        """Test custom initialization parameters."""
        segmenter = ImageSegmentor(segment_size=128, overlap_percentage=0.20)
        assert segmenter.segment_size == 128
        assert segmenter.overlap_percentage == 0.20
        assert segmenter.overlap_pixels == 25  # 128 * 0.20 = 25.6 -> 25
        assert segmenter.stride == 103  # 128 - 25
    
    def test_zero_overlap(self):
        """Test initialization with zero overlap."""
        segmenter = ImageSegmentor(segment_size=256, overlap_percentage=0.0)
        assert segmenter.overlap_pixels == 0
        assert segmenter.stride == 256
