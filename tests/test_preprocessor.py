"""
Tests for the Preprocessor class.
"""

import pytest
import numpy as np
import cv2
from src.core.preprocessor import Preprocessor

class TestPreprocessor:
    """Tests for Preprocessor class."""

    def test_initialization(self):
        """Test default and custom initialization."""
        # Default
        prep = Preprocessor()
        assert prep.target_color_mode == "BGR"
        assert prep.min_dim == 256
        assert prep.max_dim == 10000
        assert prep.allow_resize is True

        # Custom
        prep = Preprocessor(
            target_color_mode="GRAY",
            min_dim=100,
            max_dim=5000,
            allow_resize=False
        )
        assert prep.target_color_mode == "GRAY"
        assert prep.min_dim == 100
        assert prep.max_dim == 5000
        assert prep.allow_resize is False

    def test_prepare_single_image_valid(self):
        """Test preparing a valid image."""
        prep = Preprocessor()
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        
        result = prep.prepare_single_image(img)
        assert result.shape == (500, 500, 3)
        assert result.dtype == np.uint8

    def test_prepare_single_image_grayscale_conversion(self):
        """Test converting grayscale to BGR."""
        prep = Preprocessor(target_color_mode="BGR")
        img = np.zeros((500, 500), dtype=np.uint8)  # Grayscale
        
        result = prep.prepare_single_image(img)
        assert result.ndim == 3
        assert result.shape == (500, 500, 3)

    def test_prepare_single_image_bgr_to_gray(self):
        """Test converting BGR to grayscale."""
        prep = Preprocessor(target_color_mode="GRAY")
        img = np.zeros((500, 500, 3), dtype=np.uint8)  # BGR
        
        result = prep.prepare_single_image(img)
        assert result.ndim == 2
        assert result.shape == (500, 500)

    def test_validation_none_image(self):
        """Test validation for None image."""
        prep = Preprocessor()
        with pytest.raises(ValueError, match="is None"):
            prep.prepare_single_image(None)

    def test_validation_empty_image(self):
        """Test validation for empty image."""
        prep = Preprocessor()
        img = np.array([])
        with pytest.raises(ValueError, match="is empty"):
            prep.prepare_single_image(img)

    def test_validation_dimensions(self):
        """Test validation for invalid dimensions (1D or 4D)."""
        prep = Preprocessor()
        img_1d = np.zeros((500,), dtype=np.uint8)
        with pytest.raises(ValueError, match="should be either 2D or 3D"):
            prep.prepare_single_image(img_1d)
            
        img_4d = np.zeros((1, 500, 500, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="should be either 2D or 3D"):
            prep.prepare_single_image(img_4d)

    def test_validation_size_constraints(self):
        """Test size constraints."""
        prep = Preprocessor(min_dim=100, max_dim=1000)
        
        # Too small
        img_small = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="smaller than the minimum"):
            prep.prepare_single_image(img_small)
            
        # Too large
        img_large = np.zeros((2000, 2000, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds the maximum"):
            prep.prepare_single_image(img_large)

    def test_prepare_image_pair_matching(self):
        """Test preparing a pair of images with matching dimensions."""
        prep = Preprocessor()
        baseline = np.zeros((500, 500, 3), dtype=np.uint8)
        test = np.zeros((500, 500, 3), dtype=np.uint8)
        
        b_res, t_res = prep.prepare_image_pair(baseline, test)
        assert b_res.shape == t_res.shape
        assert b_res.dtype == t_res.dtype

    def test_prepare_image_pair_resizing(self):
        """Test resizing test image to match baseline."""
        prep = Preprocessor(allow_resize=True)
        baseline = np.zeros((500, 500, 3), dtype=np.uint8)
        test = np.zeros((600, 600, 3), dtype=np.uint8)
        
        b_res, t_res = prep.prepare_image_pair(baseline, test)
        assert b_res.shape == (500, 500, 3)
        assert t_res.shape == (500, 500, 3)

    def test_prepare_image_pair_no_resize_fail(self):
        """Test failure when resizing is disabled and dimensions mismatch."""
        prep = Preprocessor(allow_resize=False)
        baseline = np.zeros((500, 500, 3), dtype=np.uint8)
        test = np.zeros((600, 600, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Image dimensions don't match"):
            prep.prepare_image_pair(baseline, test)

    def test_prepare_image_pair_mixed_modes(self):
        """Test handling mixed color modes in pairs."""
        prep = Preprocessor(target_color_mode="BGR")
        baseline = np.zeros((500, 500, 3), dtype=np.uint8)
        test = np.zeros((500, 500), dtype=np.uint8)  # Grayscale
        
        b_res, t_res = prep.prepare_image_pair(baseline, test)
        assert b_res.ndim == 3
        assert t_res.ndim == 3
        assert b_res.shape == t_res.shape

    def test_get_image_info(self):
        """Test getting image info."""
        prep = Preprocessor()
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        info = prep.get_image_info(img)
        
        assert info['height'] == 100
        assert info['width'] == 200
        assert info['channels'] == 3
        assert info['dtype'] == 'uint8'
        assert info['ndim'] == 3
