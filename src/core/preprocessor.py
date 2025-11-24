import logging
import numpy as np
import cv2

from .config import settings

logger = logging.getLogger(__name__)

class Preprocessor:
    """
    A class for preprocessing images, including validation, normalization, and resizing.

    Attributes:
        target_color_mode (str): The target color mode for the images ("BGR" or "GRAY").
        min_dim (int): The minimum allowed dimension (height or width) for the images.
        max_dim (int): The maximum allowed dimension (height or width) for the images.
        allow_resize (bool): Whether resizing is allowed for mismatched image dimensions.
    """

    def __init__(
            self,
            target_color_mode: str = settings.PREPROCESSOR_TARGET_COLOR_MODE,
            min_dim: int = settings.PREPROCESSOR_MIN_DIM,
            max_dim: int = settings.PREPROCESSOR_MAX_DIM,
            allow_resize: bool = settings.PREPROCESSOR_ALLOW_RESIZE
    ):
        """
        Initializes the Preprocessor with specified parameters.

        Args:
            target_color_mode (str, optional): The target color mode for the images. Defaults to "BGR".
            min_dim (int, optional): The minimum allowed dimension for the images. Defaults to 256.
            max_dim (int, optional): The maximum allowed dimension for the images. Defaults to 10000.
            allow_resize (bool, optional): Whether resizing is allowed. Defaults to True.
        """
        self.target_color_mode = target_color_mode
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.allow_resize = allow_resize

    def prepare_single_image(self, image: np.ndarray, image_name: str = "image") -> np.ndarray:
        """
        Prepares a single image by validating and normalizing it.

        The method performs the following steps:
        1. Validates that the image exists and is not empty.
        2. Validates the dimensions of the image.
        3. Validates that the image meets size constraints.
        4. Normalizes the color mode of the image.
        5. Applies light Gaussian blur to reduce rendering artifacts.
        6. Normalizes the data type of the image.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            image_name (str, optional): The name of the image (used for error messages). Defaults to "image".

        Returns:
            np.ndarray: The processed image.
        """
        self._validate_image_exists(image, image_name)
        self._validate_dimensions(image, image_name)
        self._validate_size_constraints(image, image_name)
        image = self._normalize_color_mode(image)
        image = self._apply_gaussian_blur(image)  # Smooth out rendering artifacts
        image = self._normalize_dtype(image)
        return image

    def _validate_image_exists(self, image: np.ndarray, image_name: str) -> None:
        """
        Validates that the image exists and is not empty.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            image_name (str): The name of the image (used for error messages).

        Raises:
            ValueError: If the image is None or empty.
        """
        if image is None:
            raise ValueError(f"The image '{image_name}' is None")

        if image.size == 0:
            raise ValueError(f"The image '{image_name}' is empty")

    def _validate_dimensions(self, image: np.ndarray, image_name: str) -> None:
        """
        Validates that the image has valid dimensions (2D or 3D) and supported channel counts.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            image_name (str): The name of the image (used for error messages).

        Raises:
            ValueError: If the image does not have 2D or 3D dimensions, or has unsupported channel count.
        """
        if image.ndim not in [2, 3]:
            raise ValueError(f"The image '{image_name}' should be either 2D or 3D.")
        
        # For 3D images, enforce supported channel counts (1 or 3 channels only)
        if image.ndim == 3:
            channels = image.shape[2]
            if channels not in [1, 3]:
                raise ValueError(
                    f"The image '{image_name}' has {channels} channels. "
                    f"Only grayscale (1 channel) or BGR (3 channels) images are supported. "
                    f"If this is an RGBA image, please convert it to BGR before processing."
                )

    def _validate_size_constraints(self, image: np.ndarray, image_name: str) -> None:
        """
        Validates that the image meets the size constraints.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            image_name (str): The name of the image (used for error messages).

        Raises:
            ValueError: If the image dimensions exceed the maximum or are smaller than the minimum allowed size.
        """
        height, width = image.shape[:2]
        if height > self.max_dim or width > self.max_dim:
            raise ValueError(f"The image '{image_name}' exceeds the maximum allowed dimensions of {self.max_dim}px.")
        elif height < self.min_dim or width < self.min_dim:
            raise ValueError(
                f"The image '{image_name}' is smaller than the minimum required dimensions of {self.min_dim}px.")

    def _normalize_color_mode(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes the color mode of the image to the target color mode.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The image converted to the target color mode.
        """
        if self.target_color_mode == "BGR":
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.ndim == 3:
                pass  # Already BGR
        elif self.target_color_mode == "GRAY":
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.ndim == 2:
                pass  # Already GRAY
        return image

    def _apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Applies a light Gaussian blur to reduce rendering artifacts.
        
        This smooths out:
        - Font anti-aliasing variations
        - Screenshot rendering differences  
        - Small pixel noise
        
        While preserving:
        - Edges and text boundaries
        - Real content changes
        
        Args:
            image (np.ndarray): The input image as a NumPy array.
            
        Returns:
            np.ndarray: The blurred image.
        """
        # Light blur: 3x3 kernel with sigma=1.0
        # Larger kernels or sigma would blur too much and lose real changes
        return cv2.GaussianBlur(image, (3, 3), sigmaX=1.0, sigmaY=1.0)

    def _normalize_dtype(self, image: np.ndarray) -> np.ndarray:
        """
        Normalizes the data type of the image to uint8.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            np.ndarray: The image with data type normalized to uint8.
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        return image

    def prepare_image_pair(
            self,
            baseline: np.ndarray,
            test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepares a pair of images by validating, normalizing, and resizing them.

        The method ensures that both images have the same dimensions, color mode, and data type.

        Args:
            baseline (np.ndarray): The baseline image as a NumPy array.
            test (np.ndarray): The test image as a NumPy array.

        Returns:
            tuple[np.ndarray, np.ndarray]: The processed baseline and test images.
        """
        baseline = self.prepare_single_image(baseline, "baseline")
        test = self.prepare_single_image(test, "test")
        baseline, test = self._match_color_modes(baseline, test)
        baseline, test = self._match_dimensions(baseline, test)
        self._validate_pair_compatibility(baseline, test)
        return baseline, test

    def _match_color_modes(self,
                           baseline: np.ndarray,
                           test: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
        """
        Matches the color modes of the baseline and test images.

        If one image is grayscale and the other is BGR, converts the grayscale image to BGR.

        Args:
            baseline (np.ndarray): The baseline image as a NumPy array.
            test (np.ndarray): The test image as a NumPy array.

        Returns:
            tuple[np.ndarray, np.ndarray]: The baseline and test images with matched color modes.
        """
        if baseline.ndim == test.ndim:
            return baseline, test

        if baseline.ndim == 2:
            baseline = cv2.cvtColor(baseline, cv2.COLOR_GRAY2BGR)
        if test.ndim == 2:
            test = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
        return baseline, test

    def _match_dimensions(self,
                          baseline: np.ndarray,
                          test: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
        """
        Matches the dimensions of the baseline and test images.

        If resizing is allowed, resizes the test image to match the baseline dimensions.

        Args:
            baseline (np.ndarray): The baseline image as a NumPy array.
            test (np.ndarray): The test image as a NumPy array.

        Returns:
            tuple[np.ndarray, np.ndarray]: The baseline and test images with matched dimensions.

        Raises:
            ValueError: If resizing is not allowed and the dimensions do not match.
        """
        bl_h, bl_w = baseline.shape[:2]
        t_h, t_w = test.shape[:2]

        if bl_h == t_h and bl_w == t_w:
            return baseline, test

        if not self.allow_resize:
            raise ValueError(
                f"Image dimensions don't match: baseline={bl_h}x{bl_w},  " f"test={t_h}x{t_w}. \nResize is disabled.")

        bl_area = bl_h * bl_w
        t_area = t_h * t_w

        if t_area > bl_area:
            # Downscaling: INTER_AREA is best
            interpolation = cv2.INTER_AREA
        else:
            # Upscaling: INTER_CUBIC for smooth results
            interpolation = cv2.INTER_CUBIC

        # Resize test to match baseline
        test = cv2.resize(test, (bl_w, bl_h), interpolation=interpolation)

        logger.warning(
            "Resized test image from %dx%d to %dx%d to match baseline",
            t_h, t_w, bl_h, bl_w
        )

        return baseline, test

    def _validate_pair_compatibility(self,
                                     baseline: np.ndarray,
                                     test: np.ndarray
                                     ) -> None:
        """
        Validates that the baseline and test images are compatible.

        Ensures that the images have the same shape and data type.

        Args:
            baseline (np.ndarray): The baseline image as a NumPy array.
            test (np.ndarray): The test image as a NumPy array.

        Raises:
            ValueError: If the images have incompatible shapes or data types.
        """
        if baseline.shape != test.shape:
            raise ValueError(
                f"Images have incompatible shapes after normalization: "
                f"baseline={baseline.shape}, test={test.shape}"
            )
        if baseline.dtype != test.dtype:
            raise ValueError(
                f"Images have incompatible dtypes: "
                f"baseline={baseline.dtype}, test={test.dtype}"
            )

    def get_image_info(self, image: np.ndarray) -> dict:
        """
        Retrieves information about an image.

        Args:
            image (np.ndarray): The input image as a NumPy array.

        Returns:
            dict: A dictionary containing image properties such as shape, dtype, size, and memory usage.
        """
        return {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "ndim": image.ndim,
            "size": image.size,
            "height": image.shape[0],
            "width": image.shape[1],
            "channels": image.shape[2] if image.ndim == 3 else 1,
            "memory_mb": round(image.nbytes / (1024 * 1024), 2)
        }

# TESTING CODE
if __name__ == "__main__":
    print("=" * 60)
    print("PART 1 TESTS: Individual Image Validation")
    print("=" * 60)

    # Test valid image
    print("\n=== Test 1: Valid BGR Image ===")
    preprocessor = Preprocessor()
    img = np.ones((256, 256, 3), dtype=np.uint8) * 100
    result = preprocessor.prepare_single_image(img, "test_image")
    print(f"✓ Result shape: {result.shape}")
    print(f"✓ Result dtype: {result.dtype}")

    # Test grayscale conversion
    print("\n=== Test 2: Grayscale to BGR ===")
    gray_img = np.ones((256, 256), dtype=np.uint8) * 100
    result = preprocessor.prepare_single_image(gray_img, "gray_image")
    print(f"✓ Input shape: {gray_img.shape}")
    print(f"✓ Output shape: {result.shape}")

    # Test None image (should raise error)
    print("\n=== Test 3: None Image (should fail) ===")
    try:
        preprocessor.prepare_single_image(None, "none_image")
        print("❌ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    # Test too small image (should raise error)
    print("\n=== Test 4: Too Small Image (should fail) ===")
    small_img = np.ones((100, 100, 3), dtype=np.uint8)
    try:
        preprocessor.prepare_single_image(small_img, "small_image")
        print("❌ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")

    print("\n" + "=" * 60)
    print("PART 2 TESTS: Pairwise Image Alignment")
    print("=" * 60)

    # Test 5: Same format images
    print("\n=== Test 5: Identical Format Images ===")
    baseline = np.ones((256, 256, 3), dtype=np.uint8) * 100
    test = np.ones((256, 256, 3), dtype=np.uint8) * 150
    b_norm, t_norm = preprocessor.prepare_image_pair(baseline, test)
    print(f"✓ Baseline shape: {b_norm.shape}")
    print(f"✓ Test shape: {t_norm.shape}")
    print(f"✓ Shapes match: {b_norm.shape == t_norm.shape}")

    # Test 6: Different sizes (should resize)
    print("\n=== Test 6: Different Sizes (Auto-resize) ===")
    baseline = np.ones((256, 256, 3), dtype=np.uint8) * 100
    test = np.ones((512, 512, 3), dtype=np.uint8) * 150
    b_norm, t_norm = preprocessor.prepare_image_pair(baseline, test)
    print(f"✓ Baseline shape: {b_norm.shape}")
    print(f"✓ Test shape after resize: {t_norm.shape}")
    print(f"✓ Shapes match: {b_norm.shape == t_norm.shape}")

    # Test 7: Mixed color modes
    print("\n=== Test 7: Mixed Color Modes (Gray + BGR) ===")
    baseline = np.ones((256, 256), dtype=np.uint8) * 100  # Grayscale
    test = np.ones((256, 256, 3), dtype=np.uint8) * 150  # BGR
    b_norm, t_norm = preprocessor.prepare_image_pair(baseline, test)
    print(f"✓ Baseline shape after conversion: {b_norm.shape}")
    print(f"✓ Test shape: {t_norm.shape}")
    print(f"✓ Both are 3-channel: {b_norm.ndim == 3 and t_norm.ndim == 3}")

    # Test 8: get_image_info
    print("\n=== Test 8: Image Info Utility ===")
    img = np.ones((512, 768, 3), dtype=np.uint8)
    info = preprocessor.get_image_info(img)
    print("✓ Image info:")
    for key, value in info.items():
        print(f"  - {key}: {value}")

    # Test 9: Resize disabled (should fail)
    print("\n=== Test 9: Resize Disabled (should fail) ===")
    preprocessor_strict = Preprocessor(allow_resize=False)
    baseline = np.ones((256, 256, 3), dtype=np.uint8)
    test = np.ones((512, 512, 3), dtype=np.uint8)
    try:
        preprocessor_strict.prepare_image_pair(baseline, test)
        print("❌ Should have raised ValueError!")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")