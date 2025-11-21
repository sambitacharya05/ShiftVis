import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
from dataclasses import dataclass

from typing import Tuple, Optional
from enum import Enum

from .segmentor import Segment, ImageSegmentor
from .preprocessor import Preprocessor

class AlignmentType(Enum):
    """
    An enumeration representing the types of alignment that can occur during image comparison.

    Attributes:
        EXACT_MATCH: Indicates that the images are perfectly aligned with no shift.
        LOCAL_SHIFT: Indicates a small shift in alignment within the local search radius.
        LARGE_SHIFT: Indicates a larger shift in alignment beyond the local search radius.
        NO_MATCH: Indicates that no meaningful alignment could be found.
    """
    EXACT_MATCH = "exact_match"
    LOCAL_SHIFT = "local_shift"
    LARGE_SHIFT = "large_shift"
    NO_MATCH = "no_match"
    LOW_CONFIDENCE = "low_confidence"

@dataclass
class AlignmentResult:
    """
    A data class representing the result of an image alignment operation.

    Attributes:
        shift (Tuple[int, int]): The (x, y) pixel shift required to align the images.
        similarity_score (float): The similarity score (e.g., SSIM) between the aligned images.
        aligned_data (np.ndarray): The aligned image data as a NumPy array.
        alignment_type (AlignmentType): The type of alignment determined (e.g., EXACT_MATCH, LOCAL_SHIFT).
        search_level (int): The level of search performed (e.g., local or large radius).
        confidence (float): The confidence level of the alignment result, ranging from 0 to 1.
    """
    shift: Tuple[int, int]
    similarity_score: float
    aligned_data: np.ndarray
    alignment_type: AlignmentType
    search_level: int
    confidence: float

class ImageAligner:
    """
    A class for aligning images based on similarity metrics and search radii.

    Attributes:
        local_search_radius (int): The radius for local alignment searches, in pixels.
        large_search_radius (int): The radius for larger alignment searches, in pixels.
        tier1_similarity_threshold (float): The similarity threshold for exact or near-exact matches.
        tier2_similarity_threshold (float): The similarity threshold for approximate matches.
    """

    def __init__(
            self,
            local_search_radius: int = 20,
            large_search_radius: int = 50,
            tier1_similarity_threshold: float = 0.95,
            tier2_similarity_threshold: float = 0.90,
            entropy_threshold: float = 4.0
    ):
        """
        Initializes the ImageAligner with specified search radii and similarity thresholds.

        Args:
            local_search_radius (int, optional): The radius for local alignment searches. Defaults to 20.
            large_search_radius (int, optional): The radius for larger alignment searches. Defaults to 50.
            tier1_similarity_threshold (float, optional): The similarity threshold for exact or near-exact matches. Defaults to 0.95.
            tier2_similarity_threshold (float, optional): The similarity threshold for approximate matches. Defaults to 0.90.
            segmentor (ImageSegmentor, optional): An instance of ImageSegmentor for segmenting images. Defaults to a new ImageSegmentor instance.
        """
        self.local_search_radius = local_search_radius
        self.large_search_radius = large_search_radius
        self.tier1_similarity_threshold = tier1_similarity_threshold
        self.tier2_similarity_threshold = tier2_similarity_threshold
        self.entropy_threshold = entropy_threshold
        self.segmentor = ImageSegmentor()
        self.preprocessor = Preprocessor()

    def _calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculates the Structural Similarity Index (SSIM) between two images.

        This method compares two images and returns a similarity score between 0 and 1,
        where 1 indicates identical images. If the images have different shapes, it
        returns a similarity score of 0.0. The method supports both grayscale and color
        images.

        Args:
            img1 (np.ndarray): The first image as a NumPy array.
            img2 (np.ndarray): The second image as a NumPy array.

        Returns:
            float: The SSIM similarity score between the two images. Returns 0.0 if the
            images have mismatched shapes or if an error occurs during calculation.
        """
        if img1.shape != img2.shape:
            return 0.0
        try:
            if img1.ndim == 3:
                return ssim(img1, img2, channel_axis=2)
            else:
                return ssim(img1, img2)
        except Exception as e:
            print(f"⚠️ Error calculating SSIM: {e}")
            return 0.0

    def _is_valid_segment(self, image: np.ndarray, segment: Segment) -> bool:
        """
        Checks if a given segment is valid within the bounds of the image.

        This method ensures that the segment lies entirely within the dimensions
        of the provided image. A segment is considered valid if:
        - Its top-left corner (x, y) is within the image bounds.
        - Its bottom-right corner (x + width, y + height) does not exceed the image bounds.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            segment (Segment): The segment to validate, containing x, y, width, and height.

        Returns:
            bool: True if the segment is valid, False otherwise.
        """
        h, w = image.shape[:2]
        return (
            segment.x >= 0 and
            segment.y >= 0 and
            segment.x + segment.width <= w and
            segment.y + segment.height <= h
        )

    def _create_shifted_segment(
                    self,
                    segment: Segment,
                    dx: int,
                    dy: int
            ) -> Segment:
        """
        Creates a new segment by shifting the given segment by specified offsets.

        This method takes an existing segment and applies horizontal (dx) and vertical (dy)
        shifts to its position, returning a new segment with updated coordinates and bounding box.

        Args:
            segment (Segment): The original segment to be shifted. Contains properties such as
                               x, y, width, height, and segment_id.
            dx (int): The horizontal shift (in pixels) to be applied to the segment's x-coordinate.
            dy (int): The vertical shift (in pixels) to be applied to the segment's y-coordinate.

        Returns:
            Segment: A new segment with updated x, y coordinates and bounding box after applying the shift.
        """
        new_x = segment.x + dx
        new_y = segment.y + dy

        return Segment(
            segment_id=segment.segment_id,
            x=new_x,
            y=new_y,
            width=segment.width,
            height=segment.height,
            bbox=(new_x, new_y, new_x + segment.width, new_y + segment.height)
        )

    def _search_tier1(
            self,
            test_image: np.ndarray,
            baseline_segment: Segment,
            baseline_data: np.ndarray
            ) -> Optional[AlignmentResult]:
        """
        Performs a tier-1 search for image alignment by exploring a local search radius.

        This method attempts to align a baseline segment with a test image by shifting the segment
        within a small search radius. It calculates the similarity score for each shifted segment
        and returns the best alignment result if the score meets the tier-1 similarity threshold.

        Args:
            test_image (np.ndarray): The test image as a NumPy array.
            baseline_segment (Segment): The baseline segment to align, containing its position and dimensions.
            baseline_data (np.ndarray): The data of the baseline segment as a NumPy array.

        Returns:
            Optional[AlignmentResult]: The alignment result containing the best shift, similarity score,
            aligned data, alignment type, search level, and confidence. Returns None if no alignment
            meets the tier-1 similarity threshold.
        """
        best_score = 0.0
        best_shift = (0, 0)
        best_data = None

        for dy in range(-self.local_search_radius, self.local_search_radius + 1):
            for dx in range(-self.local_search_radius, self.local_search_radius + 1):
                shifted_segment : Segment = self._create_shifted_segment(baseline_segment, dx, dy)
                if not self._is_valid_segment(test_image, shifted_segment):
                    continue
                segment_data = self.segmentor.extract_segment_data(test_image, shifted_segment)
                score = self._calculate_similarity(baseline_data, segment_data)
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)
                    best_data = segment_data
                if best_score >= 0.999:
                    break
            if best_score >= 0.999:
                break
        if best_score >= self.tier1_similarity_threshold:
            if best_shift == (0, 0):
                alignment_type = AlignmentType.EXACT_MATCH
            else:
                alignment_type = AlignmentType.LOCAL_SHIFT
            confidence = best_score

            return AlignmentResult(
                shift = best_shift,
                similarity_score = best_score,
                aligned_data = best_data,
                alignment_type = alignment_type,
                search_level = 1,
                confidence = confidence
            )
        return None

    def _search_tier2(
            self,
            test_image: np.ndarray,
            baseline_segment: Segment,
            baseline_data: np.ndarray
        ) -> Optional[AlignmentResult]:
        """
        Performs a tier-2 search for image alignment by exploring a larger search radius.

        This method attempts to align a baseline segment with a test image by shifting the segment
        within a large search radius. It calculates the similarity score for each shifted segment
        and returns the best alignment result if the score meets the tier-2 similarity threshold.

        Args:
            test_image (np.ndarray): The test image as a NumPy array.
            baseline_segment (Segment): The baseline segment to align, containing its position and dimensions.
            baseline_data (np.ndarray): The data of the baseline segment as a NumPy array.

        Returns:
            Optional[AlignmentResult]: The alignment result containing the best shift, similarity score,
            aligned data, alignment type, search level, and confidence. Returns None if no alignment
            meets the tier-2 similarity threshold.
        """
        best_score = 0.0
        best_shift = (0, 0)
        best_data = None

        for dy in range(-self.large_search_radius, self.large_search_radius + 1):
            for dx in range(-self.large_search_radius, self.large_search_radius + 1):
                shifted_segment : Segment = self._create_shifted_segment(baseline_segment, dx, dy)
                if not self._is_valid_segment(test_image, shifted_segment):
                    continue
                segment_data = self.segmentor.extract_segment_data(test_image, shifted_segment)
                score = self._calculate_similarity(baseline_data, segment_data)
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)
                    best_data = segment_data
                if best_score >= 0.999:
                    break
            if best_score >= 0.999:
                break

        if best_score >= self.tier2_similarity_threshold:
            alignment_type = AlignmentType.LARGE_SHIFT
            confidence = best_score

            return AlignmentResult(
                shift = best_shift,
                similarity_score = best_score,
                aligned_data = best_data,
                alignment_type = alignment_type,
                search_level = 2,
                confidence = confidence
            )
        return None

    def find_best_alignment(
            self,
            test_image: np.ndarray,
            baseline_segment: Segment,
            baseline_image: np.ndarray
        ) -> AlignmentResult:
        """
         Finds the best alignment for a baseline segment within a test image.

        This method attempts to align a baseline segment from a baseline image with a test image
        by performing a tiered search. It first performs a tier-1 search (local search radius) and,
        if no satisfactory alignment is found, proceeds to a tier-2 search (larger search radius).
        If no alignment meets the thresholds, it returns a default result indicating no match.

        Args:
            test_image (np.ndarray): The test image as a NumPy array.
            baseline_segment (Segment): The baseline segment to align, containing its position and dimensions.
            baseline_image (np.ndarray): The baseline image as a NumPy array.

        Returns:
            AlignmentResult: The result of the alignment operation, including the best shift, similarity score,
            aligned data, alignment type, search level, and confidence. If no alignment is found, the result
            indicates no match.
        """
        # Check for low entropy (blank/solid color segments)
        if baseline_segment.entropy < self.entropy_threshold:
            return AlignmentResult(
                shift=(0, 0),
                similarity_score=0.0,
                aligned_data=None,
                alignment_type=AlignmentType.LOW_CONFIDENCE,
                search_level=0,
                confidence=0.0
            )

        baseline_data = self.segmentor.extract_segment_data(baseline_image, baseline_segment)
        result = self._search_tier1(test_image, baseline_segment, baseline_data)
        if result:
            return result
        result = self._search_tier2(test_image, baseline_segment, baseline_data)
        if result:
            return result
        return AlignmentResult(
            shift=(0, 0),
            similarity_score=0.0,
            aligned_data=baseline_data,
            alignment_type=AlignmentType.NO_MATCH,
            search_level=2,
            confidence=0.0
        )


# TESTING CODE
if __name__ == "__main__":
    from time import time
    print("=" * 60)
    print("STEP 5: Complete Search Algorithm Tests")
    print("=" * 60)
    start_time = time()
    print(start_time)

    # Create aligner with default settings
    aligner = ImageAligner(
        local_search_radius=20,
        large_search_radius=50,
        tier1_similarity_threshold=0.95,
        tier2_similarity_threshold=0.90
    )

    # Create test images
    np.random.seed(42)  # For reproducibility
    baseline = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
    print(f"Baseline shape: {baseline.shape}")  # (1000, 800, 3)

    # Test segment
    seg = Segment(
        segment_id="test_seg",
        x=200, y=200,
        width=256, height=256,
        bbox=(200, 200, 456, 456)
    )

    # Test 1: Exact match (no shift)
    print("\n=== Test 1: Exact Match (No Shift) ===")
    test = baseline.copy()
    result = aligner.find_best_alignment(test, seg, baseline)
    print(f"✓ Shift: {result.shift}")  # Should be (0, 0)
    print(f"✓ Score: {result.similarity_score:.4f}")  # Should be ~1.0
    print(f"✓ Type: {result.alignment_type}")  # Should be EXACT_MATCH
    print(f"✓ Level: {result.search_level}")  # Should be 1
    assert result.alignment_type == AlignmentType.EXACT_MATCH
    assert result.shift == (0, 0)

    # Test 2: Small shift (within local radius)
    print("\n=== Test 2: Small Shift (Tier 1) ===")
    test = baseline.copy()

    # Extract segment from baseline
    seg_data = baseline[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width].copy()

    # Define shift
    shift_dx = 10
    shift_dy = -5
    new_x = seg.x + shift_dx  # 210
    new_y = seg.y + shift_dy  # 195

    # Clear original position (make it different)
    test[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width] = 0

    # Paste at shifted position
    test[new_y:new_y + seg.height, new_x:new_x + seg.width] = seg_data

    result = aligner.find_best_alignment(test, seg, baseline)
    print(f"✓ Shift: {result.shift}")  # Should be (10, -5)
    print(f"✓ Score: {result.similarity_score:.4f}")
    print(f"✓ Type: {result.alignment_type}")  # Should be LOCAL_SHIFT
    print(f"✓ Level: {result.search_level}")  # Should be 1
    assert result.alignment_type == AlignmentType.LOCAL_SHIFT
    assert result.search_level == 1
    print(f"✓ Expected shift: ({shift_dx}, {shift_dy}), Got: {result.shift}")

    # Test 3: Large shift (beyond local, within large radius)
    print("\n=== Test 3: Large Shift (Tier 2) ===")
    test = baseline.copy()

    # Extract segment from baseline
    seg_data = baseline[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width].copy()

    # Define large shift
    shift_dx = 35
    shift_dy = -30
    new_x = seg.x + shift_dx  # 235
    new_y = seg.y + shift_dy  # 170

    # Clear original position
    test[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width] = 0

    # Paste at shifted position
    test[new_y:new_y + seg.height, new_x:new_x + seg.width] = seg_data

    result = aligner.find_best_alignment(test, seg, baseline)
    print(f"✓ Shift: {result.shift}")  # Should be (35, -30)
    print(f"✓ Score: {result.similarity_score:.4f}")
    print(f"✓ Type: {result.alignment_type}")  # Should be LARGE_SHIFT
    print(f"✓ Level: {result.search_level}")  # Should be 2
    assert result.alignment_type == AlignmentType.LARGE_SHIFT
    assert result.search_level == 2
    print(f"✓ Expected shift: ({shift_dx}, {shift_dy}), Got: {result.shift}")

    # Test 4: No match (completely different content)
    print("\n=== Test 4: No Match ===")
    test = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
    result = aligner.find_best_alignment(test, seg, baseline)
    print(f"✓ Shift: {result.shift}")  # Will be (0, 0)
    print(f"✓ Score: {result.similarity_score:.4f}")  # Should be very low
    print(f"✓ Type: {result.alignment_type}")  # Should be NO_MATCH
    print(f"✓ Level: {result.search_level}")  # Should be 2
    print(f"✓ Confidence: {result.confidence:.4f}")  # Should be 0.0
    assert result.alignment_type == AlignmentType.NO_MATCH
    assert result.confidence == 0.0

    # Test 5: Edge case - segment at boundary
    print("\n=== Test 5: Edge Case (Segment at Image Boundary) ===")
    seg_edge = Segment(
        segment_id="edge_seg",
        x=544, y=744,  # Right at bottom-right corner
        width=256, height=256,
        bbox=(544, 744, 800, 1000)
    )
    test = baseline.copy()
    result = aligner.find_best_alignment(test, seg_edge, baseline)
    print(f"✓ Shift: {result.shift}")  # Should be (0, 0)
    print(f"✓ Type: {result.alignment_type}")  # Should be EXACT_MATCH
    assert result.alignment_type == AlignmentType.EXACT_MATCH

    print("\n" + "=" * 60)
    print("✅ All Step 5 tests passed!")
    print("=" * 60)
    end_time = time()
    print(end_time)
    print(f"Total Time: {end_time - start_time:.2f} seconds")