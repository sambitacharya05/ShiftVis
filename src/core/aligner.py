"""
Image alignment module for ShiftVis.

Provides alignment algorithms to find shifted segments between baseline and test images,
using multi-tier search strategies and similarity metrics.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
import cv2
from typing import Tuple, Optional
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from .segmentor import Segment, ImageSegmentor
from .preprocessor import Preprocessor


class AlignmentType(str, Enum):
    """
    Enumeration representing the types of alignment that can occur during image comparison.
    
    Inherits from str for better JSON serialization.
    
    Attributes:
        EXACT_MATCH: Images are perfectly aligned with no shift (dx=0, dy=0)
        LOCAL_SHIFT: Small shift within local search radius (high confidence)
        LARGE_SHIFT: Larger shift beyond local radius but within large radius
        NO_MATCH: No meaningful alignment found (similarity below threshold)
        LOW_CONFIDENCE: Segment has low entropy/variance (blank/uniform areas)
    """
    EXACT_MATCH = "exact_match"
    LOCAL_SHIFT = "local_shift"
    LARGE_SHIFT = "large_shift"
    NO_MATCH = "no_match"
    LOW_CONFIDENCE = "low_confidence"


class AlignmentResult(BaseModel):
    """
    Represents the result of an image alignment operation.
    
    Contains information about the shift required to align segments, similarity scores,
    and metadata about the alignment quality and search process.
    
    Attributes:
        shift: The (dx, dy) pixel shift required to align the images
        similarity_score: SSIM similarity score between aligned images (0-1)
        aligned_data: The aligned image data as NumPy array (optional for memory)
        alignment_type: Classification of alignment quality/type
        search_level: Level of search performed (1=local, 2=large, 0=skipped)
        confidence: Confidence level of alignment (0-1, typically same as similarity_score)
        
    Examples:
        >>> result = AlignmentResult(
        ...     shift=(10, -5),
        ...     similarity_score=0.95,
        ...     alignment_type=AlignmentType.LOCAL_SHIFT,
        ...     search_level=1,
        ...     confidence=0.95
        ... )
        >>> result.is_good_match()
        True
        >>> result.get_shift_magnitude()
        11.18
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=False
    )
    
    shift: Tuple[int, int] = Field(
        ...,
        description="Pixel shift (dx, dy) required to align images"
    )
    
    similarity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="SSIM similarity score (0=different, 1=identical)"
    )
    
    aligned_data: Optional[np.ndarray] = Field(
        None,
        description="Aligned image data (optional for memory efficiency)",
        exclude=True  # Exclude from JSON serialization
    )
    
    alignment_type: AlignmentType = Field(
        ...,
        description="Classification of alignment type"
    )
    
    search_level: int = Field(
        ...,
        ge=0,
        le=2,
        description="Search level performed (0=skipped, 1=local, 2=large)"
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence level of alignment result (0-1)"
    )
    
    @field_validator('shift')
    @classmethod
    def validate_shift_reasonable(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """
        Validate that shift values are within reasonable bounds.
        
        Prevents absurdly large shifts that indicate alignment failure.
        """
        dx, dy = v
        max_shift = 1000  # Maximum reasonable shift in pixels
        
        if abs(dx) > max_shift or abs(dy) > max_shift:
            raise ValueError(
                f"Shift ({dx}, {dy}) exceeds maximum reasonable shift of ±{max_shift} pixels"
            )
        
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self) -> 'AlignmentResult':
        """
        Validate logical consistency between fields.
        
        Ensures:
        - EXACT_MATCH implies zero shift
        - NO_MATCH implies zero confidence
        - LOW_CONFIDENCE implies zero similarity and confidence
        - High similarity scores have appropriate alignment types
        """
        # EXACT_MATCH must have zero shift
        if self.alignment_type == AlignmentType.EXACT_MATCH:
            if self.shift != (0, 0):
                raise ValueError(
                    f"EXACT_MATCH alignment must have shift=(0, 0), got {self.shift}"
                )
            if self.similarity_score < 0.95:
                raise ValueError(
                    f"EXACT_MATCH should have high similarity (>0.95), got {self.similarity_score}"
                )
        
        # NO_MATCH should have low/zero confidence
        if self.alignment_type == AlignmentType.NO_MATCH:
            if self.confidence > 0.5:
                raise ValueError(
                    f"NO_MATCH should have low confidence (≤0.5), got {self.confidence}"
                )
        
        # LOW_CONFIDENCE should have zero scores
        if self.alignment_type == AlignmentType.LOW_CONFIDENCE:
            if self.similarity_score > 0.0 or self.confidence > 0.0:
                raise ValueError(
                    f"LOW_CONFIDENCE should have zero scores, got "
                    f"similarity={self.similarity_score}, confidence={self.confidence}"
                )
        
        # Confidence should generally match similarity_score
        if self.alignment_type not in [AlignmentType.NO_MATCH, AlignmentType.LOW_CONFIDENCE]:
            if abs(self.confidence - self.similarity_score) > 0.1:
                raise ValueError(
                    f"Confidence ({self.confidence}) should approximately match "
                    f"similarity_score ({self.similarity_score})"
                )
        
        return self
    
    def is_good_match(self, threshold: float = 0.9) -> bool:
        """
        Check if alignment is a good match above threshold.
        
        Args:
            threshold: Minimum similarity score to consider good (default 0.9)
            
        Returns:
            bool: True if similarity exceeds threshold
            
        Examples:
            >>> result = AlignmentResult(..., similarity_score=0.95, ...)
            >>> result.is_good_match()
            True
            >>> result.is_good_match(threshold=0.98)
            False
        """
        return (
            self.alignment_type not in [AlignmentType.NO_MATCH, AlignmentType.LOW_CONFIDENCE] and
            self.similarity_score >= threshold
        )
    
    def get_shift_magnitude(self) -> float:
        """
        Calculate Euclidean magnitude of shift vector.
        
        Returns:
            float: Shift magnitude in pixels (√(dx² + dy²))
            
        Examples:
            >>> result = AlignmentResult(shift=(3, 4), ...)
            >>> result.get_shift_magnitude()
            5.0
            >>> result = AlignmentResult(shift=(10, -5), ...)
            >>> result.get_shift_magnitude()
            11.18
        """
        dx, dy = self.shift
        return float(np.sqrt(dx**2 + dy**2))
    
    def has_shift(self) -> bool:
        """
        Check if any shift was detected.
        
        Returns:
            bool: True if shift is non-zero
            
        Examples:
            >>> result = AlignmentResult(shift=(0, 0), ...)
            >>> result.has_shift()
            False
            >>> result = AlignmentResult(shift=(5, 0), ...)
            >>> result.has_shift()
            True
        """
        return self.shift != (0, 0)
    
    def to_dict_serializable(self) -> dict:
        """
        Convert to JSON-serializable dictionary (excludes numpy arrays).
        
        Returns:
            dict: Serializable representation
            
        Examples:
            >>> result = AlignmentResult(...)
            >>> data = result.to_dict_serializable()
            >>> import json
            >>> json_str = json.dumps(data)
        """
        data = self.model_dump(exclude={'aligned_data'})
        data['has_aligned_data'] = self.aligned_data is not None
        data['shift_magnitude'] = self.get_shift_magnitude()
        if self.aligned_data is not None:
            data['aligned_data_shape'] = self.aligned_data.shape
            data['aligned_data_dtype'] = str(self.aligned_data.dtype)
        return data


class ImageAligner:
    """
    A class for aligning images based on similarity metrics and multi-tier search.
    
    Implements a hierarchical search strategy:
    1. Tier 1: Local search (small radius, fast)
    2. Tier 2: Large search (extended radius, slower)
    3. Filters out low-quality segments (low entropy/variance)
    
    Attributes:
        local_search_radius: Radius for local alignment searches (pixels)
        large_search_radius: Radius for larger alignment searches (pixels)
        tier1_similarity_threshold: Threshold for exact/near-exact matches
        tier2_similarity_threshold: Threshold for approximate matches
        entropy_threshold: Minimum entropy for segment to be considered informative
        variance_threshold: Minimum variance for segment to be considered informative
        segmentor: ImageSegmentor instance for extracting segment data
        preprocessor: Preprocessor instance for image preprocessing
        
    Examples:
        >>> aligner = ImageAligner(
        ...     local_search_radius=20,
        ...     large_search_radius=50,
        ...     tier1_similarity_threshold=0.95
        ... )
        >>> result = aligner.find_best_alignment(test_img, segment, baseline_img)
        >>> print(f"Shift: {result.shift}, Score: {result.similarity_score}")
    """
    
    def __init__(
        self,
        local_search_radius: int = 20,
        large_search_radius: int = 50,
        tier1_similarity_threshold: float = 0.98,
        tier2_similarity_threshold: float = 0.92,
        entropy_threshold: float = 1.0,
        variance_threshold: float = 25.0
    ):
        """
        Initialize ImageAligner with search parameters.
        
        Args:
            local_search_radius: Radius for tier-1 local search (default 20 pixels)
            large_search_radius: Radius for tier-2 large search (default 50 pixels)
            tier1_similarity_threshold: Threshold for tier-1 matches (default 0.98)
            tier2_similarity_threshold: Threshold for tier-2 matches (default 0.92)
            entropy_threshold: Minimum entropy for informative segments (default 1.0)
            variance_threshold: Minimum variance for informative segments (default 25.0)
            
        Raises:
            ValueError: If parameters are out of valid ranges
        """
        if local_search_radius <= 0:
            raise ValueError(f"local_search_radius must be positive, got {local_search_radius}")
        if large_search_radius <= local_search_radius:
            raise ValueError(
                f"large_search_radius ({large_search_radius}) must be > "
                f"local_search_radius ({local_search_radius})"
            )
        if not 0.0 <= tier1_similarity_threshold <= 1.0:
            raise ValueError(
                f"tier1_similarity_threshold must be 0-1, got {tier1_similarity_threshold}"
            )
        if not 0.0 <= tier2_similarity_threshold <= 1.0:
            raise ValueError(
                f"tier2_similarity_threshold must be 0-1, got {tier2_similarity_threshold}"
            )
        if tier2_similarity_threshold >= tier1_similarity_threshold:
            raise ValueError(
                f"tier2_threshold ({tier2_similarity_threshold}) should be < "
                f"tier1_threshold ({tier1_similarity_threshold})"
            )
        
        self.local_search_radius = local_search_radius
        self.large_search_radius = large_search_radius
        self.tier1_similarity_threshold = tier1_similarity_threshold
        self.tier2_similarity_threshold = tier2_similarity_threshold
        self.entropy_threshold = entropy_threshold
        self.variance_threshold = variance_threshold
        self.segmentor = ImageSegmentor()
        self.preprocessor = Preprocessor()
    
    def _calculate_similarity(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Calculate Structural Similarity Index (SSIM) between two images.
        
        This method compares two images and returns a similarity score between 0 and 1,
        where 1 indicates identical images. Handles both grayscale and color images.
        
        Args:
            img1: First image as NumPy array
            img2: Second image as NumPy array
            
        Returns:
            float: SSIM similarity score (0-1). Returns 0.0 if shapes mismatch or error occurs.
            
        Examples:
            >>> img1 = np.random.rand(100, 100, 3)
            >>> img2 = img1.copy()
            >>> aligner._calculate_similarity(img1, img2)
            1.0
        """
        if img1.shape != img2.shape:
            return 0.0
        
        try:
            if img1.ndim == 3:
                # Color image - use channel_axis parameter
                return float(ssim(img1, img2, channel_axis=2))
            else:
                # Grayscale image
                return float(ssim(img1, img2))
        except Exception as e:
            print(f"⚠️ Error calculating SSIM: {e}")
            return 0.0
    
    def _is_valid_segment(self, image: np.ndarray, segment: Segment) -> bool:
        """
        Check if a segment is valid within image bounds.
        
        Ensures segment lies entirely within image dimensions.
        
        Args:
            image: Input image as NumPy array
            segment: Segment to validate (Pydantic model)
            
        Returns:
            bool: True if segment is valid, False otherwise
            
        Examples:
            >>> image = np.zeros((1000, 800, 3))
            >>> segment = Segment(..., x=100, y=200, width=256, height=256, ...)
            >>> aligner._is_valid_segment(image, segment)
            True
            >>> segment_invalid = Segment(..., x=900, y=900, width=256, height=256, ...)
            >>> aligner._is_valid_segment(image, segment_invalid)
            False
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
        Create a new segment by shifting the given segment.
        
        Applies horizontal (dx) and vertical (dy) shifts to segment position,
        returning a new Pydantic-validated Segment instance.
        
        Args:
            segment: Original segment to shift (Pydantic model)
            dx: Horizontal shift in pixels
            dy: Vertical shift in pixels
            
        Returns:
            Segment: New segment with updated position and bbox
            
        Examples:
            >>> original = Segment(..., x=100, y=200, width=256, height=256, ...)
            >>> shifted = aligner._create_shifted_segment(original, 10, -5)
            >>> shifted.x, shifted.y
            (110, 195)
        """
        new_x = segment.x + dx
        new_y = segment.y + dy
        
        # Create new Segment (Pydantic validates automatically)
        return Segment(
            segment_id=segment.segment_id,
            x=new_x,
            y=new_y,
            width=segment.width,
            height=segment.height,
            bbox=(new_x, new_y, new_x + segment.width, new_y + segment.height),
            entropy=segment.entropy,
            variance=segment.variance
        )
    
    def _search_tier1(
        self,
        test_image: np.ndarray,
        baseline_segment: Segment,
        baseline_data: np.ndarray
    ) -> Optional[AlignmentResult]:
        """
        Perform tier-1 local search for alignment.
        
        Explores a small search radius around the baseline segment position,
        looking for the best alignment. Returns result if similarity exceeds
        tier-1 threshold.
        
        Args:
            test_image: Test image to search in
            baseline_segment: Baseline segment to align (Pydantic model)
            baseline_data: Pixel data of baseline segment
            
        Returns:
            Optional[AlignmentResult]: Best alignment if found, None otherwise
            
        Examples:
            >>> result = aligner._search_tier1(test_img, segment, seg_data)
            >>> if result:
            ...     print(f"Found match at shift {result.shift}")
        """
        best_score = 0.0
        best_shift = (0, 0)
        best_data = None
        
        # Search in local radius
        for dy in range(-self.local_search_radius, self.local_search_radius + 1):
            for dx in range(-self.local_search_radius, self.local_search_radius + 1):
                # Check bounds before creating segment to avoid Pydantic validation error
                if baseline_segment.x + dx < 0 or baseline_segment.y + dy < 0:
                    continue

                # Create shifted segment (Pydantic validates)
                shifted_segment = self._create_shifted_segment(baseline_segment, dx, dy)
                
                # Check if shifted segment is within image bounds
                if not self._is_valid_segment(test_image, shifted_segment):
                    continue
                
                # Extract data at shifted position
                segment_data = self.segmentor.extract_segment_data(test_image, shifted_segment)
                
                # Calculate similarity
                score = self._calculate_similarity(baseline_data, segment_data)
                
                # Update best match
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)
                    best_data = segment_data
                
                # Early exit if perfect match found
                if best_score >= 0.999:
                    break
            
            if best_score >= 0.999:
                break
        
        # Check if score meets tier-1 threshold
        if best_score >= self.tier1_similarity_threshold:
            # Determine alignment type
            if best_shift == (0, 0):
                alignment_type = AlignmentType.EXACT_MATCH
            else:
                alignment_type = AlignmentType.LOCAL_SHIFT
            
            confidence = best_score
            
            # Return Pydantic-validated result
            return AlignmentResult(
                shift=best_shift,
                similarity_score=best_score,
                aligned_data=best_data,
                alignment_type=alignment_type,
                search_level=1,
                confidence=confidence
            )
        
        return None
    
    def _search_tier2(
        self,
        test_image: np.ndarray,
        baseline_segment: Segment,
        baseline_data: np.ndarray
    ) -> Optional[AlignmentResult]:
        """
        Perform tier-2 large search for alignment.
        
        Explores a larger search radius for segments that weren't found in tier-1.
        More computationally expensive but can find larger shifts.
        
        Args:
            test_image: Test image to search in
            baseline_segment: Baseline segment to align (Pydantic model)
            baseline_data: Pixel data of baseline segment
            
        Returns:
            Optional[AlignmentResult]: Best alignment if found, None otherwise
            
        Examples:
            >>> result = aligner._search_tier2(test_img, segment, seg_data)
            >>> if result:
            ...     print(f"Found large shift: {result.shift}")
        """
        best_score = 0.0
        best_shift = (0, 0)
        best_data = None
        
        # Search in large radius
        for dy in range(-self.large_search_radius, self.large_search_radius + 1):
            for dx in range(-self.large_search_radius, self.large_search_radius + 1):
                # Check bounds before creating segment to avoid Pydantic validation error
                if baseline_segment.x + dx < 0 or baseline_segment.y + dy < 0:
                    continue

                # Create shifted segment (Pydantic validates)
                shifted_segment = self._create_shifted_segment(baseline_segment, dx, dy)
                
                # Check if shifted segment is within image bounds
                if not self._is_valid_segment(test_image, shifted_segment):
                    continue
                
                # Extract data at shifted position
                segment_data = self.segmentor.extract_segment_data(test_image, shifted_segment)
                
                # Calculate similarity
                score = self._calculate_similarity(baseline_data, segment_data)
                
                # Update best match
                if score > best_score:
                    best_score = score
                    best_shift = (dx, dy)
                    best_data = segment_data
                
                # Early exit if perfect match found
                if best_score >= 0.999:
                    break
            
            if best_score >= 0.999:
                break
        
        # Check if score meets tier-2 threshold
        if best_score >= self.tier2_similarity_threshold:
            alignment_type = AlignmentType.LARGE_SHIFT
            confidence = best_score
            
            # Return Pydantic-validated result
            return AlignmentResult(
                shift=best_shift,
                similarity_score=best_score,
                aligned_data=best_data,
                alignment_type=alignment_type,
                search_level=2,
                confidence=confidence
            )
        
        return None
    
    def find_best_alignment(
        self,
        test_image: np.ndarray,
        baseline_segment: Segment,
        baseline_image: np.ndarray
    ) -> AlignmentResult:
        """
        Find the best alignment for a baseline segment within a test image.
        
        Implements hierarchical search strategy:
        1. Check if segment has sufficient information (entropy/variance)
        2. Try tier-1 local search (fast)
        3. If tier-1 fails, try tier-2 large search (slower)
        4. If both fail, return NO_MATCH
        
        Args:
            test_image: Test image to search in
            baseline_segment: Baseline segment to align (Pydantic model)
            baseline_image: Baseline image containing the segment
            
        Returns:
            AlignmentResult: Pydantic-validated alignment result (always returns a result)
            
        Examples:
            >>> aligner = ImageAligner()
            >>> result = aligner.find_best_alignment(test_img, segment, baseline_img)
            >>> if result.is_good_match():
            ...     print(f"Found match at {result.shift} with score {result.similarity_score}")
            ... else:
            ...     print(f"No good match: {result.alignment_type}")
        """
        # Check for low entropy/variance (blank/uniform segments)
        if (baseline_segment.entropy < self.entropy_threshold and 
            baseline_segment.variance < self.variance_threshold):
            return AlignmentResult(
                shift=(0, 0),
                similarity_score=0.0,
                aligned_data=None,
                alignment_type=AlignmentType.LOW_CONFIDENCE,
                search_level=0,
                confidence=0.0
            )
        
        # Extract baseline segment data
        baseline_data = self.segmentor.extract_segment_data(baseline_image, baseline_segment)
        
        # Try tier-1 (local search)
        result = self._search_tier1(test_image, baseline_segment, baseline_data)
        if result:
            return result
        
        # Try tier-2 (large search)
        result = self._search_tier2(test_image, baseline_segment, baseline_data)
        if result:
            return result
        
        # No match found - return NO_MATCH result
        return AlignmentResult(
            shift=(0, 0),
            similarity_score=0.0,
            aligned_data=baseline_data,
            alignment_type=AlignmentType.NO_MATCH,
            search_level=2,
            confidence=0.0
        )


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    from time import time
    
    print("=" * 70)
    print("PYDANTIC ALIGNMENT TESTS")
    print("=" * 70)
    start_time = time()
    
    # Test 1: Pydantic validation tests
    print("\n[Test 1] Pydantic AlignmentResult Validation...")
    
    # Valid result
    try:
        result = AlignmentResult(
            shift=(10, -5),
            similarity_score=0.95,
            alignment_type=AlignmentType.LOCAL_SHIFT,
            search_level=1,
            confidence=0.95
        )
        print(f"✅ Valid result created: {result.alignment_type}")
        print(f"   Shift magnitude: {result.get_shift_magnitude():.2f}")
        print(f"   Is good match: {result.is_good_match()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Invalid: EXACT_MATCH with non-zero shift
    try:
        invalid_result = AlignmentResult(
            shift=(10, 0),  # ❌ Non-zero shift
            similarity_score=0.98,
            alignment_type=AlignmentType.EXACT_MATCH,  # ❌ But claims exact match
            search_level=1,
            confidence=0.98
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught error: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    # Invalid: Shift too large
    try:
        invalid_result = AlignmentResult(
            shift=(2000, 0),  # ❌ Absurdly large shift
            similarity_score=0.5,
            alignment_type=AlignmentType.LARGE_SHIFT,
            search_level=2,
            confidence=0.5
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught large shift: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    print("\n" + "=" * 70)
    print("COMPLETE ALIGNMENT ALGORITHM TESTS")
    print("=" * 70)
    
    # Create aligner
    aligner = ImageAligner(
        local_search_radius=20,
        large_search_radius=50,
        tier1_similarity_threshold=0.95,
        tier2_similarity_threshold=0.90
    )
    
    # Create test images
    np.random.seed(42)
    baseline = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
    print(f"✓ Baseline shape: {baseline.shape}")
    
    # Test segment (Pydantic-validated)
    seg = Segment(
        segment_id="seg_r0_c0",
        x=200, y=200,
        width=256, height=256,
        bbox=(200, 200, 456, 456),
        entropy=6.5,
        variance=1200.0
    )
    print(f"✓ Created Pydantic segment: {seg.segment_id}")
    
    # Test 2: Exact match
    print("\n[Test 2] Exact Match (No Shift)...")
    test = baseline.copy()
    result = aligner.find_best_alignment(test, seg, baseline)
    print(f"✓ Shift: {result.shift}")
    print(f"✓ Score: {result.similarity_score:.4f}")
    print(f"✓ Type: {result.alignment_type}")
    print(f"✓ Level: {result.search_level}")
    print(f"✓ Confidence: {result.confidence:.4f}")
    assert result.alignment_type == AlignmentType.EXACT_MATCH
    assert result.shift == (0, 0)
    assert result.is_good_match()
    
    # Test 3: Small shift (tier 1)
    print("\n[Test 3] Small Shift (Tier 1)...")
    test = baseline.copy()
    seg_data = baseline[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width].copy()
    shift_dx, shift_dy = 10, -5
    new_x, new_y = seg.x + shift_dx, seg.y + shift_dy
    test[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width] = 0
    test[new_y:new_y + seg.height, new_x:new_x + seg.width] = seg_data
    
    result = aligner.find_best_alignment(test, seg, baseline)
    print(f"✓ Shift: {result.shift}")
    print(f"✓ Score: {result.similarity_score:.4f}")
    print(f"✓ Type: {result.alignment_type}")
    print(f"✓ Shift magnitude: {result.get_shift_magnitude():.2f}")
    assert result.alignment_type == AlignmentType.LOCAL_SHIFT
    assert result.search_level == 1
    assert result.has_shift()
    
    # Test 4: Large shift (tier 2)
    print("\n[Test 4] Large Shift (Tier 2)...")
    test = baseline.copy()
    seg_data = baseline[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width].copy()
    shift_dx, shift_dy = 35, -30
    new_x, new_y = seg.x + shift_dx, seg.y + shift_dy
    test[seg.y:seg.y + seg.height, seg.x:seg.x + seg.width] = 0
    test[new_y:new_y + seg.height, new_x:new_x + seg.width] = seg_data
    
    result = aligner.find_best_alignment(test, seg, baseline)
    print(f"✓ Shift: {result.shift}")
    print(f"✓ Score: {result.similarity_score:.4f}")
    print(f"✓ Type: {result.alignment_type}")
    print(f"✓ Shift magnitude: {result.get_shift_magnitude():.2f}")
    assert result.alignment_type == AlignmentType.LARGE_SHIFT
    assert result.search_level == 2
    
    # Test 5: No match
    print("\n[Test 5] No Match...")
    test = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)
    result = aligner.find_best_alignment(test, seg, baseline)
    print(f"✓ Type: {result.alignment_type}")
    print(f"✓ Score: {result.similarity_score:.4f}")
    print(f"✓ Confidence: {result.confidence:.4f}")
    assert result.alignment_type == AlignmentType.NO_MATCH
    assert result.confidence == 0.0
    assert not result.is_good_match()
    
    # Test 6: Low confidence segment
    print("\n[Test 6] Low Confidence (Low Entropy/Variance)...")
    low_conf_seg = Segment(
        segment_id="seg_r1_c1",
        x=400, y=400,
        width=256, height=256,
        bbox=(400, 400, 656, 656),
        entropy=0.5,  # ❌ Below threshold
        variance=10.0  # ❌ Below threshold
    )
    test = baseline.copy()
    result = aligner.find_best_alignment(test, low_conf_seg, baseline)
    print(f"✓ Type: {result.alignment_type}")
    print(f"✓ Search level: {result.search_level}")
    assert result.alignment_type == AlignmentType.LOW_CONFIDENCE
    assert result.search_level == 0
    
    # Test 7: JSON serialization
    print("\n[Test 7] JSON Serialization...")
    result = AlignmentResult(
        shift=(10, -5),
        similarity_score=0.95,
        alignment_type=AlignmentType.LOCAL_SHIFT,
        search_level=1,
        confidence=0.95
    )
    json_str = result.model_dump_json(exclude={'aligned_data'})
    print(f"✓ Serialized to JSON ({len(json_str)} chars)")
    print(f"   Preview: {json_str[:100]}...")
    
    loaded = AlignmentResult.model_validate_json(json_str)
    print(f"✓ Deserialized: shift={loaded.shift}, type={loaded.alignment_type}")
    
    # Test 8: to_dict_serializable
    data = result.to_dict_serializable()
    print(f"✓ Serializable dict keys: {list(data.keys())}")
    print(f"   Shift magnitude: {data['shift_magnitude']:.2f}")
    
    end_time = time()
    print("\n" + "=" * 70)
    print(f"✅ All Pydantic alignment tests passed!")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print("=" * 70)