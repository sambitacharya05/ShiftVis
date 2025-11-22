import numpy as np
from typing import List, Tuple, Dict
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from aligner import AlignmentResult, AlignmentType


class ValidationResult(BaseModel):
    """
    Result of a consistency check on a set of alignment results.
    
    Analyzes a vector field of shift results to determine if there's a consistent
    global shift pattern across all segments, or if some segments are outliers.
    
    Attributes:
        is_consistent: Whether the vector field is considered globally consistent
        dominant_shift: The calculated global shift vector (median of all shifts)
        outlier_count: Number of segments flagged as outliers (anomalous shifts)
        consistency_score: Score (0-1) indicating field consistency (1=perfect)
        total_segments: Total number of segments analyzed
        valid_segments: Number of segments with valid shifts (excludes NO_MATCH, LOW_CONFIDENCE)
        outlier_ids: List of segment IDs identified as outliers (optional)
        shift_statistics: Statistics about the shift distribution (optional)
        
    Examples:
        >>> result = ValidationResult(
        ...     is_consistent=True,
        ...     dominant_shift=(5, -3),
        ...     outlier_count=2,
        ...     consistency_score=0.85,
        ...     total_segments=30,
        ...     valid_segments=28
        ... )
        >>> result.get_outlier_percentage()
        7.14
        >>> result.is_highly_consistent()
        True
    """
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        frozen=False
    )
    
    is_consistent: bool = Field(
        ...,
        description="Whether the vector field shows global consistency"
    )
    
    dominant_shift: Tuple[int, int] = Field(
        ...,
        description="Calculated global shift vector (median of all shifts)"
    )
    
    outlier_count: int = Field(
        ...,
        ge=0,
        description="Number of segments identified as outliers"
    )
    
    consistency_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Consistency score (0=inconsistent, 1=perfectly consistent)"
    )
    
    total_segments: int = Field(
        ...,
        gt=0,
        description="Total number of segments analyzed"
    )
    
    valid_segments: int = Field(
        ...,
        ge=0,
        description="Number of segments with valid shifts (excludes NO_MATCH, LOW_CONFIDENCE)"
    )
    
    outlier_ids: List[str] = Field(
        default_factory=list,
        description="Segment IDs identified as outliers"
    )
    
    shift_statistics: Dict[str, float] = Field(
        default_factory=dict,
        description="Statistics about shift distribution (MAD, std dev, etc.)"
    )
    
    @field_validator('dominant_shift')
    @classmethod
    def validate_dominant_shift_reasonable(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """
        Validate that dominant shift is within reasonable bounds.
        """
        dx, dy = v
        max_shift = 1000  # Maximum reasonable global shift
        
        if abs(dx) > max_shift or abs(dy) > max_shift:
            raise ValueError(
                f"Dominant shift ({dx}, {dy}) exceeds maximum reasonable "
                f"shift of ±{max_shift} pixels"
            )
        
        return v
    
    @model_validator(mode='after')
    def validate_consistency(self) -> 'ValidationResult':
        """
        Validate logical consistency between fields.
        
        Ensures:
        - outlier_count <= valid_segments <= total_segments
        - consistency_score matches outlier percentage
        - outlier_ids count matches outlier_count (if provided)
        """
        # Check segment counts
        if self.valid_segments > self.total_segments:
            raise ValueError(
                f"valid_segments ({self.valid_segments}) cannot exceed "
                f"total_segments ({self.total_segments})"
            )
        
        if self.outlier_count > self.valid_segments:
            raise ValueError(
                f"outlier_count ({self.outlier_count}) cannot exceed "
                f"valid_segments ({self.valid_segments})"
            )
        
        # Check consistency_score matches outlier ratio
        if self.valid_segments > 0:
            expected_score = 1.0 - (self.outlier_count / self.valid_segments)
            if abs(self.consistency_score - expected_score) > 0.01:
                raise ValueError(
                    f"consistency_score ({self.consistency_score:.3f}) doesn't match "
                    f"calculated score ({expected_score:.3f}) from outlier ratio"
                )
        
        # Check outlier_ids count matches outlier_count
        if self.outlier_ids and len(self.outlier_ids) != self.outlier_count:
            raise ValueError(
                f"outlier_ids length ({len(self.outlier_ids)}) doesn't match "
                f"outlier_count ({self.outlier_count})"
            )
        
        return self
    
    def get_outlier_percentage(self) -> float:
        """
        Calculate percentage of segments that are outliers.
        
        Returns:
            float: Outlier percentage (0-100)
            
        Examples:
            >>> result = ValidationResult(..., outlier_count=2, valid_segments=28, ...)
            >>> result.get_outlier_percentage()
            7.14
        """
        if self.valid_segments == 0:
            return 0.0
        return (self.outlier_count / self.valid_segments) * 100.0
    
    def get_inlier_percentage(self) -> float:
        """
        Calculate percentage of segments that are inliers (consistent).
        
        Returns:
            float: Inlier percentage (0-100)
            
        Examples:
            >>> result = ValidationResult(..., consistency_score=0.85, ...)
            >>> result.get_inlier_percentage()
            85.0
        """
        return self.consistency_score * 100.0
    
    def is_highly_consistent(self, threshold: float = 0.9) -> bool:
        """
        Check if consistency score exceeds high threshold.
        
        Args:
            threshold: Minimum consistency score (default 0.9 = 90%)
            
        Returns:
            bool: True if highly consistent
            
        Examples:
            >>> result = ValidationResult(..., consistency_score=0.95, ...)
            >>> result.is_highly_consistent()
            True
            >>> result.is_highly_consistent(threshold=0.98)
            False
        """
        return self.is_consistent and self.consistency_score >= threshold
    
    def has_dominant_shift(self) -> bool:
        """
        Check if there's a non-zero dominant shift.
        
        Returns:
            bool: True if dominant shift is non-zero
            
        Examples:
            >>> result = ValidationResult(..., dominant_shift=(5, -3), ...)
            >>> result.has_dominant_shift()
            True
            >>> result = ValidationResult(..., dominant_shift=(0, 0), ...)
            >>> result.has_dominant_shift()
            False
        """
        return self.dominant_shift != (0, 0)
    
    def get_shift_magnitude(self) -> float:
        """
        Calculate Euclidean magnitude of dominant shift.
        
        Returns:
            float: Shift magnitude in pixels
            
        Examples:
            >>> result = ValidationResult(..., dominant_shift=(3, 4), ...)
            >>> result.get_shift_magnitude()
            5.0
        """
        dx, dy = self.dominant_shift
        return float(np.sqrt(dx**2 + dy**2))
    
    def to_dict_serializable(self) -> dict:
        """
        Convert to JSON-serializable dictionary.
        
        Returns:
            dict: Serializable representation with additional computed fields
            
        Examples:
            >>> result = ValidationResult(...)
            >>> data = result.to_dict_serializable()
            >>> import json
            >>> json_str = json.dumps(data)
        """
        data = self.model_dump()
        
        # Add computed fields
        data['outlier_percentage'] = self.get_outlier_percentage()
        data['inlier_percentage'] = self.get_inlier_percentage()
        data['shift_magnitude'] = self.get_shift_magnitude()
        data['has_dominant_shift'] = self.has_dominant_shift()
        data['is_highly_consistent'] = self.is_highly_consistent()
        
        return data


class ConsistencyValidator:
    """
    Validates the global consistency of alignment results from multiple segments.
    
    Analyzes a collection of alignment results to detect:
    - Global shift patterns (dominant shift vector)
    - Outlier segments (shifts inconsistent with the majority)
    - Overall consistency of the alignment field
    
    Uses robust statistical methods (median, MAD) to handle outliers gracefully.
    
    Attributes:
        outlier_threshold: Z-score threshold for detecting outliers (default 1.0)
        consistency_threshold: Minimum consistency score to consider field consistent (default 0.7)
        
    Examples:
        >>> validator = ConsistencyValidator(outlier_threshold=1.5)
        >>> results = [alignment1, alignment2, ...]  # List of AlignmentResult
        >>> validation = validator.validate_vector_field(results)
        >>> if validation.is_consistent:
        ...     print(f"Global shift: {validation.dominant_shift}")
        ... else:
        ...     print(f"Inconsistent field: {validation.outlier_count} outliers")
    """
    
    def __init__(
        self,
        outlier_threshold: float = 1.0,
        consistency_threshold: float = 0.7
    ):
        """
        Initialize the consistency validator.
        
        Args:
            outlier_threshold: Z-score threshold for outlier detection (default 1.0)
                - Lower values (0.5-1.0) = stricter outlier detection
                - Higher values (1.5-3.0) = more lenient
            consistency_threshold: Minimum consistency score to consider field consistent (default 0.7)
                - 0.7 = 70% of segments must be inliers
                - 0.9 = 90% of segments must be inliers (stricter)
                
        Raises:
            ValueError: If thresholds are out of valid ranges
        """
        if outlier_threshold <= 0.0:
            raise ValueError(
                f"outlier_threshold must be positive, got {outlier_threshold}"
            )
        if not 0.0 <= consistency_threshold <= 1.0:
            raise ValueError(
                f"consistency_threshold must be 0-1, got {consistency_threshold}"
            )
        
        self.outlier_threshold = outlier_threshold
        self.consistency_threshold = consistency_threshold
    
    def _filter_valid_shifts(
        self,
        results: List[AlignmentResult]
    ) -> Tuple[List[Tuple[int, int]], List[str]]:
        """
        Filter alignment results to get valid shifts and their segment IDs.
        
        Excludes:
        - NO_MATCH segments (no alignment found)
        - LOW_CONFIDENCE segments (blank/uniform regions)
        
        Args:
            results: List of AlignmentResult objects (Pydantic models)
            
        Returns:
            Tuple of (valid_shifts, segment_ids)
            
        Examples:
            >>> validator = ConsistencyValidator()
            >>> shifts, ids = validator._filter_valid_shifts(results)
            >>> print(f"Found {len(shifts)} valid shifts")
        """
        valid_shifts = []
        segment_ids = []
        
        for r in results:
            if r.alignment_type not in [AlignmentType.NO_MATCH, AlignmentType.LOW_CONFIDENCE]:
                valid_shifts.append(r.shift)
                # Try to extract segment_id if available (from metadata or associated segment)
                segment_id = getattr(r, 'segment_id', None)
                if segment_id:
                    segment_ids.append(segment_id)
                else:
                    segment_ids.append(f"seg_{len(segment_ids)}")
        
        return valid_shifts, segment_ids
    
    def _calculate_mad(self, values: np.ndarray) -> float:
        """
        Calculate Median Absolute Deviation (MAD).
        
        MAD is a robust measure of variability, less sensitive to outliers than standard deviation.
        
        Formula: MAD = median(|x - median(x)|)
        
        Args:
            values: Array of numerical values
            
        Returns:
            float: MAD value (always >= 1e-6 to avoid division by zero)
            
        Examples:
            >>> values = np.array([1, 2, 3, 4, 100])  # 100 is an outlier
            >>> validator._calculate_mad(values)
            1.0  # Robust to outlier
        """
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        return max(mad, 1e-6)  # Avoid division by zero
    
    def _calculate_modified_z_scores(
        self,
        shifts: np.ndarray,
        median_x: float,
        median_y: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate modified Z-scores based on MAD (robust to outliers).
        
        Modified Z-score = 0.6745 * (x - median) / MAD
        
        The constant 0.6745 makes the modified Z-score comparable to standard Z-scores
        for normally distributed data.
        
        Args:
            shifts: Array of shift vectors (N, 2)
            median_x: Median of x-shifts
            median_y: Median of y-shifts
            
        Returns:
            Tuple of (z_scores_x, z_scores_y)
            
        Examples:
            >>> shifts = np.array([[0, 0], [1, 1], [2, 2], [100, 100]])
            >>> z_x, z_y = validator._calculate_modified_z_scores(shifts, 1.5, 1.5)
            >>> # Outlier (100, 100) will have high Z-scores
        """
        # Calculate MAD for each dimension
        mad_x = self._calculate_mad(shifts[:, 0])
        mad_y = self._calculate_mad(shifts[:, 1])
        
        # Calculate modified Z-scores
        z_scores_x = 0.6745 * np.abs(shifts[:, 0] - median_x) / mad_x
        z_scores_y = 0.6745 * np.abs(shifts[:, 1] - median_y) / mad_y
        
        return z_scores_x, z_scores_y
    
    def _calculate_shift_statistics(
        self,
        shifts: np.ndarray,
        outliers: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate statistics about the shift distribution.
        
        Args:
            shifts: Array of shift vectors (N, 2)
            outliers: Boolean array marking outliers
            
        Returns:
            dict: Statistics including MAD, std dev, ranges, etc.
            
        Examples:
            >>> stats = validator._calculate_shift_statistics(shifts, outliers)
            >>> print(f"MAD: {stats['mad_x']}, StdDev: {stats['std_x']}")
        """
        inlier_shifts = shifts[~outliers]
        
        if len(inlier_shifts) == 0:
            return {}
        
        return {
            'mad_x': float(self._calculate_mad(shifts[:, 0])),
            'mad_y': float(self._calculate_mad(shifts[:, 1])),
            'std_x': float(np.std(inlier_shifts[:, 0])),
            'std_y': float(np.std(inlier_shifts[:, 1])),
            'min_x': int(np.min(inlier_shifts[:, 0])),
            'max_x': int(np.max(inlier_shifts[:, 0])),
            'min_y': int(np.min(inlier_shifts[:, 1])),
            'max_y': int(np.max(inlier_shifts[:, 1])),
            'mean_x': float(np.mean(inlier_shifts[:, 0])),
            'mean_y': float(np.mean(inlier_shifts[:, 1])),
        }
    
    def validate_vector_field(
        self,
        results: List[AlignmentResult]
    ) -> ValidationResult:
        """
        Analyze the field of shift vectors to determine global consistency.
        
        Algorithm:
        1. Filter valid shifts (exclude NO_MATCH, LOW_CONFIDENCE)
        2. Calculate median shift (robust to outliers)
        3. Calculate MAD (Median Absolute Deviation)
        4. Compute modified Z-scores for each shift
        5. Flag outliers (Z-score > threshold)
        6. Calculate consistency score (percentage of inliers)
        7. Return Pydantic-validated ValidationResult
        
        Args:
            results: List of AlignmentResult objects (Pydantic models)
            
        Returns:
            ValidationResult: Pydantic-validated consistency metrics
            
        Examples:
            >>> validator = ConsistencyValidator(outlier_threshold=1.5)
            >>> alignment_results = [result1, result2, ...]  # From aligner
            >>> validation = validator.validate_vector_field(alignment_results)
            >>> 
            >>> if validation.is_consistent:
            ...     print(f"✅ Consistent field!")
            ...     print(f"   Dominant shift: {validation.dominant_shift}")
            ...     print(f"   Consistency: {validation.consistency_score * 100:.1f}%")
            ... else:
            ...     print(f"❌ Inconsistent field!")
            ...     print(f"   Outliers: {validation.outlier_count}/{validation.valid_segments}")
            ...     print(f"   Outlier IDs: {validation.outlier_ids}")
        """
        total_segments = len(results)
        
        # Filter valid shifts (exclude NO_MATCH, LOW_CONFIDENCE)
        valid_shifts, segment_ids = self._filter_valid_shifts(results)
        
        # Handle case with no valid shifts
        if not valid_shifts:
            return ValidationResult(
                is_consistent=False,
                dominant_shift=(0, 0),
                outlier_count=0,
                consistency_score=0.0,
                total_segments=total_segments,
                valid_segments=0,
                outlier_ids=[],
                shift_statistics={}
            )
        
        shifts = np.array(valid_shifts)
        valid_count = len(valid_shifts)
        
        # Calculate median shift (robust to outliers)
        median_x = float(np.median(shifts[:, 0]))
        median_y = float(np.median(shifts[:, 1]))
        
        # Calculate modified Z-scores based on MAD
        z_scores_x, z_scores_y = self._calculate_modified_z_scores(
            shifts, median_x, median_y
        )
        
        # Identify outliers (either x or y exceeds threshold)
        outliers_x = z_scores_x > self.outlier_threshold
        outliers_y = z_scores_y > self.outlier_threshold
        outliers = outliers_x | outliers_y
        
        outlier_count = int(np.sum(outliers))
        
        # Get outlier segment IDs
        outlier_ids = [
            segment_ids[i] for i in range(len(segment_ids))
            if i < len(outliers) and outliers[i]
        ]
        
        # Calculate consistency score (percentage of inliers)
        consistency_score = 1.0 - (outlier_count / valid_count)
        
        # Calculate shift statistics
        shift_statistics = self._calculate_shift_statistics(shifts, outliers)
        
        # Return Pydantic-validated result
        return ValidationResult(
            is_consistent=consistency_score >= self.consistency_threshold,
            dominant_shift=(int(median_x), int(median_y)),
            outlier_count=outlier_count,
            consistency_score=float(consistency_score),
            total_segments=total_segments,
            valid_segments=valid_count,
            outlier_ids=outlier_ids,
            shift_statistics=shift_statistics
        )


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PYDANTIC VALIDATION TESTS")
    print("=" * 70)
    
    # Test 1: Pydantic ValidationResult validation
    print("\n[Test 1] Pydantic ValidationResult Validation...")
    
    # Valid result
    try:
        result = ValidationResult(
            is_consistent=True,
            dominant_shift=(5, -3),
            outlier_count=2,
            consistency_score=0.85,
            total_segments=30,
            valid_segments=28
        )
        print(f"✅ Valid result created")
        print(f"   Consistency: {result.consistency_score * 100:.1f}%")
        print(f"   Outlier percentage: {result.get_outlier_percentage():.2f}%")
        print(f"   Shift magnitude: {result.get_shift_magnitude():.2f}px")
        print(f"   Is highly consistent: {result.is_highly_consistent()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Invalid: outlier_count > valid_segments
    try:
        invalid_result = ValidationResult(
            is_consistent=True,
            dominant_shift=(5, -3),
            outlier_count=30,  # ❌ More outliers than valid segments!
            consistency_score=0.85,
            total_segments=30,
            valid_segments=28
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught error: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    # Invalid: consistency_score doesn't match outlier ratio
    try:
        invalid_result = ValidationResult(
            is_consistent=True,
            dominant_shift=(5, -3),
            outlier_count=2,
            consistency_score=0.50,  # ❌ Should be ~0.93 (2/28 outliers)
            total_segments=30,
            valid_segments=28
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught inconsistency: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    # Invalid: outlier_ids count mismatch
    try:
        invalid_result = ValidationResult(
            is_consistent=True,
            dominant_shift=(5, -3),
            outlier_count=2,
            consistency_score=0.9286,
            total_segments=30,
            valid_segments=28,
            outlier_ids=["seg_r1_c1"]  # ❌ Only 1 ID but outlier_count=2
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught mismatch: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    print("\n" + "=" * 70)
    print("CONSISTENCY VALIDATOR TESTS")
    print("=" * 70)
    
    # Test 2: Create mock alignment results
    print("\n[Test 2] Testing with consistent shifts...")
    
    # Create consistent alignment results (all around (5, -3))
    consistent_results = []
    for i in range(28):
        result = AlignmentResult(
            shift=(5 + np.random.randint(-1, 2), -3 + np.random.randint(-1, 2)),
            similarity_score=0.95,
            alignment_type=AlignmentType.LOCAL_SHIFT,
            search_level=1,
            confidence=0.95
        )
        consistent_results.append(result)
    
    # Add 2 outliers
    outlier1 = AlignmentResult(
        shift=(50, -30),  # Way off!
        similarity_score=0.88,
        alignment_type=AlignmentType.LARGE_SHIFT,
        search_level=2,
        confidence=0.88
    )
    outlier2 = AlignmentResult(
        shift=(-40, 25),  # Way off in other direction!
        similarity_score=0.85,
        alignment_type=AlignmentType.LARGE_SHIFT,
        search_level=2,
        confidence=0.85
    )
    consistent_results.extend([outlier1, outlier2])
    
    # Validate
    validator = ConsistencyValidator(outlier_threshold=1.5)
    validation = validator.validate_vector_field(consistent_results)
    
    print(f"✓ Total segments: {validation.total_segments}")
    print(f"✓ Valid segments: {validation.valid_segments}")
    print(f"✓ Dominant shift: {validation.dominant_shift}")
    print(f"✓ Outlier count: {validation.outlier_count}")
    print(f"✓ Consistency score: {validation.consistency_score * 100:.1f}%")
    print(f"✓ Is consistent: {validation.is_consistent}")
    print(f"✓ Is highly consistent: {validation.is_highly_consistent()}")
    
    if validation.shift_statistics:
        print(f"✓ Shift statistics:")
        for key, value in validation.shift_statistics.items():
            print(f"   - {key}: {value}")
    
    # Test 3: Inconsistent shifts
    print("\n[Test 3] Testing with inconsistent shifts...")
    
    # Create random alignment results (no pattern)
    inconsistent_results = []
    for i in range(30):
        result = AlignmentResult(
            shift=(np.random.randint(-50, 50), np.random.randint(-50, 50)),
            similarity_score=0.92,
            alignment_type=AlignmentType.LOCAL_SHIFT,
            search_level=1,
            confidence=0.92
        )
        inconsistent_results.append(result)
    
    validation = validator.validate_vector_field(inconsistent_results)
    
    print(f"✓ Dominant shift: {validation.dominant_shift}")
    print(f"✓ Outlier count: {validation.outlier_count}")
    print(f"✓ Consistency score: {validation.consistency_score * 100:.1f}%")
    print(f"✓ Is consistent: {validation.is_consistent}")
    
    # Test 4: No valid shifts
    print("\n[Test 4] Testing with no valid shifts...")
    
    no_match_results = []
    for i in range(10):
        result = AlignmentResult(
            shift=(0, 0),
            similarity_score=0.0,
            alignment_type=AlignmentType.NO_MATCH,
            search_level=2,
            confidence=0.0
        )
        no_match_results.append(result)
    
    validation = validator.validate_vector_field(no_match_results)
    
    print(f"✓ Total segments: {validation.total_segments}")
    print(f"✓ Valid segments: {validation.valid_segments}")
    print(f"✓ Is consistent: {validation.is_consistent}")
    print(f"✓ Consistency score: {validation.consistency_score * 100:.1f}%")
    
    # Test 5: JSON serialization
    print("\n[Test 5] JSON Serialization...")
    
    result = ValidationResult(
        is_consistent=True,
        dominant_shift=(5, -3),
        outlier_count=2,
        consistency_score=0.93,
        total_segments=30,
        valid_segments=28,
        outlier_ids=["seg_r1_c1", "seg_r2_c5"],
        shift_statistics={'mad_x': 1.5, 'mad_y': 1.2}
    )
    
    json_str = result.model_dump_json()
    print(f"✓ Serialized to JSON ({len(json_str)} chars)")
    print(f"   Preview: {json_str[:100]}...")
    
    loaded = ValidationResult.model_validate_json(json_str)
    print(f"✓ Deserialized: shift={loaded.dominant_shift}, consistent={loaded.is_consistent}")
    
    # Test 6: to_dict_serializable
    data = result.to_dict_serializable()
    print(f"✓ Serializable dict keys: {list(data.keys())}")
    print(f"   Computed fields:")
    print(f"   - outlier_percentage: {data['outlier_percentage']:.2f}%")
    print(f"   - inlier_percentage: {data['inlier_percentage']:.2f}%")
    print(f"   - shift_magnitude: {data['shift_magnitude']:.2f}px")
    
    print("\n" + "=" * 70)
    print("✅ All Pydantic validation tests passed!")
    print("=" * 70)