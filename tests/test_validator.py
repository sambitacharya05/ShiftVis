"""
Tests for the ConsistencyValidator class.
"""

import pytest
import numpy as np
from src.core.validator import ConsistencyValidator, ValidationResult
from src.core.aligner import AlignmentResult, AlignmentType

class TestValidator:
    """Tests for ConsistencyValidator class."""

    def test_initialization(self):
        """Test initialization with default and custom parameters."""
        # Default
        validator = ConsistencyValidator()
        assert validator.outlier_threshold == 1.0
        assert validator.consistency_threshold == 0.7

        # Custom
        validator = ConsistencyValidator(
            outlier_threshold=0.5,
            consistency_threshold=0.9
        )
        assert validator.outlier_threshold == 0.5
        assert validator.consistency_threshold == 0.9

    def test_initialization_validation(self):
        """Test validation of initialization parameters."""
        with pytest.raises(ValueError, match="outlier_threshold must be positive"):
            ConsistencyValidator(outlier_threshold=0.0)
            
        with pytest.raises(ValueError, match="consistency_threshold must be 0-1"):
            ConsistencyValidator(consistency_threshold=1.5)

    def test_validate_consistent_field(self):
        """Test validation of a consistent vector field."""
        validator = ConsistencyValidator()
        
        # Create consistent results (shift around 5, -3)
        results = []
        for _ in range(20):
            res = AlignmentResult(
                shift=(5, -3),
                similarity_score=0.95,
                alignment_type=AlignmentType.LOCAL_SHIFT,
                search_level=1,
                confidence=0.95
            )
            results.append(res)
            
        validation = validator.validate_vector_field(results)
        
        assert validation.is_consistent
        assert validation.dominant_shift == (5, -3)
        assert validation.outlier_count == 0
        assert validation.consistency_score == 1.0

    def test_validate_field_with_outliers(self):
        """Test validation with some outliers."""
        validator = ConsistencyValidator(outlier_threshold=1.0)
        
        # 18 consistent results
        results = []
        for _ in range(18):
            res = AlignmentResult(
                shift=(0, 0),
                similarity_score=0.95,
                alignment_type=AlignmentType.EXACT_MATCH,
                search_level=1,
                confidence=0.95
            )
            results.append(res)
            
        # 2 outliers
        outlier1 = AlignmentResult(
            shift=(50, 50),
            similarity_score=0.8,
            alignment_type=AlignmentType.LARGE_SHIFT,
            search_level=2,
            confidence=0.8
        )
        outlier2 = AlignmentResult(
            shift=(-50, -50),
            similarity_score=0.8,
            alignment_type=AlignmentType.LARGE_SHIFT,
            search_level=2,
            confidence=0.8
        )
        results.extend([outlier1, outlier2])
        
        validation = validator.validate_vector_field(results)
        
        assert validation.is_consistent  # 18/20 = 0.9 > 0.7
        assert validation.dominant_shift == (0, 0)
        assert validation.outlier_count == 2
        assert validation.consistency_score == 0.9

    def test_validate_inconsistent_field(self):
        """Test validation of a random/inconsistent field."""
        validator = ConsistencyValidator()
        
        # Random shifts
        results = []
        np.random.seed(42)
        for _ in range(20):
            shift_x = np.random.randint(-50, 50)
            shift_y = np.random.randint(-50, 50)
            res = AlignmentResult(
                shift=(shift_x, shift_y),
                similarity_score=0.9,
                alignment_type=AlignmentType.LOCAL_SHIFT,
                search_level=1,
                confidence=0.9
            )
            results.append(res)
            
        validation = validator.validate_vector_field(results)
        
        # Should be inconsistent (low consistency score)
        assert not validation.is_consistent
        assert validation.consistency_score < 0.7

    def test_validate_no_valid_shifts(self):
        """Test validation when no valid shifts exist."""
        validator = ConsistencyValidator()
        
        results = []
        for _ in range(5):
            res = AlignmentResult(
                shift=(0, 0),
                similarity_score=0.0,
                alignment_type=AlignmentType.NO_MATCH,
                search_level=2,
                confidence=0.0
            )
            results.append(res)
            
        validation = validator.validate_vector_field(results)
        
        assert not validation.is_consistent
        assert validation.valid_segments == 0
        assert validation.consistency_score == 0.0

    def test_mad_calculation(self):
        """Test MAD calculation logic."""
        validator = ConsistencyValidator()
        
        # Simple case: 1, 2, 3 -> median=2, deviations=1, 0, 1 -> MAD=1
        values = np.array([1, 2, 3])
        mad = validator._calculate_mad(values)
        assert mad == 1.0
        
        # With outlier: 1, 2, 3, 100 -> median=2.5, deviations=1.5, 0.5, 0.5, 97.5 -> median dev = 1.0
        values = np.array([1, 2, 3, 100])
        mad = validator._calculate_mad(values)
        assert mad == 1.0
