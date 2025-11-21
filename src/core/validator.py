import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from .aligner import AlignmentResult, AlignmentType

@dataclass
class ValidationResult:
    """
    Result of a consistency check on a set of alignment results.
    
    Attributes:
        is_consistent (bool): Whether the vector field is considered consistent.
        dominant_shift (Tuple[int, int]): The calculated global shift vector.
        outlier_count (int): Number of segments flagged as outliers.
        consistency_score (float): A score (0-1) indicating how consistent the field is.
    """
    is_consistent: bool
    dominant_shift: Tuple[int, int]
    outlier_count: int
    consistency_score: float

class ConsistencyValidator:
    """
    Validates the global consistency of alignment results from multiple segments.
    """
    
    def __init__(self, outlier_threshold: float = 2.0):
        """
        Initialize the validator.
        
        Args:
            outlier_threshold (float): Z-score threshold for detecting outliers.
        """
        self.outlier_threshold = outlier_threshold

    def validate_vector_field(self, results: List[AlignmentResult]) -> ValidationResult:
        """
        Analyzes the field of shift vectors to determine global consistency.
        
        Args:
            results: List of AlignmentResult objects from all segments.
            
        Returns:
            ValidationResult containing consistency metrics.
        """
        # Filter valid shifts (exclude NO_MATCH and LOW_CONFIDENCE)
        valid_shifts = [
            r.shift for r in results 
            if r.alignment_type not in [AlignmentType.NO_MATCH, AlignmentType.LOW_CONFIDENCE]
        ]
        
        if not valid_shifts:
            return ValidationResult(False, (0, 0), 0, 0.0)
            
        shifts = np.array(valid_shifts)
        
        # Calculate median shift (robust to outliers)
        median_x = np.median(shifts[:, 0])
        median_y = np.median(shifts[:, 1])
        
        # Calculate Median Absolute Deviation (MAD)
        # MAD = median(|x - median(x)|)
        mad_x = np.median(np.abs(shifts[:, 0] - median_x))
        mad_y = np.median(np.abs(shifts[:, 1] - median_y))
        
        # Avoid division by zero
        mad_x = max(mad_x, 1e-6)
        mad_y = max(mad_y, 1e-6)
        
        # Calculate Z-scores based on MAD
        # Modified Z-score = 0.6745 * (x - median) / MAD
        z_scores_x = 0.6745 * np.abs(shifts[:, 0] - median_x) / mad_x
        z_scores_y = 0.6745 * np.abs(shifts[:, 1] - median_y) / mad_y
        
        # Identify outliers
        outliers_x = z_scores_x > self.outlier_threshold
        outliers_y = z_scores_y > self.outlier_threshold
        outliers = outliers_x | outliers_y
        
        outlier_count = np.sum(outliers)
        total_valid = len(valid_shifts)
        
        # Consistency score: percentage of inliers
        consistency_score = 1.0 - (outlier_count / total_valid)
        
        return ValidationResult(
            is_consistent=consistency_score > 0.7, # Threshold can be tuned
            dominant_shift=(int(median_x), int(median_y)),
            outlier_count=int(outlier_count),
            consistency_score=float(consistency_score)
        )
