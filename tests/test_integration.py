"""
Integration tests for the ShiftVis core pipeline.
"""

import pytest
import numpy as np
import cv2
from src.core.preprocessor import Preprocessor
from src.core.segmentor import ImageSegmentor
from src.core.aligner import ImageAligner, AlignmentType
from src.core.validator import ConsistencyValidator
from src.core.comparator import (
    SegmentComparisonResult, DocumentComparisonResult, ChangeType
)

class TestCoreIntegration:
    """Integration tests for the core pipeline."""

    def test_full_pipeline_no_changes(self):
        """Test the full pipeline with identical images."""
        # 1. Preprocessor
        preprocessor = Preprocessor(min_dim=100)
        baseline_img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        test_img = baseline_img.copy()
        
        b_prep, t_prep = preprocessor.prepare_image_pair(baseline_img, test_img)
        
        # 2. Segmentor
        segmentor = ImageSegmentor(segment_size=100, overlap_percentage=0.0)
        segments = segmentor.segment_image(b_prep)
        assert len(segments) > 0
        
        # 3. Aligner
        aligner = ImageAligner()
        alignment_results = []
        segment_results = []
        
        for seg in segments:
            # Align
            alignment = aligner.find_best_alignment(t_prep, seg, b_prep)
            alignment_results.append(alignment)
            
            # Compare (mocking logic for now as per instructions)
            has_changes = not alignment.is_good_match()
            change_type = ChangeType.NONE if not has_changes else ChangeType.VISUAL_CHANGE
            
            # Ensure diff values match has_changes state
            diff_percentage = 0.5 if has_changes else 0.0
            diff_pixel_count = 10 if has_changes else 0
            
            seg_res = SegmentComparisonResult(
                segment_id=seg.segment_id,
                has_changes=has_changes,
                change_type=change_type,
                diff_percentage=diff_percentage,
                diff_pixel_count=diff_pixel_count,
                alignment_info=alignment,
                baseline_segment=seg
            )
            segment_results.append(seg_res)
            
        # 4. Validator
        validator = ConsistencyValidator()
        validation = validator.validate_vector_field(alignment_results)
        
        assert validation.is_consistent
        assert validation.dominant_shift == (0, 0)
        
        # 5. Comparator (Document Level)
        doc_result = DocumentComparisonResult(
            overall_similarity=1.0,
            total_segments=len(segments),
            segments_with_changes=0,
            change_percentage=0.0,
            segment_results=segment_results,
            validation_result=validation,
            processing_time=0.5
        )
        
        assert doc_result.is_mostly_identical()
        assert not doc_result.has_significant_changes()

    def test_full_pipeline_with_shift(self):
        """Test the full pipeline with a global shift."""
        # 1. Preprocessor
        preprocessor = Preprocessor(min_dim=100)
        baseline_img = np.zeros((500, 500, 3), dtype=np.uint8)
        # Add pattern
        baseline_img[100:400, 100:400] = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Create shifted test image
        shift_x, shift_y = 10, 5
        test_img = np.zeros((500, 500, 3), dtype=np.uint8)
        test_img[100+shift_y:400+shift_y, 100+shift_x:400+shift_x] = baseline_img[100:400, 100:400]
        
        b_prep, t_prep = preprocessor.prepare_image_pair(baseline_img, test_img)
        
        # 2. Segmentor
        segmentor = ImageSegmentor(segment_size=100)
        segments = segmentor.segment_image(b_prep)
        
        # Filter out blank segments for this test to ensure robust alignment
        valid_segments = [s for s in segments if s.variance > 10.0]
        
        # 3. Aligner
        aligner = ImageAligner(local_search_radius=20)
        alignment_results = []
        
        for seg in valid_segments:
            alignment = aligner.find_best_alignment(t_prep, seg, b_prep)
            alignment_results.append(alignment)
            
        # 4. Validator
        validator = ConsistencyValidator()
        validation = validator.validate_vector_field(alignment_results)
        
        assert validation.is_consistent
        assert validation.dominant_shift == (shift_x, shift_y)
        assert validation.consistency_score > 0.8
