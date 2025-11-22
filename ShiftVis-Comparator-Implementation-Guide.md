# ShiftVis ImageComparator Implementation Guide

## Overview

This guide provides a step-by-step implementation blueprint for the `ImageComparator` class - the orchestrator that ties together all ShiftVis components (Preprocessor, Segmentor, Aligner, Validator) to perform complete document comparison.

**Purpose**: Detect visual differences between two digitally-generated documents with pixel-level precision while handling layout shifts intelligently.

**Target Use Case**: Regression testing for digital documents (PDFs, web pages, UI screenshots) with special focus on Japanese character detection.

---

## Architecture Overview

```
Input: baseline_path, candidate_path
    ↓
Phase 1: Image Loading & Preprocessing (Preprocessor)
    ↓
Phase 2: Grid-based Segmentation (Segmentor)
    ↓
Phase 3: Per-Segment Alignment (Aligner)
    ↓
Phase 4: Consistency Validation (Validator)
    ↓
Phase 5: Decision Making (Accept/Reject/Revert)
    ↓
Phase 6: Per-Segment Pixel Comparison
    ↓
Phase 7: Change Detection & Classification
    ↓
Phase 8: Result Aggregation
    ↓
Output: DocumentComparisonResult
```

---

## Phase 1: Class Initialization & Configuration

### What to Build
An `ImageComparator` class that holds:
- Configuration parameters (thresholds, segment size, etc.)
- Component instances (Preprocessor, Segmentor, Aligner, Validator)
- State management for the comparison pipeline

### Template Structure
```python
class ImageComparator:
    def __init__(self, config_params):
        # Store configuration
        # Initialize component instances
        # Set comparison thresholds
```

### Why This Matters
- **Separation of Concerns**: Each component (Preprocessor, Segmentor, etc.) handles one responsibility
- **Configuration Management**: All thresholds in one place for easy tuning
- **Testability**: Can inject mock components for unit testing
- **Reusability**: Single instance can process multiple document pairs

### Configuration Parameters to Store
1. **Segmentation Config**:
   - `segment_size`: Grid cell size (256px recommended)
   - `overlap`: Percentage overlap between segments (10% = 25.6px)

2. **Alignment Config**:
   - `tier1_threshold`: Local alignment SSIM threshold (0.98 for digital docs)
   - `tier2_threshold`: Large shift SSIM threshold (0.92)
   - `tier1_search_range`: Local search radius (±20px)
   - `tier2_search_range`: Large shift search radius (±50px)

3. **Validation Config**:
   - `outlier_threshold`: MAD-based Z-score cutoff (1.0 = strict)

4. **Comparison Config**:
   - `pixel_diff_threshold`: Pixel intensity difference to consider a change (30/255)
   - `min_change_pixels`: Minimum pixels to report a change (128 pixels = 0.2% of 256×256 segment, catches 1 Japanese character)
   - `contour_min_area`: Minimum contour area for change regions (50 pixels)

---

## Phase 2: Main Entry Point - compare_documents()

### What to Build
A public method that:
1. Accepts file paths for baseline and candidate images
2. Orchestrates the full comparison pipeline
3. Returns a `DocumentComparisonResult` (Pydantic model)

### Method Signature Template
```python
def compare_documents(
    self,
    baseline_path: str | Path,
    candidate_path: str | Path
) -> DocumentComparisonResult:
    # Phase 1: Load and preprocess
    # Phase 2: Segment both images
    # Phase 3: Align all segments
    # Phase 4: Validate alignment consistency
    # Phase 5: Decide on comparison strategy
    # Phase 6: Compare segments
    # Phase 7: Aggregate results
    # Phase 8: Return DocumentComparisonResult
```

### Why This Matters
- **Single Responsibility**: This is the only public API - clean interface
- **Error Handling**: Centralized place to catch and report failures
- **Pipeline Orchestration**: Controls the flow through all components
- **Result Packaging**: Ensures all data is properly structured in Pydantic models

### Error Cases to Handle
1. **File I/O**: Missing files, unreadable images, permission errors
2. **Image Validation**: Different dimensions, corrupt images
3. **Component Failures**: Segmentation issues, alignment failures
4. **Memory Management**: Large images causing OOM

---

## Phase 3: Image Loading & Preprocessing

### What to Build
Integration with `ImagePreprocessor` to:
1. Load both images from disk
2. Validate compatibility (same dimensions)
3. Convert to normalized format (grayscale, float32, [0,1] range)

### Integration Pattern
```python
# Inside compare_documents()
preprocessor = ImagePreprocessor()
baseline_img, candidate_img = preprocessor.prepare_image_pair(
    baseline_path,
    candidate_path
)
```

### Why This Matters
- **Normalization**: Ensures consistent pixel value ranges across images
- **Format Standardization**: OpenCV operations expect specific data types
- **Early Validation**: Catches dimension mismatches before expensive operations
- **Color Consistency**: Grayscale conversion removes color variation noise

### What the Preprocessor Does
1. Loads images using OpenCV (cv2.imread)
2. Converts BGR → Grayscale (removes color channel variations)
3. Normalizes to float32 [0, 1] range (standardizes math operations)
4. Validates dimensions match (baseline.shape == candidate.shape)
5. Checks image quality (not all black, not all white, sufficient entropy)

---

## Phase 4: Segmentation

### What to Build
Grid-based segmentation with overlap:
1. Call `Segmentor.segment_image()` on both images
2. Receive list of `Segment` objects (Pydantic models)
3. Store segments for both baseline and candidate

### Integration Pattern
```python
segmentor = Segmentor(segment_size=256, overlap=10)
baseline_segments = segmentor.segment_image(baseline_img)
candidate_segments = segmentor.segment_image(candidate_img)
```

### Why This Matters
- **Localized Processing**: Compare small regions instead of entire image (reduces memory, enables parallel processing)
- **Overlap Strategy**: 10% overlap ensures changes near boundaries aren't missed
- **Information Filtering**: Segments calculate entropy and variance to identify informative regions
- **Spatial Organization**: Grid structure enables spatial analysis (adjacent segments, rows/columns)

### Segment Properties to Use
1. **bbox**: (x, y, width, height) - Location in image
2. **row, col**: Grid position - For spatial analysis
3. **entropy**: Information content - Skip near-empty segments
4. **variance**: Pixel intensity variation - Skip uniform regions
5. **image_data**: Actual pixel data - For comparison operations

### Key Decisions
- **Skip Low-Information Segments?**: Segments with entropy < 1.0 or variance < 25.0 are mostly blank - can skip comparison to save computation
- **Segment Count**: For 1920×1080 image with 256px segments and 10% overlap: ~63 segments (9 cols × 7 rows)

---

## Phase 5: Alignment for Each Segment

### What to Build
For each baseline segment, find its corresponding location in candidate:
1. Loop through all baseline segments
2. Call `Aligner.align_segment()` for each
3. Store `AlignmentResult` for each segment pair
4. Track alignment statistics (tier distribution, average shifts)

### Integration Pattern
```python
aligner = Aligner(
    tier1_threshold=0.98,
    tier2_threshold=0.92,
    tier1_search_range=20,
    tier2_search_range=50
)

alignment_results = []
for baseline_seg in baseline_segments:
    alignment = aligner.align_segment(
        baseline_seg,
        candidate_img,
        candidate_segments
    )
    alignment_results.append(alignment)
```

### Why This Matters
- **Handles Layout Shifts**: Content may move between versions (pagination, reflow, responsive layout changes)
- **Multi-Tier Search**: Local search first (fast), fallback to large search (thorough)
- **SSIM Matching**: Structural similarity handles minor rendering differences (anti-aliasing, font hinting)
- **Confidence Scoring**: tier1 (0.98) = high confidence, tier2 (0.92) = acceptable, no_match = investigate

### Alignment Types and Their Meaning
1. **TIER1**: Perfect match found within ±20px (local shift)
   - Common cause: Line wrapping, column reflow
   - Decision: Accept alignment, proceed with comparison
   
2. **TIER2**: Good match found within ±50px (larger shift)
   - Common cause: Paragraph reflow, image insertion above
   - Decision: Accept alignment, but flag for review
   
3. **NO_MATCH**: No good match found anywhere
   - Common cause: Content deleted, heavily modified, or layout drastically changed
   - Decision: Revert to zero-shift comparison (may produce many false positives)

### Alignment Result Properties to Store
1. **alignment_type**: TIER1, TIER2, or NO_MATCH
2. **shift_x, shift_y**: Pixel displacement (-50 to +50 range)
3. **confidence**: SSIM score (0.92 to 1.0)
4. **search_iterations**: How many positions tested (performance metric)

---

## Phase 6: Consistency Validation

### What to Build
Analyze all alignment results to detect outliers:
1. Extract shift vectors (shift_x, shift_y) from all alignments
2. Call `AlignmentValidator.validate_consistency()`
3. Receive `ValidationResult` with outlier information
4. Use validation to make decisions about which alignments to trust

### Integration Pattern
```python
validator = AlignmentValidator(outlier_threshold=1.0)
validation_result = validator.validate_consistency(alignment_results)

# Access validation metrics
outlier_count = len(validation_result.outlier_indices)
consistency_good = validation_result.is_highly_consistent()
```

### Why This Matters
- **Catches Alignment Errors**: If most segments shift by (+5, +10) but one shifts by (+45, -30), it's likely a bad match
- **Robust Statistics**: Uses MAD (Median Absolute Deviation) instead of standard deviation - resistant to outliers
- **Decision Support**: Helps decide whether to accept TIER2 alignments or revert to zero-shift
- **Quality Metrics**: Provides shift statistics for reporting (median, MAD, outlier percentage)

### Validation Algorithm (MAD-based)
1. Compute median shift: `median_x, median_y` from all shift vectors
2. Compute MAD: `median(|shift_x - median_x|)` and same for y
3. Compute modified Z-scores: `|shift_x - median_x| / (1.4826 * MAD_x)`
4. Flag outliers: Z-score > outlier_threshold (1.0 = strict for digital docs)

### Decision Logic Based on Validation
```python
if validation_result.is_highly_consistent():
    # Most segments agree on shift direction
    # → Accept all alignments (including TIER2)
    decision = "accept_all_alignments"
else:
    # Too many outliers or inconsistent shifts
    # → Reject outliers, use zero-shift for NO_MATCH segments
    decision = "reject_outliers"
    
    for idx in validation_result.outlier_indices:
        # Revert this segment's alignment to (0, 0)
        alignment_results[idx].shift_x = 0
        alignment_results[idx].shift_y = 0
        alignment_results[idx].alignment_type = AlignmentType.NO_MATCH
```

### Consistency Thresholds
- **outlier_threshold = 1.0**: Strict (for digital docs, expect consistent shifts)
- **is_highly_consistent()**: Returns True if outlier_percentage < 10%

---

## Phase 7: Per-Segment Comparison

### What to Build
For each aligned segment pair, perform pixel-level comparison:
1. Extract corresponding regions from baseline and candidate
2. Compute pixel difference map
3. Threshold differences to create binary change mask
4. Detect change regions (contours)
5. Classify change type (addition, deletion, modification)
6. Package into `SegmentComparisonResult`

### Method Structure
```python
def _compare_segments(
    self,
    baseline_segments: list[Segment],
    candidate_img: np.ndarray,
    alignment_results: list[AlignmentResult],
    validation_result: ValidationResult
) -> list[SegmentComparisonResult]:
    # Loop through segments
    # For each: extract regions, compare, detect changes
    # Return list of SegmentComparisonResult objects
```

### Why This Matters
- **Pixel-Level Precision**: Catches single character changes (0.2% threshold)
- **Spatial Localization**: Know exactly where changes occurred (bounding boxes)
- **Change Classification**: Distinguish additions from deletions (helps understand what changed)
- **Noise Filtering**: Minimum pixel count prevents false positives from anti-aliasing

---

## Phase 8: Pixel Difference Computation

### What to Build
Core mathematical operation to find changed pixels:
1. Extract aligned baseline region
2. Extract corresponding candidate region (using shift_x, shift_y)
3. Compute absolute difference
4. Apply threshold to create binary mask

### Algorithm Template
```python
def _compute_pixel_diff(
    self,
    baseline_region: np.ndarray,
    candidate_region: np.ndarray,
    threshold: float = 30/255
) -> tuple[np.ndarray, int]:
    # 1. Compute absolute difference: |baseline - candidate|
    # 2. Create binary mask: diff > threshold
    # 3. Count changed pixels: np.sum(mask)
    # 4. Return (mask, change_count)
```

### Why This Matters
- **Handles Grayscale**: Operates on normalized float32 [0, 1] images
- **Threshold Selection**: 30/255 ≈ 0.118 catches visible differences while ignoring anti-aliasing
- **Binary Output**: Clear changed/unchanged classification for each pixel
- **Change Quantification**: Pixel count determines if change is significant

### Threshold Tuning
- **30/255 (0.118)**: Recommended for digital documents
  - Catches: Text changes, color shifts, new graphics
  - Ignores: Font hinting variations, JPEG compression artifacts (if any)
- **Lower (e.g., 20/255)**: More sensitive, catches subtle anti-aliasing differences
- **Higher (e.g., 50/255)**: Less sensitive, only major changes

### Edge Cases
1. **Shift Results in Out-of-Bounds**: Clip candidate region to image boundaries
2. **Different Region Sizes**: Resize or pad to match (shouldn't happen with correct alignment)
3. **All Pixels Changed**: Likely alignment failure, flag for manual review

---

## Phase 9: Change Classification

### What to Build
Determine whether detected change is an addition, deletion, or modification:
1. Analyze pixel intensities in change regions
2. Compare baseline vs candidate brightness
3. Classify based on intensity patterns

### Classification Logic Template
```python
def _classify_change(
    self,
    baseline_region: np.ndarray,
    candidate_region: np.ndarray,
    change_mask: np.ndarray
) -> ChangeType:
    # Extract pixels where mask is True
    # Compute mean intensity for baseline and candidate
    # Compare:
    #   - candidate brighter → ADDITION
    #   - baseline brighter → DELETION
    #   - similar brightness → MODIFICATION
```

### Why This Matters
- **Semantic Understanding**: Tells users *what happened*, not just *something changed*
- **Review Prioritization**: Deletions are often regressions (missing content)
- **Visualization**: Different colors for additions (green) vs deletions (red) in heatmaps

### Classification Rules
1. **ADDITION** (ChangeType.ADDITION):
   - Candidate region significantly brighter than baseline
   - Example: New text, new image, filled background
   - Threshold: `mean(candidate[mask]) > mean(baseline[mask]) + delta`
   
2. **DELETION** (ChangeType.DELETION):
   - Baseline region significantly brighter than candidate
   - Example: Removed text, removed image, cleared background
   - Threshold: `mean(baseline[mask]) > mean(candidate[mask]) + delta`
   
3. **MODIFICATION** (ChangeType.MODIFICATION):
   - Similar intensity but different structure
   - Example: Text rewording, image replacement, color change
   - Threshold: `|mean(candidate[mask]) - mean(baseline[mask])| < delta`

### Delta Selection
- **delta = 0.1** (on [0, 1] scale): Reasonable for digital documents
- Adjust based on document type (darker backgrounds need smaller delta)

---

## Phase 10: Change Region Detection (Contours)

### What to Build
Convert binary change mask into discrete bounding boxes:
1. Apply morphological operations to connect nearby pixels
2. Find contours in the processed mask
3. Compute bounding boxes for each contour
4. Filter out small contours (noise)
5. Return list of `BoundingBox` objects

### Algorithm Template
```python
def _detect_change_regions(
    self,
    change_mask: np.ndarray,
    min_contour_area: int = 50
) -> list[BoundingBox]:
    # 1. Morphological closing: connect nearby pixels
    # 2. Find contours: cv2.findContours()
    # 3. For each contour:
    #      - Compute bounding box: cv2.boundingRect()
    #      - Filter by area: if area >= min_contour_area
    #      - Create BoundingBox object
    # 4. Return list of BoundingBox
```

### Why This Matters
- **Spatial Localization**: Instead of "segment has changes", know "change at (x, y, w, h)"
- **Multiple Changes Per Segment**: One segment can have several discrete change regions
- **Noise Filtering**: Morphological operations merge adjacent pixels, filter removes tiny artifacts
- **Visualization Ready**: Bounding boxes can be drawn directly on images

### Morphological Operations
1. **Closing (Dilation + Erosion)**:
   - Purpose: Connect nearby changed pixels into coherent regions
   - Kernel: 5×5 rectangular (adjustable)
   - Effect: Fills small gaps, merges adjacent changes
   
2. **Why Not Opening (Erosion + Dilation)?**:
   - Opening removes small features - we already have min_contour_area filter
   - Closing better preserves actual change boundaries

### Contour Detection
- **cv2.findContours()**: OpenCV function to detect connected components
- **Mode**: cv2.RETR_EXTERNAL (only outer contours, ignore holes)
- **Method**: cv2.CHAIN_APPROX_SIMPLE (compress contours to save memory)

### Minimum Contour Area
- **50 pixels**: Filters single-pixel noise while catching small text changes
- **Rationale**: Single character in Japanese is ~50-100 pixels at typical PDF resolutions

---

## Phase 11: Result Aggregation

### What to Build
Combine all segment comparisons into a document-level summary:
1. Collect all `SegmentComparisonResult` objects
2. Compute document-level statistics (total changes, change types, affected area)
3. Merge overlapping change regions from adjacent segments
4. Create `DocumentComparisonResult` (Pydantic model)

### Method Template
```python
def _aggregate_results(
    self,
    segment_results: list[SegmentComparisonResult],
    baseline_img_shape: tuple,
    candidate_img_shape: tuple
) -> DocumentComparisonResult:
    # 1. Count total changed segments
    # 2. Sum changed pixels across all segments
    # 3. Merge overlapping change regions
    # 4. Compute change density (% of image changed)
    # 5. Group by change type (additions, deletions, modifications)
    # 6. Build DocumentComparisonResult
```

### Why This Matters
- **Single Source of Truth**: One object contains all comparison data
- **High-Level Metrics**: Users want "5% of document changed" not raw pixel counts
- **Change Distribution**: Shows if changes are localized or scattered
- **Visualization Data**: Provides all data needed for heatmaps, overlays, reports

### Key Aggregation Metrics
1. **total_segments**: Total number of segments processed
2. **changed_segments**: Number of segments with detected changes
3. **total_pixels_changed**: Sum of changed pixels across all segments
4. **change_percentage**: (total_pixels_changed / total_image_pixels) × 100
5. **change_type_counts**: Dictionary {ADDITION: count, DELETION: count, MODIFICATION: count}
6. **merged_change_regions**: List of BoundingBox after merging overlaps
7. **segment_results**: Full list of SegmentComparisonResult (for drill-down)

### Overlap Handling in 10%-Overlap Grid
- **Problem**: Adjacent segments overlap by 10%, so change near boundary appears in 2 segments
- **Solution**: When aggregating, merge bounding boxes that overlap or are close (within 10px)
- **Algorithm**: See next phase

---

## Phase 12: Merging Overlapping Change Regions

### What to Build
Deduplicate change regions detected in overlapping segments:
1. Collect all bounding boxes from all segments
2. Convert to absolute image coordinates (segment offset + local coords)
3. Find overlapping or nearby boxes
4. Merge them into larger bounding boxes
5. Return deduplicated list

### Algorithm Template
```python
def _merge_change_regions(
    self,
    segment_results: list[SegmentComparisonResult],
    merge_distance: int = 10
) -> list[BoundingBox]:
    # 1. Convert all BoundingBox to absolute coordinates
    # 2. Sort by x-coordinate
    # 3. Iteratively merge:
    #      - If box1 overlaps box2 OR distance < merge_distance:
    #          - merged_box = bounding_box(box1 ∪ box2)
    #      - Else: keep separate
    # 4. Return merged list
```

### Why This Matters
- **Deduplication**: Prevents double-counting changes in overlapping regions
- **Cleaner Visualization**: One large box instead of multiple small overlapping boxes
- **Accurate Metrics**: Change percentage reflects true affected area
- **Grouping**: Multiple small changes become one logical "change cluster"

### Overlap Detection
1. **Bounding Box Intersection**:
   ```
   overlap = (x1 < x2 + w2) AND (x2 < x1 + w1) AND
             (y1 < y2 + h2) AND (y2 < y1 + h1)
   ```

2. **Distance-Based Merging**:
   ```
   distance = min_distance(box1, box2)
   if distance < merge_distance:
       merge boxes
   ```

### Merge Strategy
- **Simple Union**: `merged_x = min(x1, x2)`, `merged_y = min(y1, y2)`, 
                     `merged_w = max(x1+w1, x2+w2) - merged_x`, similar for height
- **Alternative**: Convex hull if change regions are non-rectangular

### Merge Distance Tuning
- **10px**: Merges changes within typical character spacing
- **Larger (20-30px)**: Merges changes in same line/paragraph
- **0px**: Only merge actual overlaps (strictest)

---

## Phase 13: Building SegmentComparisonResult

### What to Build
Package all per-segment comparison data into Pydantic model:
1. Reference to baseline segment
2. Alignment information (shift, confidence, type)
3. Change detection results (pixel count, regions, change type)
4. Metadata (processing time, flags)

### Data to Include
```python
SegmentComparisonResult(
    segment=baseline_segment,  # Pydantic Segment object
    alignment_result=alignment,  # Pydantic AlignmentResult object
    is_changed=(changed_pixels >= min_change_pixels),
    changed_pixels=changed_pixels,
    change_regions=[list of BoundingBox],
    dominant_change_type=ChangeType.ADDITION/DELETION/MODIFICATION,
    change_type_distribution={
        ChangeType.ADDITION: pixel_count,
        ChangeType.DELETION: pixel_count,
        ChangeType.MODIFICATION: pixel_count
    }
)
```

### Why This Matters
- **Pydantic Validation**: Automatic validation of all fields (non-negative pixel counts, valid enums)
- **JSON Serialization**: Can save results to disk for later analysis
- **Type Safety**: IDE autocomplete and type checking
- **Traceable**: Each result links back to original segment and alignment

### Helper Methods to Use
1. **segment.area()**: Get segment size for normalization
2. **alignment_result.is_good_match()**: Check if alignment is reliable
3. **get_change_density()**: Compute (changed_pixels / segment.area()) × 100

---

## Phase 14: Building DocumentComparisonResult

### What to Build
Package all document-level data into final Pydantic model:
1. Image metadata (dimensions, file paths)
2. Aggregated statistics (total changes, percentages)
3. List of all SegmentComparisonResult objects
4. Merged change regions
5. Alignment and validation summaries

### Data to Include
```python
DocumentComparisonResult(
    baseline_shape=baseline_img.shape,
    candidate_shape=candidate_img.shape,
    total_segments=len(segment_results),
    changed_segments=sum(1 for r in segment_results if r.is_changed),
    segment_results=segment_results,
    total_pixels_changed=sum(r.changed_pixels for r in segment_results),
    change_percentage=compute_percentage(),
    merged_change_regions=merged_regions,
    alignment_summary={
        "tier1_count": count_tier1,
        "tier2_count": count_tier2,
        "no_match_count": count_no_match,
        "median_shift": validation_result.median_shift,
        "outlier_count": len(validation_result.outlier_indices)
    },
    change_type_summary={
        ChangeType.ADDITION: total_addition_pixels,
        ChangeType.DELETION: total_deletion_pixels,
        ChangeType.MODIFICATION: total_modification_pixels
    }
)
```

### Why This Matters
- **Complete Audit Trail**: Everything needed to reproduce and verify results
- **Dashboard Ready**: All metrics available for reporting UI
- **JSON Export**: Can save to file and load later for analysis
- **Drill-Down Support**: High-level summary + detailed segment results

### Helper Methods to Use
1. **get_change_density()**: Returns change_percentage
2. **get_detailed_statistics()**: Returns dictionary with all stats
3. **get_change_heatmap_data()**: Returns (H, W) array of change intensities for visualization

---

## Phase 15: Error Handling & Edge Cases

### Critical Error Cases
1. **File Not Found**:
   - Action: Raise clear exception with file path
   - User needs: Check file existence before calling compare_documents()

2. **Dimension Mismatch**:
   - Action: Preprocessor catches this, raise ValueError
   - User needs: Ensure baseline and candidate are same resolution

3. **All Segments Have NO_MATCH**:
   - Action: Log warning, proceed with zero-shift comparison
   - Likely cause: Completely different documents (wrong pair)

4. **Memory Issues (Large Images)**:
   - Action: Consider tiling or downsampling
   - Threshold: Images > 10,000 × 10,000 may cause issues

5. **No Changes Detected**:
   - Action: Return DocumentComparisonResult with changed_segments=0
   - This is valid: documents may be identical

### Validation Checks
1. **Before Segmentation**: Images are not None, have valid shape
2. **After Segmentation**: At least 1 segment returned
3. **After Alignment**: All alignments have valid shift ranges (-50 to +50)
4. **After Comparison**: All SegmentComparisonResult objects are valid Pydantic models

---

## Phase 16: Optimization Strategies

### Performance Bottlenecks
1. **Alignment Search**: Most expensive operation
   - Optimization: Parallelize alignment for different segments (use multiprocessing)
   - Savings: Near-linear speedup with CPU cores

2. **Pixel Difference Computation**: Memory-intensive for large segments
   - Optimization: Use NumPy vectorized operations (already optimal)
   - Alternative: Downsample segments before comparison (trade accuracy for speed)

3. **Contour Detection**: Can be slow for complex change masks
   - Optimization: Increase min_contour_area to filter more aggressively
   - Alternative: Use connected components instead of contours

### Memory Optimizations
1. **Don't Store Full Images in Results**:
   - Store only metadata (shape, file path)
   - Reload images when needed for visualization

2. **Release Segment Image Data**:
   - After comparison, can discard segment.image_data
   - Keep only bbox, entropy, variance

3. **Lazy Loading**:
   - Don't compute merged_change_regions until requested
   - Don't generate heatmap data until visualization needed

### Parallelization Strategy
```python
from multiprocessing import Pool

def compare_documents_parallel(self, ...):
    # Phase 1-2: Load and segment (sequential)
    
    # Phase 3: Align segments (PARALLELIZE)
    with Pool() as pool:
        alignment_results = pool.map(
            aligner.align_segment,
            baseline_segments
        )
    
    # Phase 4: Validate (sequential)
    # Phase 5-6: Compare segments (PARALLELIZE)
    with Pool() as pool:
        segment_results = pool.starmap(
            self._compare_single_segment,
            [(seg, align) for seg, align in zip(...)]
        )
    
    # Phase 7: Aggregate (sequential)
```

---

## Phase 17: Testing Strategy

### Unit Tests (Per Method)
1. **_compute_pixel_diff**:
   - Test: Identical images → 0 changed pixels
   - Test: Single white pixel on black background → 1 changed pixel
   - Test: Threshold edge case (diff = 30/255 exactly)

2. **_classify_change**:
   - Test: Brighter candidate → ADDITION
   - Test: Brighter baseline → DELETION
   - Test: Equal brightness → MODIFICATION

3. **_detect_change_regions**:
   - Test: No changes → empty list
   - Test: Single large change → 1 bounding box
   - Test: Multiple small changes → multiple bounding boxes
   - Test: Noise filtering (contours < min_area filtered out)

4. **_merge_change_regions**:
   - Test: Non-overlapping boxes → no merge
   - Test: Overlapping boxes → merged box
   - Test: Adjacent boxes (within merge_distance) → merged box

### Integration Tests
1. **Full Pipeline**:
   - Input: Known baseline and candidate with controlled changes
   - Expected: Specific number of changed segments, known change locations
   - Verify: DocumentComparisonResult.change_percentage matches expected

2. **Edge Cases**:
   - Test: Identical images → 0% change
   - Test: Completely different images → high % change, many NO_MATCH alignments
   - Test: Layout shift (all content moved by constant offset) → 0% change (perfect alignment)

3. **Performance Tests**:
   - Test: Large image (4K resolution) completes within time budget
   - Test: Many segments (500+ segments) doesn't OOM

### Validation Tests
1. **Pydantic Model Validation**:
   - Test: Invalid SegmentComparisonResult (negative changed_pixels) → ValidationError
   - Test: Invalid BoundingBox (width=0) → ValidationError

2. **Alignment Consistency**:
   - Test: All segments shift by (+10, +5) → is_highly_consistent() = True
   - Test: One segment shifts by (+50, -40) → outlier detected

---

## Phase 18: Visualization Preparation

### What to Build
Methods to prepare data for visualization (not actual rendering):
1. **Change heatmap**: 2D array of change intensity per pixel
2. **Overlay data**: List of bounding boxes with colors (red=deletion, green=addition)
3. **Side-by-side data**: Coordinates for splitting/highlighting differences

### Heatmap Data Structure
```python
def get_change_heatmap_data(self) -> np.ndarray:
    # Create H×W array initialized to 0
    # For each segment_result:
    #   For each change_region:
    #     heatmap[y:y+h, x:x+w] = 1.0 (or change_density)
    # Return heatmap array
```

### Why This Matters
- **Decoupling**: Comparison logic separate from visualization code
- **Flexibility**: Same data can drive web UI, CLI report, or PDF export
- **Caching**: Compute heatmap once, render multiple times

### Visualization Consumers
1. **CLI Report**: Print text summary with change statistics
2. **HTML Report**: Render side-by-side images with overlays
3. **Dashboard**: Display heatmap, bounding boxes, statistics
4. **PDF Export**: Generate annotated comparison report

---

## Phase 19: Configuration Tuning Guide

### When to Adjust Thresholds

#### Segment Size (segment_size)
- **Smaller (128px)**: More granular, catches localized changes, slower
- **Larger (512px)**: Faster, but may miss small changes
- **Recommendation**: 256px is sweet spot for most documents

#### Tier1 Threshold (tier1_threshold)
- **Higher (0.99)**: Stricter matching, fewer false alignments
- **Lower (0.95)**: More lenient, may accept bad alignments
- **Digital docs**: 0.98 works well
- **Scanned docs**: Lower to 0.95 due to noise

#### Tier2 Threshold (tier2_threshold)
- **Higher (0.95)**: Require better match for large shifts
- **Lower (0.90)**: Accept more marginal alignments
- **Recommendation**: 0.92 for digital, 0.88 for scanned

#### Outlier Threshold (outlier_threshold)
- **Lower (0.5)**: More permissive, fewer outliers flagged
- **Higher (1.5)**: Stricter, flags minor inconsistencies
- **Digital docs**: 1.0 (expect consistency)
- **Layout chaos**: 2.0 (tolerate variation)

#### Min Change Pixels (min_change_pixels)
- **Calculate**: (target_feature_size / segment_area) × segment_pixels
- **Example**: Detect single Japanese character (100px) in 256×256 segment:
  - min_change_pixels = (100 / 65536) × 65536 = 100 pixels
- **Typical**: 128 pixels (0.2% of segment) catches single characters

#### Pixel Diff Threshold (pixel_diff_threshold)
- **Lower (20/255)**: Catches subtle changes, more false positives
- **Higher (40/255)**: Only major changes, may miss small text edits
- **Digital docs**: 30/255 (ignores anti-aliasing)

---

## Phase 20: Final Implementation Checklist

### Core Methods to Implement
- [ ] `ImageComparator.__init__()` - Initialize with config
- [ ] `compare_documents()` - Main entry point
- [ ] `_compare_segments()` - Loop and orchestrate per-segment comparison
- [ ] `_compare_single_segment()` - Core comparison logic for one segment
- [ ] `_compute_pixel_diff()` - Mathematical diff operation
- [ ] `_classify_change()` - Determine ADDITION/DELETION/MODIFICATION
- [ ] `_detect_change_regions()` - Contour detection and bounding boxes
- [ ] `_aggregate_results()` - Build DocumentComparisonResult
- [ ] `_merge_change_regions()` - Deduplicate overlapping boxes

### Integration Points
- [ ] Preprocessor integration (prepare_image_pair)
- [ ] Segmentor integration (segment_image)
- [ ] Aligner integration (align_segment)
- [ ] Validator integration (validate_consistency)
- [ ] Pydantic models (Segment, AlignmentResult, ValidationResult, SegmentComparisonResult, DocumentComparisonResult)

### Testing
- [ ] Unit tests for each _method
- [ ] Integration test: full pipeline with known test images
- [ ] Edge case tests (identical images, no matches, outliers)
- [ ] Performance test (large image, many segments)

### Documentation
- [ ] Docstrings for all public methods
- [ ] Type hints for all parameters and returns
- [ ] README with usage examples
- [ ] Configuration guide

---

## Implementation Order Recommendation

### Week 1: Foundation
1. **Day 1-2**: Implement `ImageComparator.__init__()` and `compare_documents()` skeleton
2. **Day 3-4**: Integrate Preprocessor, Segmentor, Aligner, Validator (Phase 1-6)
3. **Day 5**: Implement `_compute_pixel_diff()` and basic tests
4. **Day 6-7**: Implement `_compare_single_segment()` and `_compare_segments()`

### Week 2: Refinement
1. **Day 1-2**: Implement `_classify_change()` and tests
2. **Day 3-4**: Implement `_detect_change_regions()` with morphological operations
3. **Day 5**: Implement `_merge_change_regions()`
4. **Day 6-7**: Implement `_aggregate_results()` and DocumentComparisonResult building

### Week 3: Polish
1. **Day 1-2**: Integration testing with real documents
2. **Day 3-4**: Optimization (parallelization if needed)
3. **Day 5**: Visualization data preparation methods
4. **Day 6-7**: Documentation and configuration tuning guide

---

## Key Takeaways

### Design Principles
1. **Separation of Concerns**: Each component (Preprocessor, Segmentor, etc.) handles one responsibility
2. **Data Validation**: Pydantic models ensure correctness throughout pipeline
3. **Configurability**: All thresholds and parameters adjustable for different document types
4. **Traceability**: Every result links back to source segment and alignment

### Why This Architecture
1. **Handles Layout Shifts**: Aligner finds content even if moved
2. **Pixel-Level Precision**: Catches single character changes (0.2% threshold)
3. **Robust to Outliers**: MAD-based validation rejects bad alignments
4. **Optimized for Digital Docs**: Thresholds tuned for digitally-generated content (no scanning noise)

### Success Criteria
1. **Accuracy**: Detects 99%+ of single Japanese character changes
2. **False Positive Rate**: < 1% (mostly from anti-aliasing near threshold)
3. **Performance**: Processes 1920×1080 document in < 5 seconds (without parallelization)
4. **Robustness**: Handles layout shifts up to ±50px without manual intervention

---

## Next Steps After Implementation

### Phase 1 Extensions
1. **Confidence Scoring**: Add per-change confidence scores
2. **Change Clustering**: Group related changes into logical "change events"
3. **Temporal Analysis**: Compare across multiple document versions (A → B → C)

### Phase 2: ML Intelligence
1. **Semantic Segmentation**: Replace grid with content-aware segments (paragraphs, tables)
2. **Change Classification**: Train classifier to distinguish bug vs intentional change
3. **OCR Integration**: Extract and compare text content directly

### Phase 3: Web Interface
1. **Interactive Visualization**: Click on change → see before/after detail
2. **Review Workflow**: Approve/reject changes, export report
3. **Batch Processing**: Compare hundreds of document pairs

---

## FAQ

### Q: Why not just use simple pixel diff without alignment?
**A**: Layout shifts are common (pagination, reflow, responsive design). Without alignment, every shifted character looks like a deletion+addition, creating massive false positives.

### Q: Why grid-based segmentation instead of content-aware?
**A**: Simplicity and robustness. Grid is deterministic, parallelizable, and doesn't depend on content detection (which can fail on non-text content). Content-aware segmentation is planned for Phase 2.

### Q: Why SSIM instead of simple template matching?
**A**: SSIM handles minor rendering differences (anti-aliasing, font hinting, compression artifacts) that would cause template matching to fail. It's also more robust to brightness/contrast variations.

### Q: What if documents have different page counts?
**A**: Current design assumes same-page comparison. Multi-page comparison is out of scope for Phase 1. Future: detect page insertions/deletions.

### Q: Can this handle color images?
**A**: Preprocessor converts to grayscale. Color comparison is possible but adds complexity (3 channels). For document regression testing, grayscale is sufficient.

### Q: How does this compare to existing tools (Percy, Applitools)?
**A**: Those are SaaS services with advanced ML models. ShiftVis is open-source, self-hosted, optimized for Japanese documents, and provides full control over thresholds and algorithms.

---

## Conclusion

This implementation guide provides the complete blueprint for building `ImageComparator` - the orchestrator that ties together all ShiftVis components. Follow the phases sequentially, test each method independently, and refer to this document when making architectural decisions.

**Remember**: The goal is pixel-level precision with intelligent layout shift handling. Every threshold and algorithm choice is optimized for this balance.

Good luck with the implementation! 🚀
