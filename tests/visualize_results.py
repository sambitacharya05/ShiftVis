"""
Enhanced visualization script for frontend-ready output with dual views.

Creates two comparison modes:
1. "with_shift" - Shows original comparison highlighting global shift
2. "aligned" - Shows comparison after applying shift correction to test image
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys
from typing import Dict, Any, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from src.core.output_manager import OutputManager


def load_diff_map(filepath: Path) -> np.ndarray:
    """Load a compressed diff map from storage."""
    data = np.load(filepath, allow_pickle=True)
    
    if data['storage_type'] == b'sparse' or (isinstance(data['storage_type'], np.ndarray) and data['storage_type'].item() == 'sparse'):
        # Reconstruct from sparse coordinates
        shape = tuple(data['shape'])
        diff_map = np.zeros(shape, dtype=np.uint8)
        coords = data['coords']
        if len(coords) > 0:
            diff_map[coords[:, 0], coords[:, 1]] = 255
        return diff_map
    else:
        # Dense storage
        return data['mask']


def get_change_color(change_type: str) -> Tuple[np.ndarray, str]:
    """
    Get color and name for change type.
    
    Returns:
        Tuple of (BGR color array, color name string)
    """
    change_type_lower = change_type.lower()
    
    if change_type_lower == 'content_change':
        return np.array([0, 0, 255], dtype=np.uint8), "RED (content change)"
    elif change_type_lower == 'position_shift':
        return np.array([0, 165, 255], dtype=np.uint8), "ORANGE (position shift)"
    elif change_type_lower == 'visual_change':
        return np.array([0, 255, 255], dtype=np.uint8), "YELLOW (visual change)"
    elif change_type_lower == 'none':
        return np.array([200, 200, 200], dtype=np.uint8), "GRAY (no change)"
    else:
        return np.array([0, 255, 255], dtype=np.uint8), f"YELLOW (unknown: {change_type})"


def apply_diff_overlay(
    image: np.ndarray,
    diff_map: np.ndarray,
    x: int,
    y: int,
    color: np.ndarray,
    alpha: float = 0.6
) -> None:
    """
    Apply colored diff overlay to image in-place.
    
    Args:
        image: Image to modify
        diff_map: Binary diff map
        x, y: Top-left position to place diff map
        color: BGR color to apply
        alpha: Opacity (0-1)
    """
    h, w = diff_map.shape[:2]
    img_h, img_w = image.shape[:2]
    
    # Ensure we don't go out of bounds
    y_end = min(y + h, img_h)
    x_end = min(x + w, img_w)
    actual_h = y_end - y
    actual_w = x_end - x
    
    # Resize diff map if needed
    if diff_map.shape[:2] != (actual_h, actual_w):
        diff_map = cv2.resize(diff_map, (actual_w, actual_h))
    
    # Create mask for changed pixels
    changed_mask = diff_map > 0
    
    # Apply color with blending
    for c in range(3):
        image[y:y_end, x:x_end][changed_mask, c] = (
            image[y:y_end, x:x_end][changed_mask, c] * (1 - alpha) +
            color[c] * alpha
        ).astype(np.uint8)


def create_overlay_image(
    base_image: np.ndarray,
    segments: Dict[str, Any],
    diffmaps_dir: Path,
    label: str
) -> Tuple[np.ndarray, int, int]:
    """
    Create overlay image with colored diff highlights.
    
    Returns:
        Tuple of (overlay_image, changes_found, segments_with_changes)
    """
    overlay = base_image.copy()
    h, w = overlay.shape[:2]
    
    seg_size = 256
    overlap = int(seg_size * 0.1)
    stride = seg_size - overlap
    
    changes_found = 0
    segments_with_changes = 0
    
    for seg_id, seg_info in segments.items():
        # Load diff map
        diff_file = diffmaps_dir / seg_info['filename']
        diff_map = load_diff_map(diff_file)
        
        # Skip if empty
        if not np.any(diff_map > 0):
            continue
        
        segments_with_changes += 1
        
        # Parse segment position
        parts = seg_id.split('_')
        row = int(parts[1][1:])
        col = int(parts[2][1:])
        
        x = col * stride
        y = row * stride
        
        # Get change type and color
        metadata = seg_info.get('metadata', {})
        change_type = metadata.get('change_type', 'unknown')
        color, _ = get_change_color(change_type)
        
        # Apply overlay
        apply_diff_overlay(overlay, diff_map, x, y, color)
        changes_found += 1
    
    # Add label
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, label, (10, 30), font, 1, (0, 255, 0), 2)
    
    return overlay, changes_found, segments_with_changes


def create_side_by_side(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Create side-by-side comparison with white gap."""
    h = max(img1.shape[0], img2.shape[0])
    gap = np.ones((h, 20, 3), dtype=np.uint8) * 255
    return np.hstack([img1, gap, img2])


def extract_segment_data(
    segments: Dict[str, Any],
    diffmaps_dir: Path
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Extract structured segment data for JSON files.
    
    Returns:
        Tuple of (all_segments, content_changes, statistics)
    """
    all_segments = []
    content_changes = []
    
    change_counts = {
        'none': 0,
        'content_change': 0,
        'visual_change': 0,
        'position_shift': 0,
        'no_match': 0,
        'skipped': 0
    }
    
    total_diff_pixels = 0
    total_pixels = 0
    
    for seg_id, seg_info in segments.items():
        metadata = seg_info.get('metadata', {})
        
        # Parse segment position
        parts = seg_id.split('_')
        row = int(parts[1][1:])
        col = int(parts[2][1:])
        
        # Load diff map to count pixels
        diff_file = diffmaps_dir / seg_info['filename']
        diff_map = load_diff_map(diff_file)
        diff_pixels = int(np.sum(diff_map > 0))
        segment_pixels = diff_map.shape[0] * diff_map.shape[1]
        
        total_diff_pixels += diff_pixels
        total_pixels += segment_pixels
        
        # Build segment data
        segment_data = {
            'id': seg_id,
            'row': row,
            'col': col,
            'x': metadata.get('baseline_x', 0),
            'y': metadata.get('baseline_y', 0),
            'width': metadata.get('baseline_width', 256),
            'height': metadata.get('baseline_height', 256),
            'change_type': metadata.get('change_type', 'unknown'),
            'alignment_type': metadata.get('alignment_type', 'unknown'),
            'shift': metadata.get('shift', [0, 0]),
            'diff_percentage': metadata.get('diff_percentage', 0.0),
            'diff_pixel_count': diff_pixels,
            'similarity_score': metadata.get('similarity_score', 1.0)
        }
        
        all_segments.append(segment_data)
        
        # Track change type
        change_type = metadata.get('change_type', 'unknown').lower()
        if change_type in change_counts:
            change_counts[change_type] += 1
        
        # Add to content changes if applicable
        if change_type == 'content_change' and diff_pixels > 0:
            content_changes.append(segment_data)
    
    # Calculate statistics
    statistics = {
        'total_segments': len(segments),
        'segments_with_changes': sum(1 for s in all_segments if s['diff_pixel_count'] > 0),
        'change_type_counts': change_counts,
        'overall_diff_percentage': (total_diff_pixels / total_pixels * 100) if total_pixels > 0 else 0,
        'total_diff_pixels': total_diff_pixels,
        'total_pixels': total_pixels
    }
    
    return all_segments, content_changes, statistics


def create_dual_view_visualization(base_dir: Path):
    """Create frontend-ready dual-view visualization."""
    
    # Paths
    baseline_path = base_dir / "baseline_1.png"
    test_path = base_dir / "test_3.png"
    
    # Find latest comparison results
    latest_result = sorted(Path("data/results").glob("*/comparison_*/page_001"))[-1]
    diffmaps_dir = latest_result / "diffMaps"
    index_file = diffmaps_dir / "index.json"
    
    # Output directory - use organized structure
    timestamp_dir = latest_result.parent.parent
    output_root = timestamp_dir / "frontend_output"
    
    print("=" * 80)
    print("CREATING FRONTEND-READY OUTPUT")
    print("=" * 80)
    print(f"  Baseline: {baseline_path}")
    print(f"  Test: {test_path}")
    print(f"  Results: {latest_result}")
    print(f"  Output: {output_root}")
    print("=" * 80)
    
    # Initialize OutputManager
    output_manager = OutputManager(output_root)
    
    # Load images
    baseline = cv2.imread(str(baseline_path))
    test = cv2.imread(str(test_path))
    
    if baseline is None or test is None:
        print("ERROR: Could not load images!")
        return
    
    h, w = baseline.shape[:2]
    print(f"\nImage size: {w}x{h}")
    
    # Save original images
    print("\n[1/6] Saving original images and thumbnails...")
    output_manager.save_original_images(baseline, test)
    
    # Load segment index
    with open(index_file) as f:
        index = json.load(f)
    segments = index['segments']
    
    # Check for global shift data
    comparison_dir = latest_result.parent
    global_shift_file = comparison_dir / "global_shift.json"
    global_shift_data = None
    
    if global_shift_file.exists():
        with open(global_shift_file) as f:
            global_shift_data = json.load(f)
    
    # If no global shift data found, create default
    if global_shift_data is None:
        print("  Warning: No global shift data found")
        global_shift_data = {
            'detected': False,
            'shift_vector': [0, 0],
            'confidence': 0.0,
            'direction': 'none',
            'magnitude': 0.0,
            'inlier_count': 0,
            'total_segments': len(segments)
        }
    
    print(f"\n[2/6] Creating 'with_shift' view (original comparison)...")
    baseline_overlay, changes1, segs1 = create_overlay_image(
        baseline, segments, diffmaps_dir, "BASELINE"
    )
    test_overlay, changes2, segs2 = create_overlay_image(
        test, segments, diffmaps_dir, "TEST (with global shift)"
    )
    
    side_by_side_shift = create_side_by_side(baseline_overlay, test_overlay)
    output_manager.save_shift_view(test_overlay, side_by_side_shift)
    print(f"  ✓ Found changes in {changes1} regions across {segs1} segments")
    
    # Create aligned view
    print(f"\n[3/6] Creating 'aligned' view (shift-corrected)...")
    if global_shift_data['detected']:
        shift_vec = tuple(global_shift_data['shift_vector'])
        print(f"  Applying shift correction: {shift_vec}")
        test_aligned = output_manager.apply_shift_correction(test, shift_vec)
    else:
        print("  No global shift detected, using original test image")
        test_aligned = test.copy()
    
    baseline_overlay_aligned, _, _ = create_overlay_image(
        baseline, segments, diffmaps_dir, "BASELINE"
    )
    test_overlay_aligned, _, _ = create_overlay_image(
        test_aligned, segments, diffmaps_dir, "TEST (aligned)"
    )
    
    side_by_side_aligned = create_side_by_side(baseline_overlay_aligned, test_overlay_aligned)
    output_manager.save_aligned_view(test_aligned, test_overlay_aligned, side_by_side_aligned)
    print(f"  ✓ Aligned view created")
    
    # Extract structured data
    print(f"\n[4/6] Extracting structured data...")
    all_segments, content_changes, statistics = extract_segment_data(segments, diffmaps_dir)
    print(f"  ✓ Processed {len(all_segments)} segments")
    print(f"  ✓ Found {len(content_changes)} content changes")
    
    # Save data files
    print(f"\n[5/6] Saving JSON data files...")
    output_manager.save_shift_analysis_data(global_shift_data)
    output_manager.save_segments_data(all_segments)
    output_manager.save_content_changes_data(content_changes)
    output_manager.save_statistics_data(statistics)
    print(f"  ✓ All data files saved")
    
    # Save metadata
    print(f"\n[6/6] Creating master metadata file...")
    output_manager.save_metadata(
        baseline_path=str(baseline_path),
        test_path=str(test_path),
        comparison_id=latest_result.parent.name,
        timestamp=timestamp_dir.name,
        has_global_shift=global_shift_data['detected'],
        shift_vector=tuple(global_shift_data['shift_vector']) if global_shift_data['detected'] else None
    )
    print(f"  ✓ Metadata saved")
    
    # Summary
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\n📁 Output structure:")
    print(f"  {output_root}/")
    print(f"    metadata.json (master control file)")
    print(f"    assets/")
    print(f"      original/   (baseline.png, test.png)")
    print(f"      with_shift/ (overlay.png, side_by_side.png)")
    print(f"      aligned/    (test_aligned.png, overlay.png, side_by_side.png)")
    print(f"      thumbnails/ (baseline_thumb.png, test_thumb.png)")
    print(f"    data/")
    print(f"      shift_analysis.json   (global shift detection)")
    print(f"      segments.json         (all segment details)")
    print(f"      content_changes.json  (filtered content changes)")
    print(f"      statistics.json       (summary statistics)")
    
    print(f"\n📊 Statistics:")
    print(f"  Total segments: {statistics['total_segments']}")
    print(f"  Segments with changes: {statistics['segments_with_changes']}")
    print(f"  Overall diff: {statistics['overall_diff_percentage']:.2f}%")
    
    if global_shift_data['detected']:
        shift_vec = global_shift_data['shift_vector']
        print(f"\n🔄 Global Shift Detected:")
        print(f"  Vector: ({shift_vec[0]}, {shift_vec[1]}) pixels")
        print(f"  Direction: {global_shift_data['direction']}")
        print(f"  Magnitude: {global_shift_data['magnitude']:.1f} pixels")
        print(f"  Confidence: {global_shift_data['confidence']:.1%}")
    else:
        print(f"\n✓ No global shift detected")
    
    print(f"\n✨ Color coding:")
    print(f"   🔴 RED = Content changes")
    print(f"   🟠 ORANGE = Position shifts")
    print(f"   🟡 YELLOW = Visual changes")
    print(f"   ⚪ GRAY = No change")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    base_dir = Path("data/test-docs/integration test")
    create_dual_view_visualization(base_dir)

