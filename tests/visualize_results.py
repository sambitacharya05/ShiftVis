"""
Visualization script to show detected differences overlayed on images.
"""
import cv2
import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

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

def create_visualization(base_dir: Path):
    """Create visualization of detected differences."""
    
    # Paths
    baseline_path = base_dir / "bl.png"
    test_path = base_dir / "test.png"
    
    # Find latest comparison results
    latest_result = sorted(Path("data/results").glob("*/comparison_*/page_001"))[-1]
    diffmaps_dir = latest_result / "diffMaps"
    index_file = diffmaps_dir / "index.json"
    
    # Output directory - create in the timestamp folder
    timestamp_dir = latest_result.parent.parent  # Go up to timestamp folder
    output_dir = timestamp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading images...")
    print(f"  Baseline: {baseline_path}")
    print(f"  Test: {test_path}")
    print(f"  Results: {latest_result}")
    print(f"  Output: {output_dir}")
    
    # Load images
    baseline = cv2.imread(str(baseline_path))
    test = cv2.imread(str(test_path))
    
    if baseline is None or test is None:
        print("ERROR: Could not load images!")
        return
    
    h, w = baseline.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Save original images to output folder
    cv2.imwrite(str(output_dir / "baseline.png"), baseline)
    cv2.imwrite(str(output_dir / "test.png"), test)
    
    # Create overlay images (copy originals)
    baseline_overlay = baseline.copy()
    test_overlay = test.copy()
    
    # Load segment index
    with open(index_file) as f:
        index = json.load(f)
    
    segments = index['segments']
    print(f"\nProcessing {len(segments)} diff maps...")
    
    changes_found = 0
    
    # Process each diff map
    for seg_id, seg_info in segments.items():
        # Parse segment ID to get position (e.g., "seg_r0_c0" -> row 0, col 0)
        parts = seg_id.split('_')
        row = int(parts[1][1:])  # Remove 'r' prefix
        col = int(parts[2][1:])  # Remove 'c' prefix
        
        # Load diff map
        diff_file = diffmaps_dir / seg_info['filename']
        diff_map = load_diff_map(diff_file)
        
        # Calculate segment position (256x256 with 10% overlap)
        seg_size = 256
        overlap = int(seg_size * 0.1)
        stride = seg_size - overlap
        
        x = col * stride
        y = row * stride
        
        # Ensure we don't go out of bounds
        x_end = min(x + seg_size, w)
        y_end = min(y + seg_size, h)
        actual_w = x_end - x
        actual_h = y_end - y
        
        # Resize diff map if needed
        if diff_map.shape != (actual_h, actual_w):
            diff_map = cv2.resize(diff_map, (actual_w, actual_h))
        
        # Check if there are any changes
        if np.any(diff_map > 0):
            changes_found += 1
            
            # Highlight only the exact changed pixels in neon pink
            # Neon pink: (B=255, G=20, R=255) - bright magenta/pink
            neon_pink = np.array([255, 20, 255], dtype=np.uint8)
            
            # Create mask for changed pixels
            changed_mask = diff_map > 0
            
            # Apply neon pink to changed pixels with blending
            alpha = 0.6  # 60% opacity for the highlight
            
            # Blend neon pink with baseline image at changed pixel locations
            for c in range(3):  # For each color channel
                baseline_overlay[y:y_end, x:x_end][changed_mask, c] = (
                    baseline_overlay[y:y_end, x:x_end][changed_mask, c] * (1 - alpha) +
                    neon_pink[c] * alpha
                ).astype(np.uint8)
                
                test_overlay[y:y_end, x:x_end][changed_mask, c] = (
                    test_overlay[y:y_end, x:x_end][changed_mask, c] * (1 - alpha) +
                    neon_pink[c] * alpha
                ).astype(np.uint8)
    
    print(f"\n✅ Found changes in {changes_found} segments")
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(baseline_overlay, "BASELINE", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(test_overlay, "TEST (with changes)", (10, 30), font, 1, (0, 255, 0), 2)
    
    # Save visualizations
    # The output_dir is already defined and created above
    
    # No longer saving individual overlay images, only side-by-side
    # baseline_viz_path = output_dir / "baseline_with_diff_overlay.png"
    # test_viz_path = output_dir / "test_with_diff_overlay.png"
    
    # cv2.imwrite(str(baseline_viz_path), baseline_overlay)
    # cv2.imwrite(str(test_viz_path), test_overlay)
    
    # Create side-by-side comparison
    gap = np.ones((h, 20, 3), dtype=np.uint8) * 255  # White gap
    side_by_side = np.hstack([baseline_overlay, gap, test_overlay])
    
    comparison_path = output_dir / "side_by_side_comparison.png"
    cv2.imwrite(str(comparison_path), side_by_side)
    
    print(f"\n🎨 Visualization saved:")
    print(f"  Original baseline: {output_dir / 'baseline.png'}")
    print(f"  Original test: {output_dir / 'test.png'}")
    print(f"  Side-by-side comparison: {comparison_path}")
    print(f"\n✨ Done! Neon pink highlights show exact changed pixels.")
    print(f"   Yellow boxes show segments with changes.")

if __name__ == "__main__":
    base_dir = Path("data/test-docs/integration test")
    create_visualization(base_dir)
