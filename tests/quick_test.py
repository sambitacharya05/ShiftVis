"""Quick test to generate comparison with global shift detection."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.comparator import ImageComparator

# Initialize comparator
comparator = ImageComparator()

# Disable debug mode for speed
comparator.debug_mode = False

# Paths
baseline_path = "data/test-docs/integration test/baseline_shift.png"
test_path = "data/test-docs/integration test/test_shift.png"

print(f"Running comparison...")
print(f"  Baseline: {baseline_path}")
print(f"  Test: {test_path}")

# Run comparison
result = comparator.compare_documents(baseline_path, test_path, doc_id="quick_test", page_num=1)

print(f"\n✓ Comparison complete")
print(f"  Overall similarity: {result.overall_similarity:.2%}")
print(f"  Change percentage: {result.change_percentage:.2f}%")
print(f"  Segments: {result.total_segments}")
print(f"  Changed segments: {result.changed_segments}")

# Check if global shift data was saved
from pathlib import Path
latest_result = sorted(Path("data/results").glob("*/comparison_*/"))[-1]
global_shift_file = latest_result / "global_shift.json"
if global_shift_file.exists():
    import json
    with open(global_shift_file) as f:
        shift_data = json.load(f)
    print(f"\n🔄 Global shift data saved:")
    print(f"  Detected: {shift_data['detected']}")
    if shift_data['detected']:
        print(f"  Vector: {shift_data['shift_vector']}")
        print(f"  Confidence: {shift_data['confidence']:.1%}")
else:
    print(f"\n⚠️  Global shift file not found at {global_shift_file}")
