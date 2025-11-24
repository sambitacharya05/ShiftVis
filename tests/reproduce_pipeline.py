import os
import sys
from pathlib import Path

import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

from src.core.comparator import ImageComparator
from src.core.config import settings

def test_pipeline():
    comparator = ImageComparator()
    project_root = Path.cwd()
    base_dir = project_root / "data" / "test-docs" / "integration test"
    baseline_path = base_dir / "bl_jp.png"
    test_path = base_dir / "test_jp.png"
    output_dir = base_dir / "output"

    # Ensure output dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Update settings for test
    settings.STORAGE_OUTPUT_DIR = str(output_dir)
    settings.DEBUG_MODE = True

    print(f"Testing pipeline with:")
    print(f"Baseline: {baseline_path}")
    print(f"Test: {test_path}")
    print(f"Output: {output_dir}")
    
    # Run comparison
    try:
        result = comparator.compare_documents(str(baseline_path), str(test_path))
        
        print("\nComparison successful!")
        print(f"Overall Similarity: {result.overall_similarity:.4f}")
        print(f"Change Percentage: {result.change_percentage:.2f}%")
        print(f"Segments with changes: {result.segments_with_changes}/{result.total_segments}")
        
        # Verify results
        assert result.total_segments > 0, "No segments processed"
        assert result.overall_similarity > 0, "Similarity score should be positive"
        
        # Save detailed statistics
        stats = result.get_detailed_statistics()
        output_file = output_dir / "integration_test_results.json"
        result.save_to_file(filepath=str(output_file))
        print(f"\nResults saved to: {output_file}")
            
    except Exception as e:
        print(f"\nERROR: Comparison failed with exception:")
        print(str(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_pipeline()
