from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Segment:
    """
    A data class representing a segmented region of an image.

    Attributes:
    segment_id (str): Unique identifier for the segment.
    x (int): X-coordinate of the top-left corner of the segment.
    y (int): Y-coordinate of the top-left corner of the segment.
    width (int): Width of the segment in pixels.
    height (int): Height of the segment in pixels.
    bbox (Tuple[int, int, int, int]): Bounding box of the segment as (x, y, width, height).
    data (np.ndarray, optional): Numpy array containing the segment's pixel data. Defaults to None.
    """
    segment_id: str
    x: int
    y: int
    width: int
    height: int
    bbox: Tuple[int, int, int, int]
    data: np.ndarray = None


class ImageSegmentor:
    """A class to handle segmentation of images into overlapping tiles."""
    def __init__(self, segment_size: int = 256, overlap_percentage: float = 0.10):
        """
        Initialize the Segmentor with specified segment size and overlap settings.

        Args:
            segment_size (int, optional): The size of each segment in pixels. Defaults to 256.
            overlap_percentage (float, optional): The percentage of overlap between adjacent segments,
                expressed as a decimal (e.g., 0.10 for 10%). Defaults to 0.10.

        Attributes:
            segment_size (int): The size of each segment in pixels.
            overlap_percentage (float): The percentage of overlap between segments.
            overlap_pixels (int): The number of overlapping pixels calculated from segment_size and overlap_percentage.
            stride (int): The step size between consecutive segments, calculated as segment_size minus overlap_pixels.
        """
        self.segment_size = segment_size
        self.overlap_percentage = overlap_percentage
        self.overlap_pixels = int(segment_size * overlap_percentage)
        self.stride = segment_size - self.overlap_pixels

    def _calculate_segments_for_dimension(self, dimension: int) -> int:
        """Helper to calculate segments for a single dimension."""
        remaining = dimension - self.segment_size
        return 1 + int(np.ceil(remaining / self.stride)) if remaining > 0 else 1

    def calculate_grid(self, image_height: int, image_width: int) -> Tuple[int, int]:
        """
        Calculate number of segments needed for given image dimensions.
        
        Args:
            image_height: Height of image in pixels
            image_width: Width of image in pixels
            
        Returns:
            Tuple of (num_rows, num_cols)

        Examples:
            >>> segmenter = ImageSegmenter(segment_size=256, overlap_percent=0.10)
            >>> segmenter.calculate_grid(512, 512)
            (3, 3)
            >>> segmenter.calculate_grid(1000, 800)
            (4, 5)
        """
        if image_height <= 0 or image_width <= 0:
            raise ValueError(f"Image dimensions must be positive: {image_height}x{image_width}")
        
        num_cols = self._calculate_segments_for_dimension(image_width)
        num_rows = self._calculate_segments_for_dimension(image_height)
        return num_rows, num_cols
        
    def segment_image(self, image: np.ndarray) -> list[Segment]:
        """
        Segment an image into overlapping tiles.
        
        Args:
            image: Input image as numpy array (H, W, C) or (H, W)
            
        Returns:
            List of Segment objects with metadata (data will be extracted separately)
        """
        # Determine image dimensions and extract height and width
        if image.ndim == 3:
            im_height, im_width, _ = image.shape[:2]
        else:
            im_height, im_width = image.shape

        # Calculate grid of segments
        n_rows, n_cols = self.calculate_grid(im_height, im_width)
        
        segments = [] # List to hold segment metadata

        for row in range(n_rows): # Iterate over rows: vertical position
            for col in range(n_cols): # Iterate over columns: horizontal position
                # Calculate top-left corner of segment
                x = col * self.stride
                y = row * self.stride

                # Calculate actual width and height to avoid exceeding image boundaries
                seg_width = min(self.segment_size, im_width - x)
                seg_height = min(self.segment_size, im_height - y)

                # Define bounding box as (x1, y1, x2, y2)
                bbox = (x, y, x + seg_width, y + seg_height)

                # Create Segment object with metadata
                each_segment = Segment(
                    segment_id = f"seg_r{row}_c{col}",
                    x = x,
                    y = y,
                    width = seg_width,
                    height = seg_height,
                    bbox = bbox,
                    data = None
                )

                # Append segment metadata to list
                segments.append(each_segment)
        
        # Return list of segment metadata
        return segments
    
    def extract_segment_data(self, image: np.ndarray, segment: Segment) -> np.ndarray:
        """
        Extract the actual pixel data for a segment.
        
        Args:
            image: Source image
            segment: Segment metadata
            
        Returns:
            Segment image data as numpy array
        """
        y_start = segment.y
        y_end = segment.y + segment.height
        x_start = segment.x
        x_end = segment.x + segment.width
        if image.ndim == 3:
            segment_data = image[y_start:y_end, x_start:x_end, :]
        else:
            segment_data = image[y_start:y_end, x_start:x_end]
        return segment_data

# TESTING CODE
if __name__ == "__main__":
    import cv2
    import os

    print("=== Image Segmentation Visualization ===\n")

    # Load test image
    image_path = "data/test-docs/input.png"

    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        print("Please ensure the file exists or update the path.")
        exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Failed to load image from {image_path}")
        exit(1)

    print(f"✓ Loaded image: {image.shape} (H, W, C)")

    # Create segmentor
    segmentor = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
    print(f"✓ Created segmentor: segment_size={segmentor.segment_size}, overlap={segmentor.overlap_percentage}")
    print(f"  - Overlap pixels: {segmentor.overlap_pixels}")
    print(f"  - Stride: {segmentor.stride}")

    # Calculate grid
    num_rows, num_cols = segmentor.calculate_grid(image.shape[0], image.shape[1])
    print(f"\n✓ Grid calculation: {num_rows} rows × {num_cols} cols = {num_rows * num_cols} segments")

    # Segment the image
    segments = segmentor.segment_image(image)
    print(f"✓ Created {len(segments)} segments\n")

    # Create visualization
    vis_image = image.copy()

    # Draw all segment bounding boxes
    for i, segment in enumerate(segments):
        x1, y1, x2, y2 = segment.bbox

        # Alternate colors for better visibility
        if i % 2 == 0:
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue

        # Draw rectangle
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Add segment ID label
        cv2.putText(
            vis_image,
            segment.segment_id,
            (x1 + 5, y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    # Save visualization
    output_path = "data/test-docs/segmentation_output.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_image)
    print(f"✓ Saved visualization to: {output_path}")

    # Display segment details (first 5)
    print(f"\n=== First 5 Segments ===")
    for segment in segments[:5]:
        print(f"{segment.segment_id}:")
        print(f"  Position: ({segment.x}, {segment.y})")
        print(f"  Size: {segment.width}×{segment.height}")
        print(f"  Bbox: {segment.bbox}")

    # Test extract_segment_data on first segment
    print(f"\n=== Testing extract_segment_data ===")
    first_seg = segments[0]
    seg_data = segmentor.extract_segment_data(image, first_seg)
    print(f"✓ Extracted {first_seg.segment_id}: shape={seg_data.shape}")

    # Save first segment as separate image
    seg_output_path = "data/test-docs/first_segment.png"
    cv2.imwrite(seg_output_path, seg_data)
    print(f"✓ Saved first segment to: {seg_output_path}")

    print("\n✅ Segmentation test complete!")