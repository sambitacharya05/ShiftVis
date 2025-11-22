from typing import Tuple, Optional
import numpy as np
import cv2
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class Segment(BaseModel):
    """
    Represents a segmented region of an image with its properties and metrics.
    
    A Segment encapsulates a rectangular region of an image along with its 
    spatial coordinates, dimensions, and computed statistical properties.
    
    Attributes:
        segment_id: Unique identifier for the segment (format: 'seg_r{row}_c{col}')
        x: X-coordinate of the top-left corner of the segment (non-negative)
        y: Y-coordinate of the top-left corner of the segment (non-negative)
        width: Width of the segment in pixels (positive)
        height: Height of the segment in pixels (positive)
        bbox: Bounding box coordinates as (x1, y1, x2, y2)
        data: Pixel data of the segment as a numpy array (optional, for memory efficiency)
        entropy: Shannon entropy measure of the segment (0-8 for 8-bit images)
        variance: Statistical variance of pixel values (0-65025 for 8-bit images)
    
    Examples:
        >>> segment = Segment(
        ...     segment_id="seg_r0_c0",
        ...     x=0, y=0,
        ...     width=256, height=256,
        ...     bbox=(0, 0, 256, 256),
        ...     entropy=6.5,
        ...     variance=1200.5
        ... )
        >>> segment.area()
        65536
        >>> segment.center()
        (128.0, 128.0)
    """
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow numpy arrays
        validate_assignment=True,      # Validate on attribute changes
        frozen=False,                   # Allow modification after creation
        str_strip_whitespace=True      # Strip whitespace from strings
    )
    
    segment_id: str = Field(
        ...,
        min_length=1,
        pattern=r"^seg_r\d+_c\d+$",
        description="Unique segment identifier (format: 'seg_r{row}_c{col}')",
        examples=["seg_r0_c0", "seg_r2_c3", "seg_r10_c5"]
    )
    
    x: int = Field(
        ...,
        ge=0,
        description="X-coordinate of top-left corner (non-negative)"
    )
    
    y: int = Field(
        ...,
        ge=0,
        description="Y-coordinate of top-left corner (non-negative)"
    )
    
    width: int = Field(
        ...,
        gt=0,
        le=10000,
        description="Width of segment in pixels (positive, max 10000)"
    )
    
    height: int = Field(
        ...,
        gt=0,
        le=10000,
        description="Height of segment in pixels (positive, max 10000)"
    )
    
    bbox: Tuple[int, int, int, int] = Field(
        ...,
        description="Bounding box as (x1, y1, x2, y2)"
    )
    
    data: Optional[np.ndarray] = Field(
        None,
        description="Pixel data of the segment (optional for memory efficiency)",
        exclude=True  # Exclude from JSON serialization by default
    )
    
    entropy: float = Field(
        default=0.0,
        ge=0.0,
        le=8.0,
        description="Shannon entropy (0-8 for 8-bit images, higher = more information)"
    )
    
    variance: float = Field(
        default=0.0,
        ge=0.0,
        le=65025.0,
        description="Pixel variance (0-65025 for 8-bit images, max = 255²)"
    )
    
    @field_validator('segment_id')
    @classmethod
    def validate_segment_id_format(cls, v: str) -> str:
        """
        Validate that segment_id follows the correct format.
        
        Format: seg_r{row}_c{col}
        Examples: seg_r0_c0, seg_r2_c3, seg_r10_c5
        """
        if not v.startswith('seg_r'):
            raise ValueError(
                f"segment_id must start with 'seg_r', got: {v}"
            )
        
        # Extract row and column numbers
        parts = v.split('_')
        if len(parts) != 3:
            raise ValueError(
                f"segment_id format must be 'seg_r{{row}}_c{{col}}', got: {v}"
            )
        
        try:
            row = int(parts[1][1:])  # Remove 'r' prefix
            col = int(parts[2][1:])  # Remove 'c' prefix
            
            if row < 0 or col < 0:
                raise ValueError(
                    f"Row and column indices must be non-negative, got row={row}, col={col}"
                )
        except (ValueError, IndexError) as e:
            raise ValueError(
                f"Invalid segment_id format: {v}. Expected 'seg_r{{row}}_c{{col}}'"
            ) from e
        
        return v
    
    @field_validator('bbox')
    @classmethod
    def validate_bbox_format(cls, v: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Validate that bbox has correct format and logical values.
        
        Ensures:
        - All coordinates are non-negative
        - x2 > x1 (width is positive)
        - y2 > y1 (height is positive)
        """
        x1, y1, x2, y2 = v
        
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            raise ValueError(
                f"All bbox coordinates must be non-negative, got: {v}"
            )
        
        if x2 <= x1:
            raise ValueError(
                f"bbox x2 ({x2}) must be greater than x1 ({x1})"
            )
        
        if y2 <= y1:
            raise ValueError(
                f"bbox y2 ({y2}) must be greater than y1 ({y1})"
            )
        
        return v
    
    @model_validator(mode='after')
    def validate_bbox_consistency(self) -> 'Segment':
        """
        Validate that bbox matches x, y, width, height.
        
        Ensures:
        - bbox[0] == x
        - bbox[1] == y
        - bbox[2] == x + width
        - bbox[3] == y + height
        """
        expected_bbox = (self.x, self.y, self.x + self.width, self.y + self.height)
        
        if self.bbox != expected_bbox:
            raise ValueError(
                f"bbox inconsistency: bbox={self.bbox} doesn't match "
                f"expected {expected_bbox} from x={self.x}, y={self.y}, "
                f"width={self.width}, height={self.height}"
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_data_dimensions(self) -> 'Segment':
        """
        Validate that data dimensions match width/height (if data is provided).
        """
        if self.data is not None:
            if self.data.ndim == 2:
                # Grayscale image
                data_height, data_width = self.data.shape
            elif self.data.ndim == 3:
                # Color image
                data_height, data_width = self.data.shape[:2]
            else:
                raise ValueError(
                    f"data must be 2D or 3D array, got shape: {self.data.shape}"
                )
            
            if data_height != self.height or data_width != self.width:
                raise ValueError(
                    f"data dimensions ({data_width}×{data_height}) don't match "
                    f"segment dimensions ({self.width}×{self.height})"
                )
        
        return self
    
    def area(self) -> int:
        """
        Calculate the area of the segment in pixels.
        
        Returns:
            int: Area in pixels (width × height)
        
        Examples:
            >>> segment = Segment(..., width=256, height=256, ...)
            >>> segment.area()
            65536
        """
        return self.width * self.height
    
    def center(self) -> Tuple[float, float]:
        """
        Calculate the center coordinates of the segment.
        
        Returns:
            Tuple[float, float]: (center_x, center_y)
        
        Examples:
            >>> segment = Segment(..., x=0, y=0, width=256, height=256, ...)
            >>> segment.center()
            (128.0, 128.0)
        """
        center_x = self.x + self.width / 2.0
        center_y = self.y + self.height / 2.0
        return (center_x, center_y)
    
    def contains_point(self, px: int, py: int) -> bool:
        """
        Check if a point is inside this segment.
        
        Args:
            px: X-coordinate of the point
            py: Y-coordinate of the point
        
        Returns:
            bool: True if point is inside segment
        
        Examples:
            >>> segment = Segment(..., x=0, y=0, width=256, height=256, ...)
            >>> segment.contains_point(100, 100)
            True
            >>> segment.contains_point(300, 300)
            False
        """
        return (self.x <= px < self.x + self.width and
                self.y <= py < self.y + self.height)
    
    def overlaps(self, other: 'Segment') -> bool:
        """
        Check if this segment overlaps with another segment.
        
        Args:
            other: Another Segment to check for overlap
        
        Returns:
            bool: True if segments overlap
        
        Examples:
            >>> seg1 = Segment(..., x=0, y=0, width=256, height=256, ...)
            >>> seg2 = Segment(..., x=200, y=200, width=256, height=256, ...)
            >>> seg1.overlaps(seg2)
            True
        """
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox
        
        return not (x1_max <= x2_min or x2_max <= x1_min or
                   y1_max <= y2_min or y2_max <= y1_min)
    
    def overlap_area(self, other: 'Segment') -> int:
        """
        Calculate the overlapping area with another segment.
        
        Args:
            other: Another Segment to calculate overlap with
        
        Returns:
            int: Overlapping area in pixels (0 if no overlap)
        
        Examples:
            >>> seg1 = Segment(..., x=0, y=0, width=256, height=256, ...)
            >>> seg2 = Segment(..., x=200, y=200, width=256, height=256, ...)
            >>> seg1.overlap_area(seg2)
            3136  # 56×56 pixels overlap
        """
        if not self.overlaps(other):
            return 0
        
        x1_min, y1_min, x1_max, y1_max = self.bbox
        x2_min, y2_min, x2_max, y2_max = other.bbox
        
        overlap_x_min = max(x1_min, x2_min)
        overlap_y_min = max(y1_min, y2_min)
        overlap_x_max = min(x1_max, x2_max)
        overlap_y_max = min(y1_max, y2_max)
        
        overlap_width = overlap_x_max - overlap_x_min
        overlap_height = overlap_y_max - overlap_y_min
        
        return overlap_width * overlap_height
    
    def get_row_col(self) -> Tuple[int, int]:
        """
        Extract row and column indices from segment_id.
        
        Returns:
            Tuple[int, int]: (row, col) indices
        
        Examples:
            >>> segment = Segment(segment_id="seg_r2_c3", ...)
            >>> segment.get_row_col()
            (2, 3)
        """
        parts = self.segment_id.split('_')
        row = int(parts[1][1:])  # Remove 'r' prefix
        col = int(parts[2][1:])  # Remove 'c' prefix
        return (row, col)
    
    def to_dict_serializable(self) -> dict:
        """
        Convert to JSON-serializable dictionary (excludes numpy arrays).
        
        Returns:
            dict: Serializable representation
        
        Examples:
            >>> segment = Segment(...)
            >>> data = segment.to_dict_serializable()
            >>> import json
            >>> json_str = json.dumps(data)
        """
        data = self.model_dump(exclude={'data'})
        data['has_data'] = self.data is not None
        if self.data is not None:
            data['data_shape'] = self.data.shape
            data['data_dtype'] = str(self.data.dtype)
        return data


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

    def calculate_entropy(self, image_data: np.ndarray) -> float:
        """
        Calculate the Shannon entropy of the image segment.
        
        Args:
            image_data: Input image segment (gray or color)
            
        Returns:
            float: Entropy value (higher means more information)
        """
        if image_data is None or image_data.size == 0:
            return 0.0
            
        if image_data.ndim == 3:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_data

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        return float(entropy)

    def calculate_variance(self, image_data: np.ndarray) -> float:
        """
        Calculate the variance of the image segment (measure of contrast).
        
        Args:
            image_data: Input image segment
            
        Returns:
            float: Variance value
        """
        if image_data is None or image_data.size == 0:
            return 0.0
            
        if image_data.ndim == 3:
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_data
            
        return float(np.var(gray))

    def _calculate_grid(self, image_height: int, image_width: int) -> Tuple[int, int]:
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
            im_height, im_width = image.shape[:2]
        else:
            im_height, im_width = image.shape

        # Calculate grid of segments
        n_rows, n_cols = self._calculate_grid(im_height, im_width)
        
        segments = []  # List to hold segment metadata

        for row in range(n_rows):  # Iterate over rows: vertical position
            for col in range(n_cols):  # Iterate over columns: horizontal position
                # Calculate top-left corner of segment
                x = col * self.stride
                y = row * self.stride

                # Calculate actual width and height to avoid exceeding image boundaries
                seg_width = min(self.segment_size, im_width - x)
                seg_height = min(self.segment_size, im_height - y)

                # Define bounding box as (x1, y1, x2, y2)
                bbox = (x, y, x + seg_width, y + seg_height)

                # Extract data for metric calculation
                # Note: We extract data here temporarily to calculate metrics, 
                # but we don't store it in the segment unless requested to save memory
                temp_segment = Segment(
                    segment_id=f"seg_r{row}_c{col}",
                    x=x,
                    y=y,
                    width=seg_width,
                    height=seg_height,
                    bbox=bbox
                )
                temp_data = self.extract_segment_data(image, temp_segment)
                
                entropy = self.calculate_entropy(temp_data)
                variance = self.calculate_variance(temp_data)

                # Create Segment object with metadata (Pydantic validates automatically!)
                each_segment = Segment(
                    segment_id=f"seg_r{row}_c{col}",
                    x=x,
                    y=y,
                    width=seg_width,
                    height=seg_height,
                    bbox=bbox,
                    data=None,  # Don't store data to save memory
                    entropy=entropy,
                    variance=variance
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


# ============================================================================
# TESTING CODE
# ============================================================================

if __name__ == "__main__":
    import os

    print("=" * 70)
    print("PYDANTIC SEGMENT VALIDATION TESTS")
    print("=" * 70)
    
    # Test 1: Valid segment creation
    print("\n[Test 1] Creating valid segment...")
    try:
        segment = Segment(
            segment_id="seg_r0_c0",
            x=0, y=0,
            width=256, height=256,
            bbox=(0, 0, 256, 256),
            entropy=6.5,
            variance=1200.5
        )
        print(f"✅ Valid segment created: {segment.segment_id}")
        print(f"   Area: {segment.area()} pixels")
        print(f"   Center: {segment.center()}")
        print(f"   Row, Col: {segment.get_row_col()}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Invalid dimensions (negative width)
    print("\n[Test 2] Testing negative width validation...")
    try:
        invalid_segment = Segment(
            segment_id="seg_r0_c0",
            x=0, y=0,
            width=-50,  # ❌ Invalid!
            height=256,
            bbox=(0, 0, 256, 256)
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught error: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    # Test 3: Invalid bbox consistency
    print("\n[Test 3] Testing bbox consistency validation...")
    try:
        invalid_segment = Segment(
            segment_id="seg_r0_c0",
            x=100, y=200,
            width=256, height=256,
            bbox=(999, 999, 1000, 1000)  # ❌ Doesn't match x, y, width, height!
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught bbox inconsistency: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    # Test 4: Invalid segment_id format
    print("\n[Test 4] Testing segment_id format validation...")
    try:
        invalid_segment = Segment(
            segment_id="invalid_format",  # ❌ Wrong format!
            x=0, y=0,
            width=256, height=256,
            bbox=(0, 0, 256, 256)
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught invalid segment_id: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    # Test 5: Invalid entropy range
    print("\n[Test 5] Testing entropy range validation...")
    try:
        invalid_segment = Segment(
            segment_id="seg_r0_c0",
            x=0, y=0,
            width=256, height=256,
            bbox=(0, 0, 256, 256),
            entropy=999.0  # ❌ Impossible value (max is 8 for 8-bit images)!
        )
        print(f"❌ Should have failed validation!")
    except Exception as e:
        print(f"✅ Validation caught invalid entropy: {type(e).__name__}")
        print(f"   Message: {str(e)[:80]}...")
    
    # Test 6: JSON serialization
    print("\n[Test 6] Testing JSON serialization...")
    try:
        segment = Segment(
            segment_id="seg_r2_c3",
            x=100, y=200,
            width=256, height=256,
            bbox=(100, 200, 356, 456),
            entropy=6.5,
            variance=1200.5
        )
        
        # Serialize to dict
        data = segment.to_dict_serializable()
        print(f"✅ Serialized to dict: {list(data.keys())}")
        
        # Serialize to JSON
        json_str = segment.model_dump_json(exclude={'data'})
        print(f"✅ Serialized to JSON ({len(json_str)} chars)")
        print(f"   Preview: {json_str[:100]}...")
        
        # Deserialize from JSON
        loaded_segment = Segment.model_validate_json(json_str)
        print(f"✅ Deserialized from JSON: {loaded_segment.segment_id}")
        
    except Exception as e:
        print(f"❌ Serialization error: {e}")
    
    # Test 7: Helper methods
    print("\n[Test 7] Testing helper methods...")
    try:
        seg1 = Segment(
            segment_id="seg_r0_c0",
            x=0, y=0,
            width=256, height=256,
            bbox=(0, 0, 256, 256)
        )
        
        seg2 = Segment(
            segment_id="seg_r0_c1",
            x=230, y=0,  # Overlaps with seg1 by 26 pixels
            width=256, height=256,
            bbox=(230, 0, 486, 256)
        )
        
        print(f"✅ seg1.contains_point(100, 100): {seg1.contains_point(100, 100)}")
        print(f"✅ seg1.contains_point(300, 300): {seg1.contains_point(300, 300)}")
        print(f"✅ seg1.overlaps(seg2): {seg1.overlaps(seg2)}")
        print(f"✅ seg1.overlap_area(seg2): {seg1.overlap_area(seg2)} pixels")
        
    except Exception as e:
        print(f"❌ Helper method error: {e}")
    
    print("\n" + "=" * 70)
    print("IMAGE SEGMENTATION TEST")
    print("=" * 70)

    # Load test image
    image_path = "data/test-docs/input.png"

    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at {image_path}")
        print("Please ensure the file exists or update the path.")
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Failed to load image from {image_path}")
        else:
            print(f"✓ Loaded image: {image.shape} (H, W, C)")

            # Create segmentor
            segmentor = ImageSegmentor(segment_size=256, overlap_percentage=0.10)
            print(f"✓ Created segmentor: segment_size={segmentor.segment_size}, overlap={segmentor.overlap_percentage}")
            print(f"  - Overlap pixels: {segmentor.overlap_pixels}")
            print(f"  - Stride: {segmentor.stride}")

            # Calculate grid
            num_rows, num_cols = segmentor._calculate_grid(image.shape[0], image.shape[1])
            print(f"\n✓ Grid calculation: {num_rows} rows × {num_cols} cols = {num_rows * num_cols} segments")

            # Segment the image
            segments = segmentor.segment_image(image)
            print(f"✓ Created {len(segments)} Pydantic-validated segments\n")

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
            output_path = "data/test-docs/segmentation_output_pydantic.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, vis_image)
            print(f"✓ Saved visualization to: {output_path}")

            # Display segment details (first 5)
            print(f"\n=== First 5 Segments (Pydantic-validated) ===")
            for segment in segments[:5]:
                print(f"{segment.segment_id}:")
                print(f"  Position: ({segment.x}, {segment.y})")
                print(f"  Size: {segment.width}×{segment.height}")
                print(f"  Bbox: {segment.bbox}")
                print(f"  Entropy: {segment.entropy:.2f}")
                print(f"  Variance: {segment.variance:.2f}")
                print(f"  Area: {segment.area()} pixels")
                print(f"  Center: {segment.center()}")

            # Test extract_segment_data on first segment
            print(f"\n=== Testing extract_segment_data ===")
            first_seg = segments[0]
            seg_data = segmentor.extract_segment_data(image, first_seg)
            print(f"✓ Extracted {first_seg.segment_id}: shape={seg_data.shape}")

            # Save first segment as separate image
            seg_output_path = "data/test-docs/first_segment_pydantic.png"
            cv2.imwrite(seg_output_path, seg_data)
            print(f"✓ Saved first segment to: {seg_output_path}")

            print("\n✅ Pydantic segmentation test complete!")