"""
Configuration settings for ShiftVis core components.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    
    Environment variables are case-insensitive for flexibility:
    - PREPROCESSOR_TARGET_COLOR_MODE (uppercase - recommended)
    - preprocessor_target_color_mode (lowercase - also works)
    - Preprocessor_Target_Color_Mode (mixed case - also works)
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # Allow case-insensitive env var lookups
        extra="ignore"
    )

    # Preprocessor Settings
    PREPROCESSOR_TARGET_COLOR_MODE: str = "BGR"
    PREPROCESSOR_MIN_DIM: int = 256
    PREPROCESSOR_MAX_DIM: int = 10000
    PREPROCESSOR_ALLOW_RESIZE: bool = True

    # Segmentor Settings
    SEGMENT_SIZE: int = 256
    SEGMENT_OVERLAP_PERCENTAGE: float = 0.10

    # Aligner Settings
    ALIGNER_LOCAL_SEARCH_RADIUS: int = 20
    ALIGNER_LARGE_SEARCH_RADIUS: int = 50
    ALIGNER_TIER1_THRESHOLD: float = 0.98
    ALIGNER_TIER2_THRESHOLD: float = 0.90
    ALIGNER_ENTROPY_THRESHOLD: float = 1.0
    ALIGNER_VARIANCE_THRESHOLD: float = 25.0

    # Validator Settings
    VALIDATOR_OUTLIER_THRESHOLD: float = 1.0
    VALIDATOR_CONSISTENCY_THRESHOLD: float = 0.7

    # Comparator Settings
    COMPARATOR_PIXEL_DIFF_THRESHOLD: float = 0.11764705882352941  # 30/255
    COMPARATOR_MIN_CHANGE_PIXELS: int = 128
    COMPARATOR_MIN_CONTOUR_AREA: int = 50
    COMPARATOR_MORPHOLOGY_KERNEL_SIZE: int = 5
    COMPARATOR_MERGE_DISTANCE: int = 10
    COMPARATOR_CHANGE_CLASSIFICATION_DELTA: float = 0.1
    # legacy
    COMPARATOR_SIGNIFICANCE_THRESHOLD: float = 0.2
    COMPARATOR_MATCH_THRESHOLD: float = 0.95
    COMPARATOR_CHANGE_THRESHOLD: float = 1.0

    # Concurrency Settings
    PERFORMANCE_ENABLE_PARALLEL_ALIGNMENT: bool = False
    PERFORMANCE_ENABLE_PARALLEL_COMPARISON: bool = False
    PERFORMANCE_MAX_WORKERS: int = 8
    PERFORMANCE_MAX_IMAGE_DIMENSION: int = 10000
    PERFORMANCE_CHUNK_SIZE: int = 10

    # Visualization Settings
    VISUALIZATION_HEATMAP_COLORMAP: str = "jet"
    VISUALIZATION_ADDITION_COLOR: str = "0,255,0"  # RGB as string (green)
    VISUALIZATION_DELETION_COLOR: str = "255,0,0"  # RGB as string (red)
    VISUALIZATION_MODIFICATION_COLOR: str = "0,0,255"  # RGB as string (blue)
    VISUALIZATION_BBOX_THICKNESS: int = 2
    VISUALIZATION_OVERLAY_ALPHA: float = 0.3

    # Storage Settings
    STORAGE_OUTPUT_DIR: str = "./data/results"
    STORAGE_SAVE_PIXEL_MASKS: bool = True
    STORAGE_MASK_STORAGE_FORMAT: str = "npz"  # "npz" or "png"

    # Loggers Settings
    LOGGING_SAVE_INTERMEDIATE_RESULTS: bool = False
    LOGGING_INTERMEDIATE_OUTPUT_DIR: str = "./debug_output"
    LOGGING_LOG_ALIGNMENT_FAILURES: bool = True
    LOGGING_LOG_VALIDATION_OUTLIERS: bool = True
    LOGGING_TIMING_ENABLED: bool = True

    # Debug Settings
    DEBUG_MODE: bool = False
    LOG_LEVEL: str = "INFO"

    # Helpers
    def get_addition_color_rgb(self) -> tuple[int, int, int]:
        """Parse addition color from string to RGB tuple."""
        return tuple(map(int, self.VISUALIZATION_ADDITION_COLOR.split(',')))
    
    def get_deletion_color_rgb(self) -> tuple[int, int, int]:
        """Parse deletion color from string to RGB tuple."""
        return tuple(map(int, self.VISUALIZATION_DELETION_COLOR.split(',')))
    
    def get_modification_color_rgb(self) -> tuple[int, int, int]:
        """Parse modification color from string to RGB tuple."""
        return tuple(map(int, self.VISUALIZATION_MODIFICATION_COLOR.split(',')))
    
    def get_segment_overlap_pixels(self) -> int:
        """Convert overlap percentage to pixels based on segment size."""
        return int(self.SEGMENT_SIZE * self.SEGMENT_OVERLAP_PERCENTAGE)

settings = Settings()
