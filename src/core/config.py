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
    ALIGNER_ENTROPY_THRESHOLD: float = 0.5
    ALIGNER_VARIANCE_THRESHOLD: float = 5.0

    # Validator Settings
    VALIDATOR_OUTLIER_THRESHOLD: float = 1.0
    VALIDATOR_CONSISTENCY_THRESHOLD: float = 0.7

    # Comparator Settings
    COMPARATOR_SIGNIFICANCE_THRESHOLD: float = 0.2
    COMPARATOR_MATCH_THRESHOLD: float = 0.95
    COMPARATOR_CHANGE_THRESHOLD: float = 1.0

settings = Settings()
