"""Configuration management with Pydantic models."""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class HWAccelMethod(str, Enum):
    """Hardware acceleration methods."""

    NONE = "none"
    CUDA = "cuda"
    VDPAU = "vdpau"
    VAAPI = "vaapi"
    AUTO = "auto"


class SegmentationModel(str, Enum):
    """Segmentation model types."""

    YOLO11N = "yolo11n-seg"
    FASTSAM_S = "fastsam-s"
    MEDIAPIPE = "mediapipe"


class FilterConfig(BaseModel):
    """Configuration for image filters and effects."""

    file: str | None = None
    hologram: bool = False
    blur: int | None = Field(None, ge=0, le=100)
    solid: tuple[int, int, int] | None = None  # BGR color
    cmap: str | None = None
    brightness: int | None = Field(None, ge=0, le=200)
    opacity: int | None = Field(None, ge=0, le=100)
    tile: bool = False
    crop: bool = False
    disabled: bool = False
    mask_update_speed: float | None = Field(None, ge=0.0, le=1.0)
    foreground_file: str | None = None
    mask_file: str | None = None

    @field_validator("blur")
    @classmethod
    def validate_blur(cls, v):
        """Ensure blur value is in valid range."""
        if v is not None:
            return max(0, min(100, v))
        return v

    @field_validator("opacity")
    @classmethod
    def validate_opacity(cls, v):
        """Ensure opacity value is in valid range."""
        if v is not None:
            return max(0, min(100, v))
        return v

    @field_validator("brightness")
    @classmethod
    def validate_brightness(cls, v):
        """Ensure brightness value is in valid range."""
        if v is not None:
            return max(0, min(200, v))
        return v


class CameraConfig(BaseModel):
    """Camera device configuration."""

    device: str = Field(default="/dev/video0", description="Camera device path")
    width: int = Field(default=1280, ge=1, le=3840)
    height: int = Field(default=720, ge=1, le=2160)
    fps: int = Field(default=30, ge=1, le=120)
    codec: str = Field(default="MJPG", description="Video codec fourcc code")

    @field_validator("codec")
    @classmethod
    def validate_codec(cls, v):
        """Ensure codec is valid length."""
        if len(v) != 4:
            raise ValueError("Codec must be a 4-character fourcc code")
        return v.upper()


class ProcessingConfig(BaseModel):
    """Image processing configuration."""

    use_sigmoid: bool = Field(default=False, description="Apply sigmoid to mask")
    threshold: int = Field(default=75, ge=0, le=100, description="Foreground threshold percentage")
    enable_postprocess: bool = Field(default=True, description="Enable mask postprocessing")
    model: SegmentationModel = Field(default=SegmentationModel.MEDIAPIPE)
    model_selection: int = Field(default=1, ge=0, le=1, description="MediaPipe model selection")
    hwaccel: HWAccelMethod = Field(default=HWAccelMethod.AUTO)

    # New optimization options
    use_clahe: bool = Field(
        default=False, description="Enable CLAHE for better low-light performance"
    )
    adaptive_threshold: bool = Field(
        default=True, description="Use Otsu's adaptive threshold (better for varying lighting)"
    )
    temporal_smoothing: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Temporal mask smoothing (0-1, 0=disabled, 0.3=recommended)",
    )
    fast_blur: bool = Field(
        default=True, description="Use downscale-blur-upscale optimization (3-4x faster)"
    )

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v):
        """Ensure threshold is in valid range."""
        return max(0, min(100, v))


class OptimizationProfile(BaseModel):
    """Hardware-specific optimization profile."""

    model: SegmentationModel
    resolution: tuple[int, int]
    fps: int
    blur_kernel: int
    use_gpu: bool = False


# Predefined optimization profiles
OPTIMIZATION_PROFILES: dict[str, OptimizationProfile] = {
    "low": OptimizationProfile(
        model=SegmentationModel.YOLO11N,
        resolution=(640, 480),
        fps=15,
        blur_kernel=11,
        use_gpu=False,
    ),
    "medium": OptimizationProfile(
        model=SegmentationModel.FASTSAM_S,
        resolution=(1280, 720),
        fps=30,
        blur_kernel=21,
        use_gpu=False,
    ),
    "high": OptimizationProfile(
        model=SegmentationModel.YOLO11N,
        resolution=(1920, 1080),
        fps=60,
        blur_kernel=31,
        use_gpu=True,
    ),
}


class Config(BaseModel):
    """Main application configuration."""

    # Camera settings
    real_camera: CameraConfig = Field(default_factory=CameraConfig)
    virtual_camera: str = Field(default="/dev/video2", description="Virtual camera device path")

    # Processing settings
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)

    # Filter configurations
    selfie_filter: FilterConfig = Field(default_factory=FilterConfig)
    background_filter: FilterConfig = Field(
        default_factory=lambda: FilterConfig(file="background.jpg")
    )
    mask_filter: FilterConfig = Field(default_factory=FilterConfig)

    # Runtime settings
    no_ondemand: bool = Field(
        default=False,
        description="Continue processing when no app is using virtual webcam",
    )

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "Config":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_ini_file(cls, ini_path: Path) -> "Config":
        """Load configuration from INI file.

        This is a placeholder for INI parsing that will be implemented
        to maintain backward compatibility with existing config files.
        """
        # TODO: Implement INI parsing in future phase
        raise NotImplementedError("INI file parsing not yet implemented")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.model_dump()
