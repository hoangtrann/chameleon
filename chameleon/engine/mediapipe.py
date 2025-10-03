"""Modern MediaPipe segmentation engine with GPU delegation support."""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

MEDIAPIPE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"


class MediaPipeEngine:
    """MediaPipe segmentation engine with GPU acceleration."""

    def __init__(self, use_gpu: bool = True, model_path: str | None = None):
        """Initialize MediaPipe segmentation engine.

        Args:
            use_gpu: Whether to use GPU delegation for acceleration
            model_path: Path to the selfie segmentation TFLite model.
                       If None, will attempt to download the default model.
        """
        self.use_gpu = "auto"
        self.model_path = self._get_or_download_model(model_path)
        self.segmenter = None
        self._init_segmenter()

    def _get_or_download_model(self, model_path: str | None) -> str:
        """Get model path, download if not cached.

        Args:
            model_path: Optional custom model path

        Returns:
            Path to model file
        """
        if model_path:
            return model_path

        cache_dir = Path.home() / ".cache" / "chameleon" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_file = cache_dir / "selfie_segmenter.tflite"

        if not model_file.exists():
            self._download_model(model_file)

        return str(model_file)

    def _download_model(self, target: Path):
        """Download MediaPipe selfie segmentation model.

        Args:
            target: Target path for downloaded model
        """
        import urllib.request

        logger.info("Downloading MediaPipe selfie segmentation model...")
        try:
            urllib.request.urlretrieve(MEDIAPIPE_MODEL_URL, str(target))
            logger.info("Model downloaded to %s", target)
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}") from e

    def _init_segmenter(self):
        """Initialize MediaPipe ImageSegmenter with GPU delegation."""
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            logger.info("Initializing MediaPipe segmentation engine...")
            logger.info("Model path: %s", self.model_path)

            # Configure base options with GPU delegation
            delegate = (
                python.BaseOptions.Delegate.GPU
                if (self.use_gpu or self.use_gpu == "auto")
                else python.BaseOptions.Delegate.CPU
            )

            base_options = python.BaseOptions(model_asset_path=self.model_path, delegate=delegate)

            # Configure the ImageSegmenter task
            options = vision.ImageSegmenterOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                output_confidence_masks=True,
            )

            # Create the segmenter
            self.segmenter = vision.ImageSegmenter.create_from_options(options)

            logger.info(
                "MediaPipe initialized with %s delegation",
                "GPU" if self.use_gpu else "CPU",
            )

        except ImportError:
            raise RuntimeError(
                "mediapipe is required for MediaPipe engine. Install with: pip install mediapipe"
            ) from None

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Run segmentation and return binary mask.

        Args:
            frame: Input frame (height, width, 3) in BGR format

        Returns:
            Binary mask (height, width) with values 0-1
        """
        import cv2
        import mediapipe as mp

        # Store original dimensions
        orig_h, orig_w = frame.shape[:2]

        # Downsample to model's optimal input size (256x144 landscape)
        # Maintain aspect ratio by using width-based scaling
        target_w = 256
        target_h = int(target_w * orig_h / orig_w)
        small_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Perform segmentation
        segmentation_result = self.segmenter.segment(mp_image)

        # Extract mask and upscale back to original resolution
        mask = None
        if segmentation_result.confidence_masks:
            # confidence_masks[0] is background probability
            # We need: 0 = person (keep sharp), 1 = background (blur)
            # So use background mask directly
            background_mask = segmentation_result.confidence_masks[0].numpy_view()
            mask = background_mask.astype(np.float32)
        elif segmentation_result.category_mask is not None:
            category_mask = segmentation_result.category_mask.numpy_view()
            # Invert: 1 for background (blur), 0 for person (keep sharp)
            mask = (category_mask == 0).astype(np.float32)

        # If no mask available, return empty mask
        if mask is None:
            return np.zeros((orig_h, orig_w), dtype=np.float32)

        # Upscale mask back to original resolution
        mask_upscaled = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return mask_upscaled

    def close(self):
        """Close the segmentation engine and release resources."""
        if self.segmenter:
            self.segmenter.close()
            self.segmenter = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
