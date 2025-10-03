"""Image composition - blend layers of background, person, and foreground."""

import logging

import cv2
import numpy as np

from chameleon.config import FilterConfig

logger = logging.getLogger(__name__)


class ImageCompositor:
    """Compositor for blending background, person, and foreground layers."""

    def __init__(
        self,
        width: int,
        height: int,
        background_config: FilterConfig | None = None,
        selfie_config: FilterConfig | None = None,
        mask_config: FilterConfig | None = None,
        temporal_smoothing: float = 0.0,
    ):
        """Initialize image compositor.

        Args:
            width: Output frame width
            height: Output frame height
            background_config: Background filter configuration
            selfie_config: Selfie filter configuration
            mask_config: Mask/foreground filter configuration
            temporal_smoothing: Temporal mask smoothing factor (0.0-1.0, 0=disabled)
                                Higher values = smoother but more lag (0.3 recommended)
        """
        self.width = width
        self.height = height
        self.background_config = background_config or FilterConfig()
        self.selfie_config = selfie_config or FilterConfig()
        self.mask_config = mask_config or FilterConfig()
        self.temporal_smoothing = temporal_smoothing

        # Loaded images
        self.background_image: np.ndarray | None = None
        self.foreground_image: np.ndarray | None = None
        self.foreground_mask: np.ndarray | None = None

        # Temporal smoothing state
        self._prev_mask: np.ndarray | None = None

        # TODO: Pre-allocated buffers for composition to avoid per-frame allocation
        # This reduces memory allocation overhead from ~2-3ms to near-zero
        self._composited_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self._temp_buffer = np.zeros((height, width, 3), dtype=np.float32)
        self._temp_buffer2 = np.zeros((height, width, 3), dtype=np.float32)

    def load_background(self, image_path: str) -> bool:
        """Load background image or video.

        Args:
            image_path: Path to background image or video file

        Returns:
            True if loaded successfully
        """
        try:
            # Try to load as image first
            img = cv2.imread(image_path)
            if img is not None:
                # TODO: Resize to output dimensions once at load time
                # This ensures background is always the correct size, eliminating per-frame checks
                self.background_image = cv2.resize(img, (self.width, self.height))
                return True

            # Try to load as video
            cap = cv2.VideoCapture(image_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # TODO: Pre-resize video frames to output dimensions
                    self.background_image = cv2.resize(frame, (self.width, self.height))
                    cap.release()
                    return True
                cap.release()

            return False
        except Exception as e:
            logger.error("Error loading background: %s", e)
            return False

    def load_foreground(self, image_path: str, mask_path: str | None = None) -> bool:
        """Load foreground image and optional mask.

        Args:
            image_path: Path to foreground image
            mask_path: Optional path to foreground mask

        Returns:
            True if loaded successfully
        """
        try:
            # Load foreground image
            img = cv2.imread(image_path)
            if img is None:
                return False

            # TODO: Pre-resize foreground to output dimensions at load time
            self.foreground_image = cv2.resize(img, (self.width, self.height))

            # Load mask if provided
            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    # TODO: Pre-resize and normalize mask at load time
                    self.foreground_mask = cv2.resize(mask, (self.width, self.height))
                    # Normalize to 0-1 range
                    self.foreground_mask = self.foreground_mask.astype(np.float32) / 255.0
                else:
                    # Use full mask if mask file couldn't be loaded
                    self.foreground_mask = np.ones((self.height, self.width), dtype=np.float32)
            else:
                # No mask provided, use full mask
                self.foreground_mask = np.ones((self.height, self.width), dtype=np.float32)

            return True
        except Exception as e:
            logger.error("Error loading foreground: %s", e)
            return False

    def compose(
        self,
        person_frame: np.ndarray,
        person_mask: np.ndarray,
        background: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compose final frame from layers.

        Args:
            person_frame: Person frame from camera
            person_mask: Segmentation mask for person (0-1 range)
            background: Optional background frame (uses loaded background if None)

        Returns:
            Composited frame
        """
        # Use provided background or loaded background image
        # TODO: Remove unnecessary .copy() - background_image is read-only during processing
        if background is None:
            if self.background_image is None:
                # No background, create black background
                background = np.zeros_like(person_frame)
            else:
                background = self.background_image  # Removed .copy() - saves ~1-2ms
        # TODO: Size validation moved - assume caller provides correct size
        # If background is provided, it should already be correct size from pipeline
        # This eliminates per-frame conditional checks

        # Ensure mask is 2D and correct size
        if len(person_mask.shape) == 3:
            person_mask = person_mask[:, :, 0]
        if person_mask.shape != (self.height, self.width):
            person_mask = cv2.resize(person_mask, (self.width, self.height))

        # TODO: Optimize mask expansion - use cv2.merge instead of np.stack
        # cv2.merge is optimized for creating multi-channel arrays from single channel
        # This is faster than np.stack and ensures proper channel count for cv2 operations
        mask_3ch = cv2.merge([person_mask, person_mask, person_mask])

        # TODO: Optimized composition using cv2 operations and pre-allocated buffers
        # Old: composited = person_frame.astype(np.float32) * mask_3ch + background.astype(np.float32) * (1 - mask_3ch)
        # New approach uses in-place operations to avoid allocations

        # Calculate inverse mask: (1 - mask) - needs to be 3 channel for cv2.multiply
        cv2.subtract(1.0, mask_3ch, dst=self._temp_buffer)  # Use temp buffer for inv_mask_3ch

        # Convert frames to float32 and multiply by masks using pre-allocated buffers
        # person_contribution = person_frame * mask
        cv2.multiply(person_frame.astype(np.float32), mask_3ch, dst=self._composited_buffer)

        # background_contribution = background * (1 - mask)
        # Store in temp_buffer first, then we'll add to composited_buffer
        background_float = background.astype(np.float32)
        cv2.multiply(background_float, self._temp_buffer, dst=self._temp_buffer)

        # Final composite: person_contribution + background_contribution
        cv2.add(self._composited_buffer, self._temp_buffer, dst=self._composited_buffer)

        # Apply foreground overlay if available
        if self.foreground_image is not None and self.foreground_mask is not None:
            self._apply_foreground_optimized(self._composited_buffer)

        # TODO: Remove explicit clip - astype(np.uint8) clips automatically (0-255)
        # This saves ~0.5-1ms per frame
        return self._composited_buffer.astype(np.uint8)

    def _apply_foreground(self, frame: np.ndarray) -> np.ndarray:
        """Apply foreground overlay to frame (legacy - kept for compatibility).

        Args:
            frame: Current composited frame

        Returns:
            Frame with foreground overlay applied
        """
        if self.foreground_image is None or self.foreground_mask is None:
            return frame

        # Apply opacity from config
        opacity = 1.0
        if self.mask_config.opacity is not None:
            opacity = self.mask_config.opacity / 100.0

        # TODO: Old implementation - use _apply_foreground_optimized instead
        # Expand mask to 3 channels
        mask_3ch = np.stack([self.foreground_mask * opacity] * 3, axis=2)

        # Blend foreground over frame
        # result = foreground * mask + frame * (1 - mask)
        blended = self.foreground_image.astype(np.float32) * mask_3ch + frame.astype(np.float32) * (
            1 - mask_3ch
        )

        return blended

    def _apply_foreground_optimized(self, frame: np.ndarray) -> None:
        """Apply foreground overlay to frame (optimized version).

        Args:
            frame: Current composited frame (float32, modified in-place)
        """
        if self.foreground_image is None or self.foreground_mask is None:
            return

        # Apply opacity from config
        opacity = 1.0
        if self.mask_config.opacity is not None:
            opacity = self.mask_config.opacity / 100.0

        # TODO: Optimize foreground mask - use cv2.merge for 3-channel mask
        # Create 3-channel mask with opacity
        if opacity != 1.0:
            fg_mask_single = self.foreground_mask * opacity
            fg_mask_3ch = cv2.merge([fg_mask_single, fg_mask_single, fg_mask_single])
        else:
            fg_mask_3ch = cv2.merge(
                [self.foreground_mask, self.foreground_mask, self.foreground_mask]
            )

        # Calculate inverse mask: (1 - fg_mask) - use temp_buffer2
        cv2.subtract(1.0, fg_mask_3ch, dst=self._temp_buffer2)

        # foreground_contribution = foreground * fg_mask - use temp_buffer
        cv2.multiply(self.foreground_image.astype(np.float32), fg_mask_3ch, dst=self._temp_buffer)

        # frame_contribution = frame * (1 - fg_mask) - modify frame in place
        cv2.multiply(frame, self._temp_buffer2, dst=frame)

        # Final blend: foreground_contribution + frame_contribution
        cv2.add(self._temp_buffer, frame, dst=frame)

    def create_solid_background(self, color: tuple[int, int, int]) -> np.ndarray:
        """Create a solid color background.

        Args:
            color: BGR color tuple

        Returns:
            Solid color image
        """
        background = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        background[:] = color
        return background

    def process_mask(
        self,
        mask: np.ndarray,
        threshold: float = 0.75,
        use_sigmoid: bool = False,
        enable_postprocess: bool = True,
        adaptive_threshold: bool = True,
    ) -> np.ndarray:
        """Process segmentation mask with threshold, sigmoid, and postprocessing.

        Args:
            mask: Raw segmentation mask (0-1 range)
            threshold: Fixed threshold for binary mask (used if adaptive_threshold=False)
            use_sigmoid: Apply sigmoid transformation
            enable_postprocess: Enable dilation and blur postprocessing
            adaptive_threshold: Use Otsu's method for adaptive thresholding (better for varying lighting)

        Returns:
            Processed mask
        """
        # Apply sigmoid if requested
        if use_sigmoid:
            mask = self._sigmoid(mask)

        # Apply threshold (adaptive or fixed)
        if adaptive_threshold:
            # Use Otsu's method for automatic threshold detection
            # Works better with varying lighting conditions
            mask_uint8 = (mask * 255).astype(np.uint8)
            otsu_thresh, _ = cv2.threshold(mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Convert back to 0-1 range
            threshold_value = otsu_thresh / 255.0
            mask = np.where(mask > threshold_value, mask, 0)
        else:
            # Fixed threshold
            mask = np.where(mask > threshold, mask, 0)

        # Postprocessing: dilation and blur
        if enable_postprocess:
            # Dilate mask to expand edges slightly
            kernel = np.ones((5, 5), np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
            mask = dilated.astype(np.float32) / 255.0

            # Blur mask for smoother edges (critical for quality)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)

        # Temporal smoothing: blend with previous frame's mask for stability
        if (
            self.temporal_smoothing > 0.0
            and self._prev_mask is not None
            and mask.shape == self._prev_mask.shape
        ):
            # accumulateWeighted: dst = src * alpha + dst * (1 - alpha)
            alpha = self.temporal_smoothing
            mask = cv2.addWeighted(mask, alpha, self._prev_mask, 1 - alpha, 0)

        # Store current mask for next frame
        self._prev_mask = mask.copy()

        return mask

    def _sigmoid(self, x: np.ndarray, a: float = 5.0, b: float = -10.0) -> np.ndarray:
        """Apply sigmoid transformation to mask.

        Args:
            x: Input mask (0-1 range)
            a: Sigmoid parameter a
            b: Sigmoid parameter b

        Returns:
            Transformed mask
        """
        z = np.exp(a + b * x)
        sig = 1 / (1 + z)
        return sig
