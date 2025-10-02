"""Image composition - blend layers of background, person, and foreground."""

import cv2
import numpy as np

from chameleon.config import FilterConfig


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
                # Resize to output dimensions
                self.background_image = cv2.resize(img, (self.width, self.height))
                return True

            # Try to load as video
            cap = cv2.VideoCapture(image_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.background_image = cv2.resize(frame, (self.width, self.height))
                    cap.release()
                    return True
                cap.release()

            return False
        except Exception as e:
            print(f"Error loading background: {e}")
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

            self.foreground_image = cv2.resize(img, (self.width, self.height))

            # Load mask if provided
            if mask_path:
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
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
            print(f"Error loading foreground: {e}")
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
        if background is None:
            if self.background_image is None:
                # No background, create black background
                background = np.zeros_like(person_frame)
            else:
                background = self.background_image.copy()
        else:
            # Ensure background is correct size
            if background.shape[:2] != (self.height, self.width):
                background = cv2.resize(background, (self.width, self.height))

        # Ensure mask is 2D and correct size
        if len(person_mask.shape) == 3:
            person_mask = person_mask[:, :, 0]
        if person_mask.shape != (self.height, self.width):
            person_mask = cv2.resize(person_mask, (self.width, self.height))

        # Expand mask to 3 channels for blending
        mask_3ch = np.stack([person_mask] * 3, axis=2)

        # Composite person over background
        # result = person * mask + background * (1 - mask)
        composited = person_frame.astype(np.float32) * mask_3ch + background.astype(np.float32) * (
            1 - mask_3ch
        )

        # Apply foreground overlay if available
        if self.foreground_image is not None and self.foreground_mask is not None:
            composited = self._apply_foreground(composited)

        # Clip and convert back to uint8
        return np.clip(composited, 0, 255).astype(np.uint8)

    def _apply_foreground(self, frame: np.ndarray) -> np.ndarray:
        """Apply foreground overlay to frame.

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

        # Expand mask to 3 channels
        mask_3ch = np.stack([self.foreground_mask * opacity] * 3, axis=2)

        # Blend foreground over frame
        # result = foreground * mask + frame * (1 - mask)
        blended = self.foreground_image.astype(np.float32) * mask_3ch + frame.astype(np.float32) * (
            1 - mask_3ch
        )

        return blended

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
