"""Real webcam input abstraction.

Implementation Choice: OpenCV VideoCapture
==========================================

This module uses OpenCV's VideoCapture with V4L2 backend for webcam access.
While alternatives like PyAV (FFmpeg), direct V4L2, or GStreamer exist, OpenCV
is the optimal choice for this use case:

Performance Considerations:
- Camera capture: ~20-30ms per frame
- MediaPipe segmentation: ~80-150ms per frame (the actual bottleneck)
- Switching to PyAV would save ~5-10ms on MJPEG decoding
- Direct V4L2 would save ~10ms for raw formats, but YUYV→BGR conversion
  negates most gains (~8ms cost)
- Net improvement: <1% of total pipeline time

Advantages of OpenCV:
- Simple, reliable API for frame-by-frame access
- Excellent format negotiation and codec support
- Native BGR output (required by MediaPipe/downstream processing)
- Well-tested V4L2 integration
- Minimal code, easy to maintain
- Already a core dependency

Alternative backends should only be considered if profiling demonstrates that
camera capture (not segmentation) is the performance bottleneck, which is
highly unlikely given MediaPipe's computational cost.
"""

import cv2
import numpy as np


class RealCamera:
    """Real webcam input using OpenCV."""

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        codec: str = "MJPG",
    ):
        """Initialize real camera.

        Args:
            device: Camera device path
            width: Frame width
            height: Frame height
            fps: Frame rate
            codec: Video codec (fourcc code)
        """
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.cap: cv2.VideoCapture | None = None
        self._open_camera()

    def _open_camera(self):
        """Open the camera device."""
        # Extract device number from path (e.g., /dev/video0 -> 0)
        if self.device.startswith("/dev/video"):
            device_num = int(self.device.replace("/dev/video", ""))
        else:
            device_num = 0

        # Open camera with V4L2 backend
        self.cap = cv2.VideoCapture(device_num, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.device}")

        # Set codec
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._set_property(cv2.CAP_PROP_FOURCC, fourcc, "codec")

        # Set resolution
        self._set_property(cv2.CAP_PROP_FRAME_WIDTH, self.width, "width")
        self._set_property(cv2.CAP_PROP_FRAME_HEIGHT, self.height, "height")

        # Set FPS
        self._set_property(cv2.CAP_PROP_FPS, self.fps, "fps")

    def _set_property(self, prop: int, value, name: str):
        """Set camera property with error handling.

        Args:
            prop: OpenCV property ID
            value: Property value
            name: Property name (for logging)
        """
        if not self.cap.set(prop, value):
            print(
                f"Warning: Cannot set camera property {name} to {value}. "
                f"Defaulting to auto-detected property set by OpenCV"
            )

    def read_frame(self) -> np.ndarray | None:
        """Read a frame from the camera.

        Returns:
            Frame as numpy array (height, width, 3) in BGR format, or None if failed
        """
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def close(self):
        """Close the camera."""
        if self.cap:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
