"""Virtual camera output using v4l2loopback."""

import os

import numpy as np
import pyfakewebcam


class VirtualCameraOutput:
    """Virtual camera output via v4l2loopback.

    Uses pyfakewebcam which provides optimal performance when OpenCV is installed
    (~3ms/frame vs ~26ms/frame without OpenCV for RGB→YUV420p conversion).
    """

    def __init__(self, device: str, width: int, height: int):
        """Initialize virtual camera output.

        Args:
            device: Virtual camera device path (e.g., /dev/video10)
            width: Frame width
            height: Frame height

        Raises:
            FileNotFoundError: If device does not exist
            RuntimeError: If device initialization fails
        """
        self.device = device
        self.width = width
        self.height = height

        # Validate device exists
        if not os.path.exists(device):
            raise FileNotFoundError(
                f"Virtual camera device not found: {device}\n"
                f"Make sure v4l2loopback is loaded: sudo modprobe v4l2loopback"
            )

        try:
            self.camera = pyfakewebcam.FakeWebcam(device, width, height)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize virtual camera {device}: {e}") from e

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the virtual camera.

        Args:
            frame: Frame to write (height, width, 3) in BGR format
        """
        try:
            # pyfakewebcam expects RGB, so convert from BGR using efficient slice
            # This is faster than cv2.cvtColor and avoids extra memory allocation
            frame_rgb = frame[:, :, ::-1]
            self.camera.schedule_frame(frame_rgb)
        except Exception as e:
            # Don't raise to avoid crashing the pipeline on transient errors
            print(f"Warning: Failed to write frame to {self.device}: {e}")

    def close(self):
        """Close the virtual camera."""
        # pyfakewebcam doesn't have an explicit close method
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
