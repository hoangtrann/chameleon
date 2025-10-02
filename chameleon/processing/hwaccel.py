"""Hardware acceleration detection and management."""

import os
import subprocess

from chameleon.config import HWAccelMethod


class HWAccelManager:
    """Manage hardware acceleration methods."""

    def __init__(self):
        """Initialize hardware acceleration manager."""
        self._available_methods = self._detect_available()
        self._current_method: HWAccelMethod | None = None

    def _detect_available(self) -> list[HWAccelMethod]:
        """Detect available hardware acceleration methods.

        Returns:
            List of available hardware acceleration methods
        """
        available = [HWAccelMethod.NONE]

        # Test CUDA (NVIDIA)
        if self._test_nvidia_gpu():
            available.append(HWAccelMethod.CUDA)
            if self._test_vdpau():
                available.append(HWAccelMethod.VDPAU)

        # Test VAAPI (Intel/AMD)
        if self._test_vaapi():
            available.append(HWAccelMethod.VAAPI)

        return available

    def _test_nvidia_gpu(self) -> bool:
        """Check if NVIDIA GPU is available.

        Returns:
            True if NVIDIA GPU is detected
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                timeout=2,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _test_vdpau(self) -> bool:
        """Test VDPAU availability.

        Returns:
            True if VDPAU is available
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return "vdpau" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _test_vaapi(self) -> bool:
        """Test VAAPI availability.

        Returns:
            True if VAAPI is available
        """
        # Check for render nodes
        return os.path.exists("/dev/dri/renderD128")

    def _test_cuda(self) -> bool:
        """Test CUDA availability in FFmpeg.

        Returns:
            True if CUDA is available
        """
        try:
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            return "cuda" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def select_best(self, preference: HWAccelMethod = HWAccelMethod.AUTO) -> HWAccelMethod:
        """Select best available hardware acceleration method.

        Args:
            preference: Preferred hardware acceleration method

        Returns:
            Selected hardware acceleration method
        """
        if preference != HWAccelMethod.AUTO and preference in self._available_methods:
            self._current_method = preference
            return preference

        # Priority: CUDA > VAAPI > VDPAU > NONE
        for method in [
            HWAccelMethod.CUDA,
            HWAccelMethod.VAAPI,
            HWAccelMethod.VDPAU,
        ]:
            if method in self._available_methods:
                self._current_method = method
                return method

        self._current_method = HWAccelMethod.NONE
        return HWAccelMethod.NONE

    def get_onnx_providers(self) -> list:
        """Get ONNX Runtime execution providers for current hardware acceleration.

        Returns:
            List of execution providers with configuration
        """
        from pathlib import Path

        providers = []

        if self._current_method == HWAccelMethod.CUDA:
            # Try TensorRT first (1.5-2x faster than CUDA)
            if self._check_tensorrt_available():
                cache_path = str(Path.home() / ".cache" / "chameleon" / "tensorrt")
                Path(cache_path).mkdir(parents=True, exist_ok=True)
                providers.append(
                    (
                        "TensorRTExecutionProvider",
                        {
                            "trt_fp16_enable": True,
                            "trt_engine_cache_enable": True,
                            "trt_engine_cache_path": cache_path,
                        },
                    )
                )

            # CUDA fallback
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "cudnn_conv_algo_search": "HEURISTIC",
                    },
                )
            )

        # CPU fallback (always add)
        providers.append("CPUExecutionProvider")

        return providers

    def _check_tensorrt_available(self) -> bool:
        """Check if TensorRT execution provider is available."""
        try:
            import onnxruntime as ort

            return "TensorRTExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False

    @property
    def available_methods(self) -> list[HWAccelMethod]:
        """Get list of available hardware acceleration methods.

        Returns:
            List of available methods
        """
        return self._available_methods

    @property
    def current_method(self) -> HWAccelMethod | None:
        """Get currently selected hardware acceleration method.

        Returns:
            Current method or None if not selected
        """
        return self._current_method

    def has_cuda(self) -> bool:
        """Check if CUDA is available.

        Returns:
            True if CUDA is available
        """
        return HWAccelMethod.CUDA in self._available_methods

    def has_vaapi(self) -> bool:
        """Check if VAAPI is available.

        Returns:
            True if VAAPI is available
        """
        return HWAccelMethod.VAAPI in self._available_methods

    def has_vdpau(self) -> bool:
        """Check if VDPAU is available.

        Returns:
            True if VDPAU is available
        """
        return HWAccelMethod.VDPAU in self._available_methods


def check_opencv_cuda() -> bool:
    """Check if OpenCV has CUDA support enabled.

    Returns:
        True if OpenCV CUDA is available
    """
    try:
        import cv2

        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except (ImportError, AttributeError):
        return False
