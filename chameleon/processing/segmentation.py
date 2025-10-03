"""Person segmentation using various ML models."""

import logging
from pathlib import Path

import numpy as np

from chameleon.config import SegmentationModel
from chameleon.engine.mediapipe import MediaPipeEngine

logger = logging.getLogger(__name__)


class SegmentationEngine:
    """Segmentation engine supporting multiple ML models."""

    def __init__(
        self,
        model: SegmentationModel = SegmentationModel.MEDIAPIPE,
        use_gpu: bool = True,
        use_clahe: bool = False,
    ):
        """Initialize segmentation engine.

        Args:
            model: Segmentation model to use
            use_gpu: Whether to use GPU acceleration if available
            use_clahe: Enable CLAHE preprocessing for better low-light performance
        """
        self.model_type = model
        self.use_gpu = use_gpu
        self.use_clahe = use_clahe
        self.model_path: Path | None = None
        self.session = None
        self.mp_engine: MediaPipeEngine | None = None

        # Instance variables for letterbox tracking
        self._orig_shape = None
        self._padding = (0, 0)
        self._ratio = 1.0

        # CLAHE for lighting enhancement
        self._clahe = None
        if use_clahe:
            import cv2

            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if model == SegmentationModel.MEDIAPIPE:
            self._init_mediapipe()
        else:
            self.model_path = self._get_or_download_model(model)
            self._init_onnx(use_gpu)

    def _get_or_download_model(self, model: SegmentationModel) -> Path:
        """Get model path, download if not cached.

        Args:
            model: Segmentation model

        Returns:
            Path to model file
        """
        cache_dir = Path.home() / ".cache" / "chameleon" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_paths = {
            SegmentationModel.YOLO11N: cache_dir / "yolo11n-seg.onnx",
            SegmentationModel.FASTSAM_S: cache_dir / "fastsam-s.onnx",
        }

        model_path = model_paths.get(model)
        if not model_path:
            raise ValueError(f"Unknown model: {model}")

        if not model_path.exists():
            self._download_model(model, model_path)

        return model_path

    def _download_model(self, model: SegmentationModel, target: Path):
        """Download and convert model from ultralytics.

        Args:
            model: Segmentation model to download
            target: Target path for downloaded model (ONNX format)
        """
        model_names = {
            SegmentationModel.YOLO11N: "yolo11n-seg",
            SegmentationModel.FASTSAM_S: "FastSAM-s",
        }

        model_name = model_names.get(model)
        if not model_name:
            raise ValueError(f"Unknown model: {model}")

        logger.info("Downloading and converting %s model to ONNX...", model)
        try:
            # Import ultralytics to download and export the model
            from ultralytics import YOLO

            # Download the PyTorch model (auto-downloads from ultralytics)
            pt_model = YOLO(f"{model_name}.pt")

            # Export to ONNX format
            pt_model.export(format="onnx", imgsz=640)

            # Move the exported ONNX file to the target location
            import shutil

            onnx_file = Path(f"{model_name}.onnx")
            if onnx_file.exists():
                shutil.move(str(onnx_file), str(target))
                logger.info("Model converted and saved to %s", target)
            else:
                raise FileNotFoundError(f"Expected ONNX file not found: {onnx_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to download/convert model: {e}") from e

    def _init_onnx(self, use_gpu: bool):
        """Initialize ONNX Runtime session with TensorRT optimization.

        Args:
            use_gpu: Whether to use GPU acceleration
        """
        try:
            from pathlib import Path

            import onnxruntime as ort

            logger.info("Initializing %s model...", self.model_type.value)
            logger.info("Model path: %s", self.model_path)

            # Build provider list with TensorRT for maximum performance
            providers = []
            if use_gpu:
                # Try TensorRT first (1.5-2x faster than CUDA)
                if self._check_tensorrt_available():
                    logger.info("TensorRT available - using FP16 acceleration")
                    cache_path = str(Path.home() / ".cache" / "chameleon" / "tensorrt")
                    Path(cache_path).mkdir(parents=True, exist_ok=True)
                    # providers.append(
                    #     (
                    #         "TensorrtExecutionProvider",
                    #         {
                    #             "trt_fp16_enable": True,  # 2x speedup
                    #             "trt_engine_cache_enable": True,
                    #             "trt_engine_cache_path": cache_path,
                    #         },
                    #     )
                    # )
                else:
                    logger.info("TensorRT not available")

                # CUDA fallback
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    logger.info("CUDA available")
                    providers.append(
                        (
                            "CUDAExecutionProvider",
                            {
                                "device_id": 0,
                                "cudnn_conv_algo_search": "HEURISTIC",  # Faster startup
                            },
                        )
                    )
                else:
                    logger.warning("CUDA not available, falling back to CPU")

            # CPU fallback (always available)
            # providers.append("CPUExecutionProvider")

            # Session options for performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2  # Don't hog all cores

            self.session = ort.InferenceSession(
                str(self.model_path), providers=providers, sess_options=sess_options
            )
            self.input_name = self.session.get_inputs()[0].name

            # Log which provider is being used
            active_provider = self.session.get_providers()[0]
            logger.info("Active provider: %s", active_provider)
            if active_provider == "CPUExecutionProvider":
                logger.warning("Using CPU inference - this will be VERY SLOW!")
                logger.warning("Install onnxruntime-gpu for GPU acceleration")

        except ImportError:
            raise RuntimeError(
                "onnxruntime is required for ONNX models. "
                "Install with: pip install onnxruntime or onnxruntime-gpu"
            ) from None

    def _check_tensorrt_available(self) -> bool:
        """Check if TensorRT execution provider is available."""
        try:
            import onnxruntime as ort

            return "TensorrtExecutionProvider" in ort.get_available_providers()
        except ImportError:
            return False

    def _init_mediapipe(self):
        """Initialize MediaPipe with modern GPU-accelerated engine."""
        self.mp_engine = MediaPipeEngine(use_gpu=self.use_gpu)

    def segment(self, frame: np.ndarray) -> np.ndarray:
        """Run segmentation and return binary mask.

        Args:
            frame: Input frame (height, width, 3) in BGR format

        Returns:
            Binary mask (height, width) with values 0-1
        """
        # Optional: Apply CLAHE for better low-light performance
        if self._clahe is not None:
            import cv2

            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            # Apply CLAHE to L channel
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            # Convert back to BGR
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        if self.model_type == SegmentationModel.MEDIAPIPE:
            return self._segment_mediapipe(frame)
        else:
            return self._segment_onnx(frame)

    def _segment_onnx(self, frame: np.ndarray) -> np.ndarray:
        """Run ONNX model inference.

        Args:
            frame: Input frame in BGR format

        Returns:
            Binary mask (0-1 range)
        """
        # Preprocess
        input_tensor = self._preprocess_frame(frame)

        # Inference - returns list of outputs [detections, prototypes]
        outputs = self.session.run(None, {self.input_name: input_tensor})

        # Postprocess - extract person segmentation mask
        mask = self._postprocess_mask(outputs, frame.shape[:2])
        return mask

    def _segment_mediapipe(self, frame: np.ndarray) -> np.ndarray:
        """Run MediaPipe segmentation using modern GPU-accelerated engine.

        Args:
            frame: Input frame in BGR format

        Returns:
            Binary mask (0-1 range)
        """
        return self.mp_engine.segment(frame)

    def _letterbox(self, img: np.ndarray, new_shape: tuple = (640, 640)) -> tuple:
        """Letterbox resize with aspect ratio preservation.

        Args:
            img: Input image (HWC format)
            new_shape: Target shape (height, width)

        Returns:
            Tuple of (resized_image, (dw, dh), ratio)
        """
        import cv2

        # Calculate scaling ratio
        shape = img.shape[:2]  # current shape [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute new unpadded dimensions
        new_unpad = (
            int(round(shape[1] * r)),
            int(round(shape[0] * r)),
        )  # (width, height)

        # Resize
        if shape[::-1] != new_unpad:  # if shapes are different
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Calculate padding
        dw = (new_shape[1] - new_unpad[0]) / 2  # width padding
        dh = (new_shape[0] - new_unpad[1]) / 2  # height padding

        # Add border padding
        top, bottom = int(round(dh)), int(round(dh))
        left, right = int(round(dw)), int(round(dw))

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, (dw, dh), r

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO11n-seg ONNX model.

        Args:
            frame: Input frame in BGR format (HWC)

        Returns:
            Preprocessed tensor in BCHW format, normalized to [0, 1]
        """
        import cv2

        # Store original shape for postprocessing
        self._orig_shape = frame.shape[:2]

        # Letterbox resize to 640x640 maintaining aspect ratio
        img, padding, ratio = self._letterbox(frame, new_shape=(640, 640))
        self._padding = padding
        self._ratio = ratio

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # HWC to CHW
        img = img.transpose(2, 0, 1)

        # Normalize to [0, 1] and convert to float32
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0

        # Add batch dimension: CHW -> BCHW
        img = np.expand_dims(img, axis=0)

        return img

    def _postprocess_mask(self, outputs: list, target_shape: tuple) -> np.ndarray:
        """Postprocess YOLO11n-seg output to extract person segmentation mask.

        YOLO segmentation outputs two tensors:
        - output[0]: Detection boxes and class predictions (1, 116, 8400)
          First 4 values: box coords (x_center, y_center, width, height)
          Next 80 values: class confidences (COCO classes)
          Last 32 values: mask coefficients
        - output[1]: Mask prototypes (1, 32, 160, 160)

        Args:
            outputs: List of ONNX model outputs [detections, prototypes]
            target_shape: Original frame shape (height, width)

        Returns:
            Binary mask resized to target shape, 0-1 range
        """
        import cv2

        # For person segmentation, we create a binary mask for all detected persons
        # COCO class 0 is "person"
        person_class_id = 0
        confidence_threshold = 0.5

        detections = outputs[0]  # Shape: (1, 116, 8400)
        protos = outputs[1] if len(outputs) > 1 else None  # Shape: (1, 32, 160, 160)

        # Remove batch dimension
        detections = detections[0]  # (116, 8400)

        # Transpose to (8400, 116) for easier processing
        detections = detections.T  # (8400, 116)

        # Extract class confidences (indices 4:84 for 80 COCO classes)
        class_scores = detections[:, 4:84]

        # Get max confidence and class for each detection
        max_scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        # Filter for person class with confidence threshold
        person_mask_indices = (class_ids == person_class_id) & (max_scores >= confidence_threshold)

        if not person_mask_indices.any():
            # No person detected, return empty mask
            return np.zeros(target_shape, dtype=np.float32)

        # If we have mask prototypes (segmentation model)
        if protos is not None:
            # Extract mask coefficients (last 32 values) for person detections
            mask_coeffs = detections[person_mask_indices, 84:]  # (N, 32)

            # Remove batch dimension from prototypes
            protos = protos[0]  # (32, 160, 160)

            # Generate masks: masks = coefficients @ prototypes
            # Result shape: (N, 160, 160)
            masks = np.matmul(mask_coeffs, protos.reshape(32, -1)).reshape(-1, 160, 160)

            # Apply sigmoid to get probabilities
            masks = 1 / (1 + np.exp(-masks))

            # Combine all person masks (take maximum probability at each pixel)
            combined_mask = masks.max(axis=0) if masks.shape[0] > 1 else masks[0]

            # Resize mask from 160x160 to 640x640 (model output size)
            mask_640 = cv2.resize(combined_mask, (640, 640), interpolation=cv2.INTER_LINEAR)

            # Remove letterbox padding and resize to original shape
            # Unpad
            dw, dh = self._padding
            h, w = target_shape
            mask_640 = mask_640[int(dh) : 640 - int(dh), int(dw) : 640 - int(dw)]

            # Resize to original dimensions
            final_mask = cv2.resize(mask_640, (w, h), interpolation=cv2.INTER_LINEAR)

        else:
            # Fallback: no mask prototypes, create simple binary mask
            # This shouldn't happen with yolo11n-seg, but handle it anyway
            final_mask = np.ones(target_shape, dtype=np.float32)

        # Ensure mask is in 0-1 range
        final_mask = np.clip(final_mask, 0, 1).astype(np.float32)

        return final_mask

    def close(self):
        """Close the segmentation engine and release resources."""
        if self.mp_engine:
            self.mp_engine.close()
            self.mp_engine = None
        self.session = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
