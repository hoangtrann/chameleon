#!/usr/bin/env python3
"""Re-export YOLO11n-seg model with ONNX opset 21 for compatibility with onnxruntime-gpu 1.20.1"""

import logging

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Load the model
model = YOLO("yolo11n-seg.pt")

# Export with opset 21 (compatible with onnxruntime-gpu 1.20.1)
model.export(format="onnx", opset=21, simplify=True)

logger.info("\n✓ Model exported successfully with opset 21")
logger.info("Output: yolo11n-seg.onnx")
logger.info("\nMove it to cache directory:")
logger.info("  mv yolo11n-seg.onnx ~/.cache/chameleon/models/yolo11n-seg.onnx")
