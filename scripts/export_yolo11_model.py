#!/usr/bin/env python3
"""Re-export YOLO11n-seg model with ONNX opset 21 for compatibility with onnxruntime-gpu 1.20.1"""

from ultralytics import YOLO

# Load the model
model = YOLO("yolo11n-seg.pt")

# Export with opset 21 (compatible with onnxruntime-gpu 1.20.1)
model.export(format="onnx", opset=21, simplify=True)

print("\n✓ Model exported successfully with opset 21")
print("Output: yolo11n-seg.onnx")
print("\nMove it to cache directory:")
print("  mv yolo11n-seg.onnx ~/.cache/chameleon/models/yolo11n-seg.onnx")
