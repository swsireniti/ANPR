from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    opset=11,
    simplify=True,
    dynamic=False,
    imgsz=640
)