import onnxruntime as ort

sess = ort.InferenceSession("anpr_yolov5m.onnx")

print("INPUTS:")
for i in sess.get_inputs():
    print(i.name, i.shape, i.type)

print("\nOUTPUTS:")
for o in sess.get_outputs():
    print(o.name, o.shape, o.type)
