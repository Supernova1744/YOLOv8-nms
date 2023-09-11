import onnx

def load_model(path):
    onnx_model = onnx.load_model(path)
    print(f"{path} Loaded")
    onnx.checker.check_model(onnx_model)
    return onnx_model