import onnx
import torch
import onnxsim
import onnxruntime as ort
import numpy as np

from onnx.tools import update_model_dims
from onnx.compose import merge_models
from onnx.version_converter import convert_version

from utils import load_model
from transformation import Transform

def convert():
    onnx_model = load_model("best.onnx")
    graph = onnx_model.graph

    score_threshold = onnx.helper.make_tensor("score_threshold", onnx.TensorProto.FLOAT, [], [0.25])
    iou_threshold = onnx.helper.make_tensor("iou_threshold", onnx.TensorProto.FLOAT, [], [0.45])
    max_output_boxes_per_class = onnx.helper.make_tensor("max_output_boxes_per_class", onnx.TensorProto.INT64, [], [300])

    graph.initializer.append(score_threshold)
    graph.initializer.append(iou_threshold)
    graph.initializer.append(max_output_boxes_per_class)

    Mul = "/model.22/Mul_5_output_0" #TODO: Change the value
    Sig = "/model.22/Sigmoid_output_0" #TODO: Change the value
    Con = "/model.22/Concat_25" #TODO: Change the value

    transpose_bboxes_node = onnx.helper.make_node("Transpose", inputs=[Mul], outputs=["bboxes"], perm=(0, 2, 1))
    graph.node.append(transpose_bboxes_node)

    inputs = ['bboxes', Sig, 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold']
    outputs = ["selected_indices"]
    nms_node = onnx.helper.make_node('NonMaxSuppression', inputs, outputs, center_point_box=1)
    graph.node.append(nms_node)

    output_value_info = onnx.helper.make_tensor_value_info("selected_indices", onnx.TensorProto.INT64, shape=[None, 3])
    graph.output.append(output_value_info)

    nc = 80 #TODO: Update with the Number of classes
    
    last_concat_node = [node for node in onnx_model.graph.node if node.name == Con][0]
    onnx_model.graph.node.remove(last_concat_node)

    output0 = [o for o in onnx_model.graph.output if o.name == "output0"][0]
    onnx_model.graph.output.remove(output0)

    mul_node = onnx.helper.make_tensor_value_info(Mul, onnx.TensorProto.FLOAT, shape=["batch", 4, 8400])
    sig_node = onnx.helper.make_tensor_value_info(Sig, onnx.TensorProto.FLOAT, shape=["batch", nc, 8400])

    graph.output.append(mul_node)
    graph.output.append(sig_node)

    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, "best_nms.onnx")

    # Use the nms model to simulate the input of the module
    session = ort.InferenceSession("best_nms.onnx")
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    image = np.random.rand(8, 3, 640, 640).astype(np.float32)
    output = session.run(outname, {inname[0]: image})

    torch.onnx.export(Transform(), (
                            torch.tensor(output[0], dtype=torch.int64),
                            torch.Tensor(output[1]),
                            torch.Tensor(output[2])
                            ),
                        "./NMS_after.onnx", 
                        input_names=outname, 
                        output_names=["det_bboxes", "batches"], 
                        dynamic_axes={
                            "det_bboxes": {0: "num_results"},
                            "batches": {0: "num_results"},
                        })

    nms_postprocess_onnx_model = onnx.load_model("./NMS_after.onnx")
    nms_postprocess_onnx_model_sim, check = onnxsim.simplify(nms_postprocess_onnx_model)
    onnx.save(nms_postprocess_onnx_model_sim, "./NMS_after_sim.onnx")

    input_dims = {
        "images": ["batch", 3, 640, 640],
    }

    output_dims = {
        "selected_indices": ["batch", 3],
        Mul: ["batch", "boxes", "num_anchors"],
        Sig: ["batch", "classes", "num_anchors"],
    }

    target_ir_version = 18
    updated_onnx_model = update_model_dims.update_inputs_outputs_dims(
                                                    onnx_model, 
                                                    input_dims,
                                                    output_dims)
    core_model = convert_version(updated_onnx_model, target_ir_version)
    onnx.checker.check_model(core_model)

    core_model.ir_version = 8
    post_process_model = convert_version(nms_postprocess_onnx_model_sim, target_ir_version)
    onnx.checker.check_model(post_process_model)
    post_process_model.ir_version = 8

    combined_onnx_model = merge_models(core_model, post_process_model, io_map=[
        (Mul, Mul),
        (Sig, Sig),
        ('selected_indices', 'selected_indices')
    ])
    onnx.save(combined_onnx_model, './final_model.onnx')

if __name__ == "__main__":
    convert()