import onnx
from onnx import TensorProto, helper

model = onnx.load("vggt.onnx")
for node in model.graph.node:
    if node.op_type == "Einsum":
        for idx, inp in enumerate(node.input):
            # insert Cast -> float before inp if its dtype is DOUBLE
            cast = helper.make_node(
                "Cast", inputs=[inp], outputs=[inp + "_f32"], to=TensorProto.FLOAT
            )
            node.input[idx] = inp + "_f32"
            model.graph.node.insert(0, cast)
onnx.save(model, "vggt_all_32_cast.onnx")
