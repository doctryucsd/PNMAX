import onnx
from onnx import helper, shape_inference, TensorProto
from resnet import ResNet18

def onnx_decode(graph):
    
    values = {}
    for value_info in graph.value_info:
        values[value_info.name] = value_info

    weights = {}
    for weight_info in graph.initializer:
        weights[weight_info.name] = weight_info.dims

    node = graph.node

    for node in graph.node:

        print("\n")

        if node.op_type in {"Conv","AveragePool","MaxPool"}:
            
            # print("attribute:\n")
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = attr.ints
                elif attr.name == "strides":
                    strides = attr.ints
                elif attr.name == "pads":
                    pads = attr.ints
                # elif attr.name == "dilations":
                #     dilations = attr.ints
            print(f"[{node.op_type}], kernel_shape: {kernel_shape}, strides: {strides}, pads: {pads}")
        
        elif node.op_type == "Gemm":
            print(f"[Gemm] alpha: {node.attribute[0].f}, beta: {node.attribute[1].f}, transB: {node.attribute[2].f}")
        
        else: 
            print(f"[{node.op_type}]")
            # print(node.attribute)

        # NOTE: in Add, 2 inputs

        for _, input in enumerate(node.input):
            
            if input in values.keys():
                input_info = values[input]
            elif input == graph.input[0].name:
                input_info = graph.input[0]
            else:
                weight_info = weights[input]
                print('Weight', input, weight_info)
                continue
            input_shape = [input_info.type.tensor_type.shape.dim[i].dim_value for i in range(len(input_info.type.tensor_type.shape.dim))]
            print('Input',input, input_shape)
        
        # NOTE: output only have 1 entry
        output = node.output[0]
        # print("\noutput:")
        if output in values.keys():
            output_info = values[output]
        elif output == graph.output[0].name:
            output_info = graph.output[0]
        else:
            assert False, f"miss output"
        output_shape = [output_info.type.tensor_type.shape.dim[i].dim_value for i in range(len(output_info.type.tensor_type.shape.dim))]
        print('Output',output, output_shape)


def test():
    # 前端转化为Onnx
    model = ResNet18() # Resnet18
    # llama2
    
    # torch.onnx.export(model, torch.randn(1, 3, 32, 32), 'resnet18.onnx')
    model = onnx.load('./dump/resnet18.onnx')
    # shape inference
    inferred_model = shape_inference.infer_shapes(model)
    # print(model)
    graph = inferred_model.graph