#!/usr/bin/env python3
# type: ignore
# Usage: python __ai_onnx_.py

import os, json, warnings, math
import onnxruntime as ort
import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
from types import SimpleNamespace

warnings.filterwarnings("ignore")

# Try to import onnx for model analysis
try:
    import onnx
    from onnx import helper, numpy_helper
    HAVE_ONNX = True
except ImportError:
    onnx = None
    helper = None
    numpy_helper = None
    HAVE_ONNX = False
    print("Warning: onnx not available. Advanced model analysis will be limited.")

# Constants
DTYPE_SIZES = {
    'float32': 4,
    'float16': 2,
    'int8': 1,
    'int32': 4,
    'int64': 8,
    'bool': 1,
}

KERNEL_CLASSES = {
    1: "Dense GEMM / Standard Conv",
    2: "Depthwise / Grouped Conv", 
    3: "Attention Core (QK^T, softmax, AV)",
    4: "Elementwise / Pointwise",
    5: "Reductions / Pooling / Norms",
    6: "Embedding / Gather / Scatter",
    7: "Data Movement / Layout"
}

# Known FLOP values for validation (batch=1; InceptionV3 uses 299x299)
KNOWN_FLOPS = {
    'resnet18': 3.6,
    'resnet50': 8.2,
    'resnet50_v1': 8.2, 
    'densenet121': 5.8,
    'efficientnet_b0': 0.78,
    'mobilenet_v2': 0.60,
    'squeezenet1_1': 0.70,
    'inception_v3': 11.5,
    'mobilenet_v3_small': 0.12,
    'convnext_tiny': 0.90,
    'mnasnet1': 0.63,
}

# Models that should have significant depthwise convolution FLOPs
DEPTHWISE_MODELS = [
    'mobilenet_v2', 'mobilenet_v3', 'efficientnet', 'convnext', 'mnasnet'
]

def elem_size_bytes(dtype_str: str) -> int:
    return DTYPE_SIZES.get(dtype_str, 4)

def numel(shape: List[int]) -> int:
    if shape is None or len(shape) == 0:
        return 0
    result = 1
    for dim in shape:
        if dim is not None and dim > 0:
            result *= dim
    return result

def tensor_bytes(shape: List[int], dtype_str: str) -> int:
    if shape is None or dtype_str is None:
        return 0
    return numel(shape) * elem_size_bytes(dtype_str)

def get_input_size_from_shape(shape: List[int]) -> int:
    """Extract input size from NCHW format"""
    if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return max(shape[2], shape[3])
    return 224

def expected_input_size(model_name: str) -> int:
    return 299 if "inception_v3" in model_name.lower() else 224

class TensorMeta:
    def __init__(self, shape: List[int], dtype: str):
        self.shape = shape
        self.dtype = dtype

class KernelClassifier:
    def __init__(self):
        self.attention_tagged_node_names = set()
        
    def is_depthwise_conv(self, node, input_shapes, weight_shape, groups) -> bool:
        """Check if this is a depthwise convolution using actual weight shape"""
        if node.op_type != 'Conv' or not weight_shape or len(weight_shape) < 2:
            return False
        cout, cin_pg = weight_shape[0], weight_shape[1]
        cin = cin_pg * max(groups, 1)
        # depthwise hallmark: Cin_per_group == 1 and groups == Cin (Cout == Cin or multiple)
        return (groups > 1) and (cin_pg == 1) and (cout % cin == 0)
        
    def is_attention_core(self, node) -> bool:
        node_name = node.name if node.name else f"node_{id(node)}"
        return node_name in self.attention_tagged_node_names
        
    def classify_node(self, node, input_shapes, weight_shape=None, groups=1) -> int:
        op_type = node.op_type
        
        if op_type in ['Conv']:
            if self.is_depthwise_conv(node, input_shapes, weight_shape, groups):
                return 2  # Depthwise/Grouped Conv
            return 1  # Dense GEMM/Standard Conv
            
        elif op_type in ['MatMul', 'Gemm']:
            if self.is_attention_core(node):
                return 3  # Attention Core
            return 1  # Dense GEMM
            
        elif op_type in ['Relu', 'Sigmoid', 'Tanh', 'Add', 'Mul', 'Sub', 'Div', 'Clip', 'LeakyRelu', 'Elu', 'Selu', 'ThresholdedRelu', 'HardSigmoid', 'Softplus', 'Softsign', 'Gelu', 'Swish']:
            return 4  # Elementwise/Pointwise
            
        elif op_type in ['BatchNormalization', 'LayerNormalization', 'GroupNormalization', 'InstanceNormalization', 'LpNormalization', 'MaxPool', 'AveragePool', 'GlobalMaxPool', 'GlobalAveragePool', 'ReduceMean', 'ReduceSum', 'ReduceMax', 'ReduceMin', 'Softmax', 'LogSoftmax']:
            return 5  # Reductions/Pooling/Norms
            
        elif op_type in ['Gather', 'GatherElements', 'GatherND', 'Scatter', 'ScatterElements', 'ScatterND']:
            return 6  # Embedding/Gather/Scatter
            
        elif op_type in ['Reshape', 'Transpose', 'Concat', 'Split', 'Slice', 'Pad', 'Squeeze', 'Unsqueeze', 'Flatten', 'Expand', 'Tile', 'Identity']:
            return 7  # Data Movement/Layout
            
        # Default to elementwise for unknown operations
        return 4

class FLOPsCalculator:
    @staticmethod
    def conv_flops(node, input_shapes, output_shape, weight_shape, groups: int) -> int:
        """Calculate FLOPs for convolution operations using true weights and dimensions"""
        if not output_shape or not weight_shape or len(weight_shape) < 4 or len(output_shape) < 4:
            return 0
        # weight: [Cout, Cin_per_group, kH, kW] ; out: [N, Cout, Hout, Wout]
        cout, cin_pg, kh, kw = weight_shape[:4]
        _, _, hout, wout = output_shape[:4]
        # MACs = Cout * Hout * Wout * (Cin_per_group * kH * kW)
        macs = cout * hout * wout * (cin_pg * kh * kw)
        return 2 * macs  # count mul+add
        
    @staticmethod
    def matmul_flops(input_shapes, output_shape) -> int:
        """Calculate FLOPs for matrix multiplication with real dims"""
        if len(input_shapes) < 2 or not output_shape: 
            return 0
        a, b = input_shapes[0], input_shapes[1]
        if not a or not b or len(a) < 2 or len(b) < 2: 
            return 0
        M, K = a[-2], a[-1]
        K2, N = b[-2], b[-1]
        if not all(isinstance(x, int) and x > 0 for x in [M, K, K2, N]): 
            return 0
        if K != K2: 
            return 0
        # batch: broadcast max across leading dims
        ba = math.prod([d for d in a[:-2]]) if len(a) > 2 else 1
        bb = math.prod([d for d in b[:-2]]) if len(b) > 2 else 1
        batch = max(ba, bb)
        return 2 * batch * M * N * K
            
    @staticmethod
    def gemm_flops(node, input_shapes, output_shape) -> int:
        """Calculate FLOPs for GEMM operation with proper transposition handling"""
        if len(input_shapes) < 2 or not output_shape: 
            return 0
        A, B = input_shapes[0], input_shapes[1]
        if not A or not B or len(A) != 2 or len(B) != 2: 
            return 0
        transA = any(a.name == 'transA' and a.i for a in node.attribute)
        transB = any(a.name == 'transB' and a.i for a in node.attribute)
        M = A[1] if transA else A[0]
        K = A[0] if transA else A[1]
        N = B[0] if transB else B[1]
        if not all(isinstance(x, int) and x > 0 for x in [M, N, K]): 
            return 0
        return 2 * M * N * K
        
    @staticmethod
    def elementwise_flops(output_shape) -> int:
        """Calculate FLOPs for elementwise operations"""
        if output_shape is None:
            return 0
        return numel(output_shape)
        
    @staticmethod
    def pooling_flops(node, input_shapes, output_shape) -> int:
        """Calculate FLOPs for pooling operations"""
        if output_shape is None:
            return 0
            
        # Get kernel size
        kernel_size = 1
        for attr in node.attribute:
            if attr.name == 'kernel_shape':
                kernel_shape = attr.ints
                kernel_size = 1
                for k in kernel_shape:
                    kernel_size *= k
                break
                
        output_numel = numel(output_shape)
        return output_numel * kernel_size
        
    @staticmethod
    def reduction_flops(input_shapes, output_shape) -> int:
        """Calculate FLOPs for reduction operations"""
        if len(input_shapes) < 1 or output_shape is None:
            return 0
        input_numel = numel(input_shapes[0])
        return input_numel
        
    @staticmethod
    def softmax_flops(input_shapes, output_shape) -> int:
        """Calculate FLOPs for softmax operations"""
        if len(input_shapes) < 1 or output_shape is None:
            return 0
        # Softmax requires: exp(x), sum(exp(x)), div(exp(x), sum)
        # Approximate as 3 operations per element
        input_numel = numel(input_shapes[0])
        return 3 * input_numel

class BytesCalculator:
    @staticmethod
    def standard_bytes(input_shapes, input_dtypes, output_shape, output_dtype) -> int:
        """Calculate memory bytes for standard operations"""
        total_bytes = 0
        
        # Input tensors
        for shape, dtype in zip(input_shapes, input_dtypes):
            total_bytes += tensor_bytes(shape, dtype)
            
        # Output tensor
        total_bytes += tensor_bytes(output_shape, output_dtype)
        
        return total_bytes

def build_shape_type_map(model):
    """Build a single lookup map for tensor shapes and types"""
    st = {}

    def npdtype(elem_type):
        try:
            return onnx.helper.tensor_dtype_to_np_dtype(elem_type).name
        except:
            return 'float32'

    def grab(t):
        if not t or not t.type.tensor_type: 
            return None, None
        shp = []
        for d in t.type.tensor_type.shape.dim:
            if d.dim_value:
                shp.append(int(d.dim_value))
            else:
                shp.append(1)  # default for symbolic dims
        return shp, npdtype(t.type.tensor_type.elem_type)

    # inputs/outputs/value_info
    for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        s, dt = grab(vi)
        if s is not None:
            st[vi.name] = (s, dt)

    # initializers (weights)
    for init in model.graph.initializer:
        try:
            dt = onnx.helper.tensor_dtype_to_np_dtype(init.data_type).name
        except:
            dt = 'float32'
        st[init.name] = (list(init.dims), dt)

    return st

def tag_attention_patterns(model) -> set:
    """Identify attention patterns in ONNX model"""
    attention_node_names = set()
    nodes = list(model.graph.node)
    
    # Look for MatMul -> Softmax -> MatMul patterns
    for i in range(len(nodes) - 2):
        node1, node2, node3 = nodes[i], nodes[i+1], nodes[i+2]
        
        if (node1.op_type == 'MatMul' and 
            node2.op_type in ['Softmax', 'LogSoftmax'] and
            node3.op_type == 'MatMul'):
            # Use node index as identifier if name is not available
            name1 = node1.name if node1.name else f"node_{i}"
            name2 = node2.name if node2.name else f"node_{i+1}"
            name3 = node3.name if node3.name else f"node_{i+2}"
            attention_node_names.update([name1, name2, name3])
            
    # Look for more complex attention patterns with scaling
    for i in range(len(nodes) - 4):
        if (nodes[i].op_type == 'MatMul' and
            nodes[i+1].op_type in ['Div', 'Mul'] and
            nodes[i+2].op_type == 'Softmax' and
            nodes[i+3].op_type == 'MatMul'):
            names = []
            for j in range(4):
                node = nodes[i+j]
                name = node.name if node.name else f"node_{i+j}"
                names.append(name)
            attention_node_names.update(names)
            
    return attention_node_names

def get_attr(node, name, default=None):
    """Get attribute value from ONNX node"""
    for a in node.attribute:
        if a.name == name:
            if a.type == onnx.AttributeProto.INTS: 
                return list(a.ints)
            if a.type == onnx.AttributeProto.INT:  
                return a.i
    return default

def infer_output_shape(node, input_shapes: List[List[int]]) -> Optional[List[int]]:
    """Infer output shape for common operations"""
    op_type = node.op_type
    
    if op_type == 'Conv' and len(input_shapes) >= 2:
        input_shape = input_shapes[0]  # [N, C, H, W]
        weight_shape = input_shapes[1]  # [out_channels, in_channels_per_group, kH, kW]
        
        if len(input_shape) >= 4 and len(weight_shape) >= 4:
            n, c, h, w = input_shape[:4]
            out_channels, _, kh, kw = weight_shape[:4]
            
            # Get padding, stride, dilation from attributes
            pads = [0, 0, 0, 0]  # [pad_top, pad_left, pad_bottom, pad_right]
            strides = [1, 1]
            dilations = [1, 1]
            
            for attr in node.attribute:
                if attr.name == 'pads':
                    pads = list(attr.ints)
                elif attr.name == 'strides':
                    strides = list(attr.ints)
                elif attr.name == 'dilations':
                    dilations = list(attr.ints)
                    
            # Calculate output dimensions
            pad_h = pads[0] + pads[2] if len(pads) >= 4 else pads[0] * 2 if len(pads) >= 1 else 0
            pad_w = pads[1] + pads[3] if len(pads) >= 4 else pads[1] * 2 if len(pads) >= 2 else pad_h
            
            stride_h = strides[0]
            stride_w = strides[1] if len(strides) >= 2 else stride_h
            
            dilation_h = dilations[0]
            dilation_w = dilations[1] if len(dilations) >= 2 else dilation_h
            
            effective_kh = (kh - 1) * dilation_h + 1
            effective_kw = (kw - 1) * dilation_w + 1
            
            out_h = (h + pad_h - effective_kh) // stride_h + 1
            out_w = (w + pad_w - effective_kw) // stride_w + 1
            
            return [n, out_channels, out_h, out_w]
            
    elif op_type in ['MatMul', 'Gemm'] and len(input_shapes) >= 2:
        a_shape = input_shapes[0]
        b_shape = input_shapes[1]
        
        if len(a_shape) >= 2 and len(b_shape) >= 2:
            # For MatMul: [..., M, K] @ [..., K, N] -> [..., M, N]
            if op_type == 'MatMul':
                batch_dims_a = a_shape[:-2]
                batch_dims_b = b_shape[:-2]
                m = a_shape[-2]
                n = b_shape[-1]
                
                # Use broadcasting rules for batch dimensions
                max_batch_len = max(len(batch_dims_a), len(batch_dims_b))
                batch_dims = []
                for i in range(max_batch_len):
                    dim_a = batch_dims_a[-(i+1)] if i < len(batch_dims_a) else 1
                    dim_b = batch_dims_b[-(i+1)] if i < len(batch_dims_b) else 1
                    batch_dims.insert(0, max(dim_a, dim_b))
                    
                return batch_dims + [m, n]
            else:  # GEMM
                # Check for transpositions
                trans_a = False
                trans_b = False
                for attr in node.attribute:
                    if attr.name == 'transA':
                        trans_a = bool(attr.i)
                    elif attr.name == 'transB':
                        trans_b = bool(attr.i)
                        
                m = a_shape[1] if trans_a else a_shape[0]
                n = b_shape[0] if trans_b else b_shape[1]
                return [m, n]
                
    elif op_type in ['Add', 'Mul', 'Sub', 'Div'] and len(input_shapes) >= 2:
        # Broadcasting rules - return the larger shape
        shape_a = input_shapes[0]
        shape_b = input_shapes[1]
        
        # Simple broadcasting: return the shape with more elements
        if numel(shape_a) >= numel(shape_b):
            return shape_a
        else:
            return shape_b
            
    elif op_type in ['Relu', 'Sigmoid', 'Tanh', 'Softmax'] and len(input_shapes) >= 1:
        # Element-wise operations preserve input shape
        return input_shapes[0]
        
    elif op_type in ['MaxPool', 'AveragePool'] and len(input_shapes) >= 1:
        input_shape = input_shapes[0]
        if len(input_shape) >= 4:
            n, c, h, w = input_shape[:4]
            
            # Get kernel size, stride, pads
            kernel_shape = [1, 1]
            strides = [1, 1]
            pads = [0, 0, 0, 0]
            
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    kernel_shape = list(attr.ints)
                elif attr.name == 'strides':
                    strides = list(attr.ints)
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                    
            kh, kw = kernel_shape[0], kernel_shape[1] if len(kernel_shape) >= 2 else kernel_shape[0]
            stride_h, stride_w = strides[0], strides[1] if len(strides) >= 2 else strides[0]
            pad_h = pads[0] + pads[2] if len(pads) >= 4 else pads[0] * 2 if len(pads) >= 1 else 0
            pad_w = pads[1] + pads[3] if len(pads) >= 4 else pads[1] * 2 if len(pads) >= 2 else pad_h
            
            out_h = (h + pad_h - kh) // stride_h + 1
            out_w = (w + pad_w - kw) // stride_w + 1
            
            return [n, c, out_h, out_w]
            
    # For other operations, try to return first input shape as default
    if len(input_shapes) >= 1:
        return input_shapes[0]
        
    return None

def analyze_model(model_path: str, model_name: str, input_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
    """Analyze ONNX model to extract kernel class statistics"""
    try:
        if not HAVE_ONNX:
            print(f"ONNX library not available for {model_name}")
            return None
            
        # Load ONNX model
        try:
            model = onnx.load(model_path)
            # Run ONNX shape inference to populate value_info
            try:
                # strict=False keeps going even if some ops miss shapes
                model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
            except Exception as e:
                print(f"  Warning: shape inference failed for {model_name}: {e}")
        except Exception as e:
            print(f"Failed to load ONNX model {model_name}: {e}")
            return None
            
        # Create ONNX Runtime session for shape inference
        try:
            providers = ['CPUExecutionProvider']  # Use CPU for analysis
            session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            print(f"Failed to create ONNX Runtime session for {model_name}: {e}")
            return None
            
        # Get input information
        input_info = session.get_inputs()[0]
        H, W = input_size
        
        # Run shape inference by creating a dummy input
        dummy_input = np.zeros((1, 3, H, W), dtype=np.float32)
        try:
            # Run inference to get intermediate shapes (this is expensive but necessary for accurate analysis)
            session.run(None, {input_info.name: dummy_input})
        except Exception as e:
            print(f"Warning: Could not run inference for shape analysis on {model_name}: {e}")
            
        classifier = KernelClassifier()
        try:
            classifier.attention_tagged_node_names = tag_attention_patterns(model)
        except NameError:
            classifier.attention_tagged_node_names = set()
        
        # Build shape/type lookup map from inferred model
        shape_type_map = build_shape_type_map(model)
        
        def get_from_map(tname):
            return shape_type_map.get(tname, (None, None))
        
        per_class: Dict[int, Dict[str, float]] = {cid: {"F_c": 0.0, "U_c": 0.0} for cid in range(1, 8)}
        flops_calc = FLOPsCalculator()
        bytes_calc = BytesCalculator()
        
        total_nodes = len(model.graph.node)
        processed_nodes = 0
        compute_nodes = 0
        zeroF_compute_nodes = 0
        
        print(f"  Analyzing {total_nodes} nodes...")
        
        for i, node in enumerate(model.graph.node):
            processed_nodes += 1
            
            # Resolve shapes: activations & weights
            input_shapes, input_dtypes = [], []
            for input_name in node.input:
                if not input_name: 
                    continue
                s, d = get_from_map(input_name)
                input_shapes.append(s)
                input_dtypes.append(d or 'float32')
                
            # Output shape (prefer inferred)
            output_shape, output_dtype = (None, 'float32')
            if node.output and node.output[0]:
                output_shape, output_dtype = get_from_map(node.output[0])
                
            # Special handling for Conv: get true weight shape and groups
            groups = get_attr(node, 'group', 1) if node.op_type == 'Conv' else 1
            weight_shape = None
            if node.op_type == 'Conv' and len(node.input) >= 2:
                weight_shape, _ = get_from_map(node.input[1])
                
            # If any crucial shape is still None, last-chance inference (cheap)
            if node.op_type == 'Conv' and (output_shape is None or weight_shape is None):
                output_shape = infer_output_shape(node, input_shapes) if output_shape is None else output_shape
                
            # Skip nodes with no usable shapes
            if output_shape is None and not any(input_shapes):
                continue
                
            # Final fallback for output shape
            if output_shape is None:
                if len(input_shapes) > 0 and input_shapes[0] is not None:
                    output_shape = input_shapes[0]
                else:
                    continue  # Skip this node
                    
            if output_dtype is None:
                output_dtype = 'float32'
                
            # Classify the node
            cid = classifier.classify_node(node, input_shapes, weight_shape, groups)
            
            if processed_nodes <= 5:
                print(f"    Node {i}: {node.op_type} -> Class {cid} ({KERNEL_CLASSES[cid]})")
                
            # Calculate FLOPs based on operation type
            flops = 0
            op_type = node.op_type
            
            if op_type == 'Conv':
                flops = flops_calc.conv_flops(node, input_shapes, output_shape, weight_shape, groups)
                compute_nodes += 1
                
            elif op_type == 'MatMul':
                flops = flops_calc.matmul_flops(input_shapes, output_shape)
                compute_nodes += 1
                
            elif op_type == 'Gemm':
                flops = flops_calc.gemm_flops(node, input_shapes, output_shape)
                compute_nodes += 1
                
            elif op_type in ['Relu', 'Sigmoid', 'Tanh', 'Add', 'Mul', 'Sub', 'Div', 'Clip']:
                flops = flops_calc.elementwise_flops(output_shape)
                
            elif op_type in ['MaxPool', 'AveragePool']:
                flops = flops_calc.pooling_flops(node, input_shapes, output_shape)
                
            elif op_type in ['ReduceMean', 'ReduceSum', 'ReduceMax', 'ReduceMin']:
                flops = flops_calc.reduction_flops(input_shapes, output_shape)
                
            elif op_type in ['Softmax', 'LogSoftmax']:
                flops = flops_calc.softmax_flops(input_shapes, output_shape)
                
            # Calculate bytes
            bytes_consumed = 0
            if input_shapes and input_dtypes and output_shape:
                bytes_consumed = bytes_calc.standard_bytes(input_shapes, input_dtypes, output_shape, output_dtype)
            
            # Track zero-FLOP compute nodes for fallback detection
            if cid in [1, 2, 3] and flops == 0:
                zeroF_compute_nodes += 1
                
            # Add to per-class statistics
            per_class[cid]["F_c"] += flops
            per_class[cid]["U_c"] += bytes_consumed
            
        print(f"  Processed {processed_nodes}/{total_nodes} nodes")
        
        total_compute_flops = sum(per_class[cid]["F_c"] for cid in [1, 2, 3])
        
        # Robust fallback trigger - much more conservative
        need_fallback = (
            total_compute_flops < 1e6 and zeroF_compute_nodes >= compute_nodes * 0.8
        )
        
        if need_fallback:
            print(f"  Triggering fallback analysis for {model_name}")
            per_class = analyze_model_fallback(model, input_size, per_class)
            
        # Calculate AI ratios
        for cid in per_class:
            flops = per_class[cid]["F_c"]
            bytes_val = per_class[cid]["U_c"]
            per_class[cid]["AI_c"] = flops / max(bytes_val, 1e-9)
            
        return per_class
        
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        return None

def analyze_model_fallback(model, input_size: Tuple[int, int], per_class: Dict) -> Dict:
    """Fallback analysis using simpler heuristics"""
    print("  Using fallback analysis...")
    
    # Reset per_class
    for cid in range(1, 8):
        per_class[cid] = {"F_c": 0.0, "U_c": 0.0}
        
    H, W = input_size
    
    for node in model.graph.node:
        op_type = node.op_type
        
        # Simple heuristic FLOPs based on operation type
        if op_type == 'Conv':
            # Assume standard conv with reasonable parameters
            flops = 50e6  # 50M FLOPs as rough estimate
            per_class[1]["F_c"] += flops
            per_class[1]["U_c"] += 10e6  # 10MB
            
        elif op_type in ['MatMul', 'Gemm']:
            # Assume reasonable matrix sizes
            flops = 10e6  # 10M FLOPs
            per_class[1]["F_c"] += flops
            per_class[1]["U_c"] += 5e6  # 5MB
            
        elif op_type in ['Relu', 'Sigmoid', 'Tanh', 'Add', 'Mul']:
            flops = 1e6  # 1M operations
            per_class[4]["F_c"] += flops
            per_class[4]["U_c"] += 1e6  # 1MB
            
        elif op_type in ['BatchNormalization', 'MaxPool', 'AveragePool']:
            flops = 2e6  # 2M operations
            per_class[5]["F_c"] += flops
            per_class[5]["U_c"] += 2e6  # 2MB
            
    return per_class

def print_analysis_results(model_name: str, per_class: Dict[str, Any]):
    """Print analysis results in a formatted table"""
    print(f"\n=== {model_name} ===")
    print(f"{'CID':<3} | {'Kernel Class':<30} | {'FLOPs (G)':<12} | {'Bytes (GB)':<12} | {'AI (FLOP/B)':<12}")
    print("-" * 85)
    
    total_flops = 0
    total_bytes = 0
    
    for cid in range(1, 8):
        if cid in per_class:
            flops = per_class[cid]["F_c"]
            bytes_val = per_class[cid]["U_c"] 
            ai = per_class[cid]["AI_c"]
            
            print(f"{cid:<3} | {KERNEL_CLASSES[cid]:<30} | {flops/1e9:<12.3f} | {bytes_val/1e9:<12.3f} | {ai:<12.3f}")
            
            total_flops += flops
            total_bytes += bytes_val
            
    print("-" * 85)
    print(f"{'TOT':<3} | {'Total':<30} | {total_flops/1e9:<12.3f} | {total_bytes/1e9:<12.3f} | {total_flops/max(total_bytes, 1e-9):<12.3f}")

def validate_results(per_class: Dict[str, Any], model_name: str) -> bool:
    """Validate analysis results for reasonableness"""
    total_flops = sum(per_class[cid]["F_c"] for cid in per_class)
    total_bytes = sum(per_class[cid]["U_c"] for cid in per_class)
    
    if total_flops <= 0:
        print(f"⚠️  Warning: Total FLOPs is zero or negative for {model_name}")
        return False
        
    if total_bytes <= 0:
        print(f"⚠️  Warning: Total bytes is zero or negative for {model_name}")
        return False
        
    # Check for negative values in individual classes
    for cid in [1, 2, 3]:
        if cid in per_class:
            if per_class[cid]["F_c"] < 0 or per_class[cid]["U_c"] < 0:
                print(f"⚠️  Warning: Negative values in class {cid} for {model_name}")
                return False
                
    # Check if depthwise models have depthwise convolution FLOPs
    model_lower = model_name.lower()
    is_depthwise_model = any(dw_name in model_lower for dw_name in DEPTHWISE_MODELS)
    if is_depthwise_model and 2 in per_class:
        class2_flops = per_class[2]["F_c"]
        if class2_flops == 0:
            print(f"⚠️  Warning: Depthwise model {model_name} has no Class 2 FLOPs")
            
    # Validate against known FLOP values
    for known_model, expected_gflops in KNOWN_FLOPS.items():
        if known_model in model_lower:
            actual_gflops = total_flops / 1e9
            ratio = actual_gflops / expected_gflops
            if ratio < 0.5 or ratio > 2.0:
                print(f"⚠️  Warning: FLOP count for {model_name} ({actual_gflops:.2f}G) differs significantly from expected ({expected_gflops:.2f}G)")
                
    return True

def save_results_json(all_results: Dict[str, Any], output_path: str):
    """Save analysis results to JSON file"""
    summary = {
        "total_models_analyzed": len(all_results),
        "kernel_classes": KERNEL_CLASSES,
        "analysis_metadata": {
            "input_sizes": {"default": [224, 224], "inception_v3": [299, 299]},
            "batch_size": 1,
            "dtype": "float32", 
            "framework": "onnx",
            "device": "cpu",
            "preprocessing": "none (zeros input for graph analysis)",
            "matching_eval_script": "__eval_onnx_.py"
        }
    }
    
    output_data = {"summary": summary, "per_model_results": all_results}
    
    if all_results:
        # Calculate aggregated statistics
        aggregated: Dict[int, Dict[str, float]] = {}
        for cid in range(1, 8):
            aggregated[cid] = {"F_c": 0.0, "U_c": 0.0, "AI_c": 0.0}
            
        for model_name, per_class in all_results.items():
            for cid in range(1, 8):
                if cid in per_class:
                    aggregated[cid]["F_c"] += per_class[cid]["F_c"]
                    aggregated[cid]["U_c"] += per_class[cid]["U_c"]
                    
        # Calculate aggregated AI ratios
        for cid in aggregated:
            flops = aggregated[cid]["F_c"]
            bytes_val = aggregated[cid]["U_c"]
            aggregated[cid]["AI_c"] = flops / max(bytes_val, 1e-9)
            
        output_data["aggregated_statistics"] = aggregated
        
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\nResults saved to: {output_path}")

def main():
    """Main function to analyze all ONNX models"""
    models_dir = "models/onnx"
    output_file = "__ai_onnx_results_.json"
    
    if not os.path.exists(models_dir):
        print(f"Error: Models directory {models_dir} does not exist")
        return
        
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
    if not model_files:
        print(f"No .onnx files found in {models_dir}")
        return
        
    print(f"Found {len(model_files)} models to analyze:")
    for model_file in model_files:
        print(f"  - {model_file}")
        
    all_results: Dict[str, Any] = {}
    successful_analyses = 0
    
    for model_file in model_files:
        print(f"\n{'='*60}")
        print(f"Analyzing: {model_file}")
        print(f"{'='*60}")
        
        model_path = os.path.join(models_dir, model_file)
        model_name = model_file.replace('.onnx', '')
        
        # Skip problematic models
        SKIP = {'levit'}
        if any(k in model_name.lower() for k in SKIP):
            print(f"Skipping {model_name} (unstable shapes)")
            continue
        
        size = expected_input_size(model_name)
        input_size = (size, size)
        
        per_class = analyze_model(model_path, model_name, input_size)
        
        if per_class is not None:
            print_analysis_results(model_name, per_class)
            if validate_results(per_class, model_name):
                all_results[model_name] = per_class
                successful_analyses += 1
        else:
            print(f"Failed to analyze {model_name}")
            
    print(f"\n{'='*60}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Successfully analyzed: {successful_analyses}/{len(model_files)} models")
    
    if successful_analyses > 0:
        save_results_json(all_results, output_file)
        
        print(f"\nOverall Statistics:")
        total_flops_all = 0
        total_bytes_all = 0
        
        for model_name, per_class in all_results.items():
            model_flops = sum(per_class[cid]["F_c"] for cid in per_class)
            model_bytes = sum(per_class[cid]["U_c"] for cid in per_class)
            total_flops_all += model_flops
            total_bytes_all += model_bytes
            
        print(f"Total FLOPs across all models: {total_flops_all/1e12:.3f} TFLOPs")
        print(f"Total Bytes across all models: {total_bytes_all/1e12:.3f} TB")
        print(f"Overall AI ratio: {total_flops_all/max(total_bytes_all, 1e-9):.3f}")
    else:
        print("No models were successfully analyzed.")

if __name__ == "__main__":
    main()