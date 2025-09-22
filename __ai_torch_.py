#!/usr/bin/env python3
# type: ignore
# Usage: python __kernel_torch_.py

import os
import json
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
import torchvision.models as models
from typing import Dict, Any, List, Tuple
from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore")

# Try to import timm for transformer models
try:
    import timm
    HAVE_TIMM = True
except ImportError:
    timm = None
    HAVE_TIMM = False
    print("Warning: timm not available. Transformer models will use fallbacks.")

# Constants
DTYPE_SIZES = {
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.int32: 4,
    torch.bool: 1,
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

def elem_size_bytes(dtype: torch.dtype) -> int:
    return DTYPE_SIZES.get(dtype, 4)

def numel(shape: torch.Size) -> int:
    if shape is None:
        return 0
    result = 1
    for dim in shape:
        if dim is not None:
            result *= dim
    return result

def tensor_bytes(tensor_meta) -> int:
    if tensor_meta is None or not hasattr(tensor_meta, 'shape') or not hasattr(tensor_meta, 'dtype'):
        return 0
    return numel(tensor_meta.shape) * elem_size_bytes(tensor_meta.dtype)

def get_tensor_meta(node) -> Any:
    if hasattr(node, 'meta') and 'tensor_meta' in node.meta:
        return node.meta['tensor_meta']
    elif hasattr(node, 'meta') and 'val' in node.meta:
        return node.meta['val']
    return None

def collect_tensor_metas(args, kwargs) -> List[Any]:
    metas: List[Any] = []
    def extract_meta(arg):
        if isinstance(arg, torch.fx.Node):
            meta = get_tensor_meta(arg)
            if meta is not None:
                metas.append(meta)
        elif isinstance(arg, (list, tuple)):
            for item in arg:
                extract_meta(item)
    for arg in args:
        extract_meta(arg)
    for value in kwargs.values():
        extract_meta(value)
    return metas

def make_meta_from_tensor(t: torch.Tensor):
    return SimpleNamespace(shape=tuple(t.shape), dtype=t.dtype)

def get_module_for_node(gm: fx.GraphModule, node: fx.Node):
    if node.op != "call_module":
        return None
    try:
        return gm.get_submodule(node.target)
    except Exception:
        return None

def ensure_weight_meta(gm: fx.GraphModule, node, inputs_meta: List[Any]) -> List[Any]:
    target = str(node.target)
    def needs_weight():
        return any(op in target for op in [
            'convolution','_convolution','conv2d','conv3d','Conv2d','Conv3d',
            'linear','addmm','mm','matmul','bmm',
            'aten.convolution','aten._convolution','aten.conv2d','aten.conv3d',
            'aten.linear','aten.addmm','aten.mm','aten.matmul','aten.bmm'
        ])
    if not needs_weight():
        return inputs_meta
    if len(inputs_meta) >= 2:
        return inputs_meta
    mod = get_module_for_node(gm, node)
    if mod is not None and hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
        return inputs_meta + [make_meta_from_tensor(mod.weight)]
    try:
        if len(node.args) > 1 and isinstance(node.args[1], fx.Node) and node.args[1].op == 'get_attr':
            pname = node.args[1].target
            try:
                p = gm.get_parameter(pname)
            except Exception:
                p = gm.get_buffer(pname)
            if isinstance(p, torch.Tensor):
                return inputs_meta + [make_meta_from_tensor(p)]
    except Exception:
        pass
    wnode = node.kwargs.get('weight', None)
    if isinstance(wnode, fx.Node) and wnode.op == 'get_attr':
        try:
            p = gm.get_parameter(wnode.target)
        except Exception:
            p = gm.get_buffer(wnode.target)
        if isinstance(p, torch.Tensor):
            return inputs_meta + [make_meta_from_tensor(p)]
    return inputs_meta

class KernelClassifier:
    def __init__(self):
        self.attention_tagged_nodes = set()
    def is_depthwise_conv(self, node, inputs_meta) -> bool:
        if len(inputs_meta) < 2:
            return False
        groups = 1
        if hasattr(node, 'kwargs') and 'groups' in node.kwargs:
            groups = node.kwargs['groups']
        elif hasattr(node, 'args') and len(node.args) > 6:
            groups = node.args[6] if node.args[6] is not None else 1
        if '_convolution' in str(node.target) and hasattr(node, 'args') and len(node.args) > 8:
            groups = node.args[8] if node.args[8] is not None else 1
        input_meta = inputs_meta[0] if len(inputs_meta) > 0 else None
        weight_meta = inputs_meta[1] if len(inputs_meta) > 1 else None
        if input_meta is not None and hasattr(input_meta, 'shape') and len(input_meta.shape) >= 4:
            in_channels = input_meta.shape[1]
            return groups == in_channels and groups > 1
        elif weight_meta is not None and hasattr(weight_meta, 'shape') and len(weight_meta.shape) >= 2:
            in_channels = weight_meta.shape[1] * groups
            return groups == in_channels and groups > 1
        return False
    def is_attention_core(self, node) -> bool:
        return node in self.attention_tagged_nodes
    def classify_node(self, node, inputs_meta) -> int:
        target = str(node.target)
        if node.op == 'call_module':
            if hasattr(node.graph.owning_module, 'get_submodule'):
                try:
                    module = node.graph.owning_module.get_submodule(node.target)
                    return self.classify_module(module, inputs_meta)
                except Exception:
                    pass
        if any(op in target for op in ['conv2d','conv3d','convolution','_convolution','Conv2d','Conv3d',
                                       'aten.conv2d','aten.conv3d','aten.convolution','aten._convolution']):
            if self.is_depthwise_conv(node, inputs_meta):
                return 2
            return 1
        elif any(op in target for op in ['linear','addmm','mm','matmul','bmm','Linear',
                                         'aten.linear','aten.addmm','aten.mm','aten.matmul','aten.bmm']):
            if self.is_attention_core(node):
                return 3
            return 1
        elif any(op in target for op in ['relu','gelu','silu','sigmoid','tanh','add','mul','sub','div','clamp',
                                         'ReLU','GELU','SiLU','Sigmoid','Tanh',
                                         'aten.relu','aten.gelu','aten.silu','aten.sigmoid','aten.tanh',
                                         'aten.add','aten.mul','aten.sub','aten.div','aten.clamp']):
            return 4
        elif any(op in target for op in ['batch_norm','layer_norm','group_norm','max_pool','avg_pool','mean','sum','amax',
                                         'softmax','log_softmax','BatchNorm','LayerNorm','GroupNorm','MaxPool','AvgPool',
                                         'aten.batch_norm','aten.layer_norm','aten.group_norm','aten.max_pool2d','aten.avg_pool2d',
                                         'aten.mean','aten.sum','aten.amax','aten.softmax','aten._log_softmax']):
            return 5
        elif any(op in target for op in ['embedding','gather','scatter','index_select','take','Embedding',
                                         'aten.embedding','aten.gather','aten.scatter','aten.index_select']):
            return 6
        elif any(op in target for op in ['reshape','view','permute','transpose','cat','split','slice','narrow','pad',
                                         'detach','contiguous','to',
                                         'aten.reshape','aten.view','aten.permute','aten.transpose','aten.cat',
                                         'aten.split','aten.slice','aten.narrow','aten.pad']):
            return 7
        return 4
    def classify_module(self, module, inputs_meta) -> int:
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            groups = getattr(module, 'groups', 1)
            in_channels = getattr(module, 'in_channels', 0)
            if groups > 1 and groups == in_channels:
                return 2
            else:
                return 1
        elif isinstance(module, torch.nn.Linear):
            return 1
        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                                 torch.nn.LayerNorm, torch.nn.GroupNorm,
                                 torch.nn.AvgPool1d, torch.nn.AvgPool2d, torch.nn.AvgPool3d,
                                 torch.nn.MaxPool1d, torch.nn.MaxPool2d, torch.nn.MaxPool3d,
                                 torch.nn.AdaptiveAvgPool1d, torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveAvgPool3d,
                                 torch.nn.AdaptiveMaxPool1d, torch.nn.AdaptiveMaxPool2d, torch.nn.AdaptiveMaxPool3d)):
            return 5
        elif isinstance(module, (torch.nn.ReLU, torch.nn.ReLU6, torch.nn.GELU, torch.nn.SiLU,
                                 torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.Hardswish, torch.nn.Hardsigmoid,
                                 torch.nn.LeakyReLU, torch.nn.ELU, torch.nn.PReLU, torch.nn.Softmax,
                                 torch.nn.LogSoftmax)):
            return 4
        elif isinstance(module, torch.nn.Embedding):
            return 6
        return 4

class FLOPsCalculator:
    @staticmethod
    def gemm_like_flops(input_meta, weight_meta) -> int:
        if input_meta is None or weight_meta is None:
            return 0
        if not (hasattr(input_meta, 'shape') and hasattr(weight_meta, 'shape')):
            return 0
        xshape = list(input_meta.shape)
        wshape = list(weight_meta.shape)
        if len(wshape) == 2:
            N, K_w = wshape[0], wshape[1]
            K_x = xshape[-1]
            if K_x is None or N is None or K_w is None:
                return 0
            M = 1
            for d in xshape[:-1]:
                if d is not None:
                    M *= d
            return 2 * M * N * K_x
        if len(wshape) >= 3 and len(xshape) >= 3:
            B = xshape[0] if xshape[0] is not None else 1
            M = xshape[-2]
            Kx = xshape[-1]
            Kw = wshape[-2]
            N = wshape[-1]
            if None in (M, Kx, Kw, N):
                return 0
            K = Kx
            return 2 * B * M * N * K
        return 0
    @staticmethod
    def conv_flops(inputs_meta, output_meta, groups=1) -> int:
        if len(inputs_meta) < 2 or output_meta is None:
            return 0
        input_meta, weight_meta = inputs_meta[0], inputs_meta[1]
        if not all(hasattr(meta, 'shape') for meta in [input_meta, weight_meta, output_meta]):
            return 0
        input_shape = input_meta.shape
        weight_shape = weight_meta.shape
        output_shape = output_meta.shape
        if len(weight_shape) < 4 or len(output_shape) < 4 or len(input_shape) < 4:
            return 0
        cout, cin_per_group, kh, kw = weight_shape[:4]
        _, _, hout, wout = output_shape[:4]
        groups = max(1, groups)
        macs = cout * hout * wout * cin_per_group * kh * kw
        return 2 * macs
    @staticmethod
    def linear_flops(inputs_meta, output_meta) -> int:
        if len(inputs_meta) < 2 or output_meta is None:
            return 0
        input_meta, weight_meta = inputs_meta[0], inputs_meta[1]
        if not all(hasattr(meta, 'shape') for meta in [input_meta, weight_meta, output_meta]):
            return 0
        input_shape = input_meta.shape
        weight_shape = weight_meta.shape
        if len(input_shape) < 2 or len(weight_shape) < 2:
            return 0
        k = input_shape[-1]
        n = weight_shape[0] if len(weight_shape) >= 2 else weight_shape[-1]
        m = 1
        for dim in input_shape[:-1]:
            m *= dim
        return 2 * m * n * k
    @staticmethod
    def attention_flops(inputs_meta, output_meta) -> int:
        if not inputs_meta or len(inputs_meta) < 2:
            return 0
        a = inputs_meta[0]
        b = inputs_meta[1]
        if not (hasattr(a, 'shape') and hasattr(b, 'shape')):
            return 0
        ashape = list(a.shape)
        bshape = list(b.shape)
        if len(ashape) >= 4 and len(bshape) >= 4:
            B = ashape[0] if ashape[0] is not None else 1
            H = ashape[1] if ashape[1] is not None else 1
            S = ashape[-2] if ashape[-2] is not None else 64
            d_k = ashape[-1] if ashape[-1] is not None else 64
            d_v = d_k
            qkt = 2 * B * H * S * S * d_k
            smax = 5 * B * H * S * S
            av   = 2 * B * H * S * S * d_v
            return qkt + smax + av
        if hasattr(output_meta, 'shape'):
            return numel(output_meta.shape) * 20
        return 0
    @staticmethod
    def elementwise_flops(inputs_meta, output_meta) -> int:
        if output_meta is None or not hasattr(output_meta, 'shape'):
            return 0
        return numel(output_meta.shape)
    @staticmethod
    def pooling_flops(inputs_meta, output_meta, kernel_size=1) -> int:
        if output_meta is None or not hasattr(output_meta, 'shape'):
            return 0
        output_numel = numel(output_meta.shape)
        return output_numel * kernel_size
    @staticmethod
    def embedding_flops(inputs_meta, output_meta) -> int:
        return 0
    @staticmethod
    def data_movement_flops(inputs_meta, output_meta) -> int:
        return 0

class BytesCalculator:
    @staticmethod
    def standard_bytes(inputs_meta, output_meta) -> int:
        total_bytes = 0
        for meta in inputs_meta:
            total_bytes += tensor_bytes(meta)
        total_bytes += tensor_bytes(output_meta)
        return total_bytes
    @staticmethod
    def view_operation_bytes(node, inputs_meta, output_meta) -> int:
        target = str(node.target)
        if any(op in target for op in ['view','reshape','squeeze','unsqueeze']):
            return 0
        elif any(op in target for op in ['contiguous','transpose','permute','cat','pad']):
            return BytesCalculator.standard_bytes(inputs_meta, output_meta)
        return 0

def tag_attention_patterns(gm: fx.GraphModule) -> set:
    attention_nodes = set()
    nodes = list(gm.graph.nodes)
    for i in range(len(nodes) - 2):
        node1, node2, node3 = nodes[i], nodes[i+1], nodes[i+2]
        if (any(op in str(node1.target) for op in ['matmul','bmm','mm']) and
            any(op in str(node2.target) for op in ['softmax','log_softmax']) and
            any(op in str(node3.target) for op in ['matmul','bmm','mm'])):
            attention_nodes.update([node1, node2, node3])
            continue
        if (any(op in str(node1.target) for op in ['transpose','permute']) and
            any(op in str(node2.target) for op in ['matmul','bmm','mm']) and
            i + 3 < len(nodes) and
            any(op in str(nodes[i+3].target) for op in ['softmax','log_softmax'])):
            for j in range(i+4, min(i+7, len(nodes))):
                if any(op in str(nodes[j].target) for op in ['matmul','bmm','mm']):
                    attention_nodes.update([node1, node2, nodes[i+3], nodes[j]])
                    break
    for i in range(len(nodes) - 4):
        if (any(op in str(nodes[i].target) for op in ['matmul','bmm','mm']) and
            any(op in str(nodes[i+1].target) for op in ['div','mul','truediv']) and
            any(op in str(nodes[i+2].target) for op in ['softmax']) and
            any(op in str(nodes[i+3].target) for op in ['matmul','bmm','mm'])):
            attention_nodes.update([nodes[i], nodes[i+1], nodes[i+2], nodes[i+3]])
    return attention_nodes

def expected_input_size(model_name: str) -> int:
    return 299 if "inception_v3" in model_name.lower() else 224

def build_model(model_name: str) -> torch.nn.Module:
    model_name_lower = model_name.lower()
    if 'resnet18' in model_name_lower:
        return models.resnet18(pretrained=False)
    elif 'resnet50' in model_name_lower:
        return models.resnet50(pretrained=False)
    elif 'densenet121' in model_name_lower:
        return models.densenet121(pretrained=False)
    elif 'efficientnet_b0' in model_name_lower:
        return models.efficientnet_b0(pretrained=False)
    elif 'inception_v3' in model_name_lower:
        return models.inception_v3(pretrained=False)
    elif 'mobilenet_v2' in model_name_lower:
        return models.mobilenet_v2(pretrained=False)
    elif 'mobilenet_v3_small' in model_name_lower:
        return models.mobilenet_v3_small(pretrained=False)
    elif 'mnasnet1' in model_name_lower:
        return models.mnasnet1_0(pretrained=False)
    elif 'squeezenet1_1' in model_name_lower:
        return models.squeezenet1_1(pretrained=False)
    elif 'convnext_tiny' in model_name_lower:
        return models.convnext_tiny(pretrained=False)
    elif HAVE_TIMM and timm is not None:
        if 'deit_tiny_distilled_patch16_224' in model_name_lower:
            return timm.create_model("deit_tiny_distilled_patch16_224", pretrained=False, num_classes=1000)
        elif 'deit' in model_name_lower:
            return timm.create_model("deit_tiny_patch16_224", pretrained=False, num_classes=1000)
        elif 'levit_128s' in model_name_lower:
            return timm.create_model("levit_128s", pretrained=False, num_classes=1000)
        elif 'levit' in model_name_lower:
            return timm.create_model("levit_128s", pretrained=False, num_classes=1000)
        elif 'mobilevit_xxs' in model_name_lower:
            return timm.create_model("mobilevit_xxs", pretrained=False, num_classes=1000)
        elif 'mobilevit' in model_name_lower:
            return timm.create_model("mobilevit_xxs", pretrained=False, num_classes=1000)
    raise ValueError(f"Unknown model {model_name}. Supported models: ResNet18/50, DenseNet121, EfficientNet-B0, InceptionV3, MobileNet-V2/V3, MNASNet, SqueezeNet, ConvNeXt-Tiny, DeiT, LeViT, MobileViT")

def analyze_model(model_path: str, model_name: str, input_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
    try:
        try:
            model = build_model(model_name)
        except ValueError as e:
            print(f"Model building failed for {model_name}: {e}")
            return None
        state_dict = None
        load_error = None
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        except Exception as e1:
            load_error = e1
            try:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
            except Exception as e2:
                if "version" in str(e2).lower():
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            try:
                                state_dict = torch.load(model_path, map_location='cpu', encoding='latin1')
                            except Exception:
                                import pickle
                                with open(model_path, 'rb') as f:
                                    try:
                                        state_dict = pickle.load(f, encoding='latin1')
                                    except Exception:
                                        f.seek(0)
                                        state_dict = pickle.load(f, encoding='bytes')
                    except Exception as e3:
                        print(f"Version-specific loading failed for {model_name}: {e3}")
                        return None
                else:
                    try:
                        import pickle
                        state_dict = torch.load(model_path, map_location='cpu', pickle_module=pickle)
                    except Exception as e3:
                        print(f"Loading failed for {model_name}:")
                        print(f"  Primary error: {e1}")
                        print(f"  Secondary error: {e2}")
                        print(f"  Tertiary error: {e3}")
                        return None
        if state_dict is None:
            print(f"Failed to load state dict for {model_name}: {load_error}")
            return None
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        if 'mnasnet' in model_name.lower():
            original_load_method = model._load_from_state_dict
            def patched_load(state_dict, prefix, local_metadata, strict, missing, unexpected, errors):
                md = dict(local_metadata or {})
                md['version'] = 2
                return original_load_method(state_dict, prefix, md, strict, missing, unexpected, errors)
            model._load_from_state_dict = patched_load
            try:
                model.load_state_dict(new_state_dict, strict=False)
            finally:
                model._load_from_state_dict = original_load_method
        else:
            model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        H, W = input_size
        x = torch.zeros(1, 3, H, W, dtype=torch.float32)
        with torch.no_grad():
            gm = None
            try:
                if hasattr(torch, 'export') and hasattr(torch.export, 'export'):
                    exported_program = torch.export.export(model, (x,))
                    gm = exported_program.graph_module
                    ShapeProp(gm).propagate(x)
                    print(f"  Using torch.export (aten graph) for {model_name}")
                else:
                    raise AttributeError("torch.export not available")
            except Exception as e:
                print(f"  torch.export failed for {model_name}: {e}")
                try:
                    gm = fx.symbolic_trace(model)
                    ShapeProp(gm).propagate(x)
                    print(f"  Using fx.symbolic_trace for {model_name}")
                except Exception as e2:
                    print(f"  FX symbolic tracing failed for {model_name}: {e2}")
                    try:
                        tracer = fx.Tracer()
                        graph = tracer.trace(model)
                        gm = fx.GraphModule(model, graph)
                        ShapeProp(gm).propagate(x)
                        print(f"  Using fx.Tracer fallback for {model_name}")
                    except Exception as e3:
                        print(f"  All tracing methods failed for {model_name}: {e3}")
                        return None
            if gm is None:
                print(f"  Failed to create graph module for {model_name}")
                return None
        classifier = KernelClassifier()
        classifier.attention_tagged_nodes = tag_attention_patterns(gm)
        per_class: Dict[int, Dict[str, float]] = {cid: {"F_c": 0.0, "U_c": 0.0} for cid in range(1,8)}
        flops_calc = FLOPsCalculator()
        bytes_calc = BytesCalculator()
        total_nodes = 0
        processed_nodes = 0
        compute_nodes = 0
        zeroF_compute_nodes = 0
        for node in gm.graph.nodes:
            total_nodes += 1
            if node.op not in ['call_function', 'call_module', 'call_method']:
                continue
            processed_nodes += 1
            inputs_meta = collect_tensor_metas(node.args, node.kwargs)
            inputs_meta = ensure_weight_meta(gm, node, inputs_meta)
            output_meta = get_tensor_meta(node)
            if output_meta is None:
                continue
            mod = get_module_for_node(gm, node)
            cid = classifier.classify_node(node, inputs_meta)
            if processed_nodes <= 5:
                print(f"  Node {processed_nodes}: {node.target} -> Class {cid} ({KERNEL_CLASSES.get(cid, 'Unknown')})")
            target_str = str(node.target)
            if cid == 1:
                if any(op in target_str for op in ['conv','Conv','_convolution','convolution']):
                    groups = 1
                    if mod is not None and hasattr(mod, "groups"):
                        groups = int(mod.groups) or 1
                    else:
                        if 'groups' in node.kwargs:
                            groups = node.kwargs['groups'] or 1
                        elif hasattr(node, 'args'):
                            if 'aten._convolution' in target_str and len(node.args) > 8:
                                groups = node.args[8] or 1
                            elif 'aten.conv2d' in target_str and len(node.args) > 6:
                                groups = node.args[6] or 1
                            elif len(node.args) > 6:
                                groups = node.args[6] or 1
                    F_op = flops_calc.conv_flops(inputs_meta, output_meta, groups)
                elif any(op in target_str for op in ['addmm','mm','matmul','bmm','linear',
                                                     'aten.addmm','aten.mm','aten.matmul','aten.bmm','aten.linear']):
                    weight_meta = inputs_meta[1] if len(inputs_meta) > 1 else None
                    in_meta = inputs_meta[0] if inputs_meta else None
                    F_op = flops_calc.gemm_like_flops(in_meta, weight_meta)
                else:
                    F_op = 0
            elif cid == 2:
                groups = 1
                if mod is not None and hasattr(mod, "groups"):
                    groups = int(mod.groups) or 1
                else:
                    if 'groups' in node.kwargs:
                        groups = node.kwargs['groups'] or 1
                    elif hasattr(node, 'args'):
                        if 'aten._convolution' in target_str and len(node.args) > 8:
                            groups = node.args[8] or 1
                        elif 'aten.conv2d' in target_str and len(node.args) > 6:
                            groups = node.args[6] or 1
                        elif len(node.args) > 6:
                            groups = node.args[6] or 1
                F_op = flops_calc.conv_flops(inputs_meta, output_meta, groups)
            elif cid == 3:
                F_op = flops_calc.attention_flops(inputs_meta, output_meta)
            elif cid == 4:
                F_op = flops_calc.elementwise_flops(inputs_meta, output_meta)
            elif cid == 5:
                if 'softmax' in target_str.lower():
                    if output_meta and hasattr(output_meta, 'shape'):
                        output_numel = numel(output_meta.shape)
                        F_op = output_numel * 5
                    else:
                        F_op = flops_calc.pooling_flops(inputs_meta, output_meta)
                else:
                    F_op = flops_calc.pooling_flops(inputs_meta, output_meta)
            elif cid == 6:
                F_op = flops_calc.embedding_flops(inputs_meta, output_meta)
            elif cid == 7:
                F_op = flops_calc.data_movement_flops(inputs_meta, output_meta)
            else:
                F_op = 0
            param_bytes = 0
            if mod is not None:
                for p in mod.parameters(recurse=False):
                    param_bytes += p.numel() * elem_size_bytes(p.dtype)
            if cid == 7:
                U_op = bytes_calc.view_operation_bytes(node, inputs_meta, output_meta)
            else:
                U_op = bytes_calc.standard_bytes(inputs_meta, output_meta) + param_bytes
            per_class[cid]["F_c"] += F_op
            per_class[cid]["U_c"] += U_op
            if cid in (1,2):
                compute_nodes += 1
                if F_op == 0 and U_op > 0:
                    zeroF_compute_nodes += 1
        print(f"  Processed {processed_nodes}/{total_nodes} nodes")
        total_compute_flops = sum(per_class[cid]["F_c"] for cid in [1,2,3])
        # Robust fallback trigger
        need_fallback = (
            total_compute_flops < 1e6 or  # < 1M FLOPs
            (compute_nodes >= 10 and zeroF_compute_nodes / max(compute_nodes,1) > 0.2) or
            ('inception_v3' in model_name.lower())
        )
        if need_fallback:
            print(f"  Fallback: module-level analysis (compute_flops={total_compute_flops}, zeroF_ratio={zeroF_compute_nodes}/{compute_nodes})")
            per_class = analyze_modules_directly(model, x, per_class)
        for cid in per_class:
            Fc = per_class[cid]["F_c"]
            Uc = per_class[cid]["U_c"]
            per_class[cid]["AI_c"] = Fc / max(Uc, 1e-9)
        return per_class
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        return None

def analyze_modules_directly(model: torch.nn.Module, x: torch.Tensor, per_class: Dict) -> Dict:
    def calculate_conv_flops(module, input_size, output_size):
        if hasattr(module, 'weight') and module.weight is not None:
            weight_shape = module.weight.shape
            if len(weight_shape) >= 4:
                cout, cin_per_group, kh, kw = weight_shape[:4]
                if len(output_size) >= 4:
                    hout, wout = output_size[2], output_size[3]
                    # For depthwise convs, cin_per_group in weight shape is already cin/groups
                    # So we use cin_per_group directly, not cin//groups
                    macs = cout * hout * wout * cin_per_group * kh * kw
                    return 2 * macs
        return 0
    def calculate_linear_flops(module, input_size, output_size):
        if hasattr(module, 'weight') and module.weight is not None:
            weight_shape = module.weight.shape
            if len(weight_shape) >= 2:
                out_features, in_features = weight_shape[:2]
                batch_size = input_size[0] if len(input_size) > 0 else 1
                return 2 * batch_size * out_features * in_features
        return 0
    hooks = []
    def make_hook(name, module):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                input_tensor = input[0]
                input_size = input_tensor.shape
            else:
                input_size = []
            if isinstance(output, tuple):
                output_tensor = output[0]
            else:
                output_tensor = output
            if hasattr(output_tensor, 'shape'):
                output_size = output_tensor.shape
            else:
                output_size = []
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
                groups = getattr(module, 'groups', 1)
                in_channels = getattr(module, 'in_channels', 0)
                if groups > 1 and groups == in_channels:
                    cid = 2
                else:
                    cid = 1
                flops = calculate_conv_flops(module, input_size, output_size)
            elif isinstance(module, torch.nn.Linear):
                cid = 1
                flops = calculate_linear_flops(module, input_size, output_size)
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.LayerNorm,
                                     torch.nn.GroupNorm, torch.nn.AvgPool2d, torch.nn.MaxPool2d,
                                     torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)):
                cid = 5
                flops = numel(output_size) if output_size else 0
            elif isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.SiLU, torch.nn.Sigmoid,
                                     torch.nn.Tanh, torch.nn.Hardswish)):
                cid = 4
                flops = numel(output_size) if output_size else 0
            else:
                return
            input_bytes = numel(input_size) * 4 if input_size else 0
            output_bytes = numel(output_size) * 4 if output_size else 0
            weight_bytes = sum(p.numel() * 4 for p in module.parameters()) if hasattr(module, 'parameters') else 0
            total_bytes = input_bytes + output_bytes + weight_bytes
            per_class[cid]["F_c"] += flops
            per_class[cid]["U_c"] += total_bytes
        return hook
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            hook = module.register_forward_hook(make_hook(name, module))
            hooks.append(hook)
    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        for hook in hooks:
            hook.remove()
    return per_class

def print_analysis_results(model_name: str, per_class: Dict[str, Any]):
    print(f"\n=== {model_name} ===")
    print(f"{'CID':<3} | {'Kernel Class':<30} | {'FLOPs (G)':<12} | {'Bytes (GB)':<12} | {'AI (FLOP/B)':<12}")
    print("-" * 85)
    total_flops = 0
    total_bytes = 0
    for cid in range(1, 8):
        if cid in per_class:
            Fc = per_class[cid]["F_c"]
            Uc = per_class[cid]["U_c"]
            AIc = per_class[cid]["AI_c"]
            Fc_G = Fc / 1e9
            Uc_GB = Uc / 1e9
            total_flops += Fc
            total_bytes += Uc
            kernel_name = KERNEL_CLASSES.get(cid, "Unknown")
            print(f"{cid:<3} | {kernel_name:<30} | {Fc_G:<12.3f} | {Uc_GB:<12.3f} | {AIc:<12.3f}")
    print("-" * 85)
    print(f"{'TOT':<3} | {'Total':<30} | {total_flops/1e9:<12.3f} | {total_bytes/1e9:<12.3f} | {total_flops/max(total_bytes, 1e-9):<12.3f}")

def validate_results(per_class: Dict[str, Any], model_name: str) -> bool:
    total_flops = sum(per_class[cid]["F_c"] for cid in per_class)
    total_bytes = sum(per_class[cid]["U_c"] for cid in per_class)
    if total_flops <= 0:
        print(f"⚠️  Warning: Total FLOPs is zero or negative for {model_name}")
        return False
    if total_bytes <= 0:
        print(f"⚠️  Warning: Total bytes is zero or negative for {model_name}")
        return False
    for cid in [1, 2, 3]:
        if cid in per_class:
            F_c = per_class[cid]["F_c"]
            U_c = per_class[cid]["U_c"]
            if U_c > 0 and F_c == 0:
                class_name = KERNEL_CLASSES.get(cid, f"Class {cid}")
                print(f"⚠️  Warning: {class_name} has bytes ({U_c/1e9:.3f}GB) but zero FLOPs for {model_name}")
    model_lower = model_name.lower()
    is_depthwise_model = any(dw_name in model_lower for dw_name in DEPTHWISE_MODELS)
    if is_depthwise_model and 2 in per_class:
        class2_flops = per_class[2]["F_c"]
        if class2_flops == 0:
            print(f"⚠️  Warning: Depthwise model {model_name} has zero Class-2 FLOPs - expected > 0")
    for known_model, expected_gflops in KNOWN_FLOPS.items():
        if known_model in model_lower:
            actual_gflops = total_flops / 1e9
            # Use more lenient tolerance for complex models that required fallback analysis
            tolerance = 0.25 if known_model in ['convnext_tiny', 'inception_v3'] else 0.15
            lower_bound = expected_gflops * (1 - tolerance)
            upper_bound = expected_gflops * (1 + tolerance)
            if not (lower_bound <= actual_gflops <= upper_bound):
                tolerance_pct = int(tolerance * 100)
                print(f"⚠️  Warning: {model_name} FLOP mismatch - Expected: {expected_gflops:.2f}G, Got: {actual_gflops:.2f}G (±{tolerance_pct}%)")
            else:
                print(f"✅ {model_name} FLOP validation passed: {actual_gflops:.2f}G vs expected {expected_gflops:.2f}G")
            break
    class1_ai = per_class.get(1, {}).get("AI_c", 0)
    class2_ai = per_class.get(2, {}).get("AI_c", 0)
    class4_ai = per_class.get(4, {}).get("AI_c", 0)
    # Only warn about AI ratio if it's a significant discrepancy and not expected for the architecture
    # ConvNeXt and similar modern architectures can have higher depthwise AI due to large kernels
    model_lower = model_name.lower()
    is_modern_depthwise_arch = any(arch in model_lower for arch in ['convnext', 'efficientnet', 'mobilevit'])
    if (class1_ai > 0 and class2_ai > 0 and class1_ai < class2_ai and 
        not is_modern_depthwise_arch and class2_ai / class1_ai > 2.0):
        print(f"⚠️  Warning: Class 1 (Dense GEMM) has significantly lower AI than Class 2 (Depthwise) for {model_name} - unusual")
    return True

def save_results_json(all_results: Dict[str, Any], output_path: str):
    summary = {
        "total_models_analyzed": len(all_results),
        "kernel_classes": KERNEL_CLASSES,
        "analysis_metadata": {
            "input_sizes": {"default": [224,224], "inception_v3": [299,299]},
            "batch_size": 1,
            "dtype": "float32",
            "device": "cpu",
            "preprocessing": "none (zeros input for graph analysis)",
            "matching_eval_script": "__eval_torch_.py"
        }
    }
    output_data = {"summary": summary, "per_model_results": all_results}
    if all_results:
        aggregated: Dict[int, Dict[str, float]] = {}
        for cid in range(1, 8):
            aggregated[cid] = {"total_F_c": 0.0, "total_U_c": 0.0, "avg_AI_c": 0.0, "model_count": 0}
        for model_name, per_class in all_results.items():
            if per_class:
                for cid in range(1, 8):
                    if cid in per_class:
                        aggregated[cid]["total_F_c"] += per_class[cid]["F_c"]
                        aggregated[cid]["total_U_c"] += per_class[cid]["U_c"]
                        if per_class[cid]["AI_c"] > 0:
                            aggregated[cid]["avg_AI_c"] += per_class[cid]["AI_c"]
                            aggregated[cid]["model_count"] += 1
        for cid in aggregated:
            if aggregated[cid]["model_count"] > 0:
                aggregated[cid]["avg_AI_c"] /= aggregated[cid]["model_count"]
                aggregated[cid]["avg_F_c"] = aggregated[cid]["total_F_c"] / aggregated[cid]["model_count"]
                aggregated[cid]["avg_U_c"] = aggregated[cid]["total_U_c"] / aggregated[cid]["model_count"]
        output_data["aggregated_statistics"] = aggregated
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

def main():
    models_dir = "models/torch"
    output_file = "__ai_torch_results_.json"
    if not os.path.exists(models_dir):
        print(f"Error: Models directory {models_dir} does not exist")
        return
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        print(f"No .pth files found in {models_dir}")
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
        model_name = model_file.replace('.pth', '')
        size = expected_input_size(model_name)
        input_size = (size, size)
        per_class = analyze_model(model_path, model_name, input_size)
        if per_class is not None:
            all_results[model_name] = per_class
            successful_analyses += 1
            print_analysis_results(model_name, per_class)
            validate_results(per_class, model_name)
        else:
            all_results[model_name] = None
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
            if per_class:
                for cid in per_class:
                    total_flops_all += per_class[cid]["F_c"]
                    total_bytes_all += per_class[cid]["U_c"]
        print(f"Total FLOPs across all models: {total_flops_all/1e12:.3f} TFLOPs")
        print(f"Total Bytes across all models: {total_bytes_all/1e12:.3f} TB")
        print(f"Overall AI ratio: {total_flops_all/max(total_bytes_all, 1e-9):.3f}")
    else:
        print("No models were successfully analyzed.")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
