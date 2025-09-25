#!/usr/bin/env python3
# type: ignore
# Usage: python __ai_tflite_.py

import os, json, warnings, math
import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple

warnings.filterwarnings("ignore")

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

# FLOPs calculator (simplified heuristics)
def conv_flops(params, out_shape):
    cout, kh, kw, cin = params
    _, hout, wout, _ = out_shape
    return 2 * cout * hout * wout * (cin * kh * kw)

def matmul_flops(a_shape, b_shape):
    if len(a_shape) < 2 or len(b_shape) < 2: return 0
    M, K = a_shape[-2], a_shape[-1]
    K2, N = b_shape[-2], b_shape[-1]
    if K != K2: return 0
    return 2 * M * N * K

def tensor_bytes(shape, dtype):
    if not shape: return 0
    return np.prod(shape) * DTYPE_SIZES.get(dtype, 4)

def analyze_tflite(model_path: str, model_name: str) -> Dict[str, Any]:
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        details = interpreter.get_tensor_details()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    per_class = {cid: {"F_c": 0.0, "U_c": 0.0} for cid in range(1, 8)}

    for d in details:
        shape, dtype = d['shape'], d['dtype'].name
        size_bytes = tensor_bytes(shape, dtype)

        op_type = d.get('name', '').lower()

        if 'conv2d' in op_type and 'depthwise' not in op_type:
            per_class[1]["F_c"] += 50e6
            per_class[1]["U_c"] += size_bytes
        elif 'depthwise' in op_type:
            per_class[2]["F_c"] += 20e6
            per_class[2]["U_c"] += size_bytes
        elif 'matmul' in op_type or 'fullyconnected' in op_type:
            per_class[1]["F_c"] += 10e6
            per_class[1]["U_c"] += size_bytes
        elif any(x in op_type for x in ['add','mul','sub','div','relu','sigmoid','tanh']):
            per_class[4]["F_c"] += 1e6
            per_class[4]["U_c"] += size_bytes
        elif any(x in op_type for x in ['maxpool','averagepool','reduce','softmax']):
            per_class[5]["F_c"] += 2e6
            per_class[5]["U_c"] += size_bytes
        else:
            per_class[7]["F_c"] += 1e5
            per_class[7]["U_c"] += size_bytes

    # AI ratios
    for cid in per_class:
        f, u = per_class[cid]["F_c"], per_class[cid]["U_c"]
        per_class[cid]["AI_c"] = f / max(u, 1e-9)

    return per_class

def print_results(model_name, per_class):
    print(f"\n=== {model_name} ===")
    print(f"{'CID':<3} | {'Kernel Class':<30} | {'FLOPs (G)':<12} | {'Bytes (MB)':<12} | {'AI (FLOP/B)':<12}")
    print("-" * 85)
    totalF, totalU = 0, 0
    for cid in range(1, 8):
        f, u, ai = per_class[cid]["F_c"], per_class[cid]["U_c"], per_class[cid]["AI_c"]
        print(f"{cid:<3} | {KERNEL_CLASSES[cid]:<30} | {f/1e9:<12.3f} | {u/1e6:<12.3f} | {ai:<12.3f}")
        totalF += f; totalU += u
    print("-"*85)
    print(f"TOT | {'Total':<30} | {totalF/1e9:<12.3f} | {totalU/1e6:<12.3f} | {totalF/max(totalU,1e-9):<12.3f}")

def main():
    models_dir = "models/tflite"
    output_file = "__ai_tflite_results_.json"

    if not os.path.exists(models_dir):
        print(f"Error: {models_dir} does not exist")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith('.tflite')]
    if not model_files:
        print("No .tflite files found")
        return

    all_results = {}
    for mf in model_files:
        model_path = os.path.join(models_dir, mf)
        name = mf.replace(".tflite","")
        print(f"\nAnalyzing {mf}")
        res = analyze_tflite(model_path, name)
        if res:
            print_results(name, res)
            all_results[name] = res

    with open(output_file, "w") as f:
        json.dump({"summary": {"kernel_classes":KERNEL_CLASSES}, "per_model_results": all_results}, f, indent=2)
    print(f"\nSaved results to {output_file}")

if __name__ == "__main__":
    main()
