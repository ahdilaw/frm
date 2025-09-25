#!/usr/bin/env python3
# type: ignore
# Usage: python __ai_tflite_.py

import os, json, warnings
import numpy as np
import tensorflow as tf
from typing import Dict, Any
from typing import Optional

warnings.filterwarnings("ignore")

# ===== Constants =====
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

# ===== Helpers (robust to TFLite quirks) =====
def dtype_name(x) -> str:
    # Works for numpy types (np.float32), dtype objects (np.dtype('float32')), and strings
    try:
        return np.dtype(x).name
    except Exception:
        return "float32"

def normalize_shape(s) -> list:
    # TFLite may give np.ndarray (possibly empty) or include -1 (dynamic)
    if s is None:
        return []
    try:
        arr = list(s) if hasattr(s, "__iter__") and not isinstance(s, (str, bytes)) else [int(s)]
    except Exception:
        return []
    out = []
    for v in arr:
        try:
            iv = int(v)
            out.append(iv if iv > 0 else 1)   # replace -1/0 with 1 for accounting
        except Exception:
            out.append(1)
    return out

def tensor_bytes(shape, dtype) -> int:
    shape = normalize_shape(shape)      # ALWAYS a Python list now
    if len(shape) == 0:
        return 0
    return int(np.prod(shape)) * DTYPE_SIZES.get(dtype_name(dtype), 4)

# ===== Very simple FLOPs heuristics (can be upgraded later) =====
def analyze_tflite(model_path: str, model_name: str) -> Optional[Dict[str, Any]]:
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        details = interpreter.get_tensor_details()
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return None

    per_class = {cid: {"F_c": 0.0, "U_c": 0.0} for cid in range(1, 8)}

    for d in details:
        # Prefer shape_signature (handles dynamic dims); fallback to shape
        raw_shape = d.get("shape_signature", None)
        if raw_shape is None or (hasattr(raw_shape, "size") and raw_shape.size == 0):
            raw_shape = d.get("shape", None)

        dtype = d.get("dtype")
        size_bytes = tensor_bytes(raw_shape, dtype)

        # NOTE: d['name'] is a tensor name, not strictly an op. Still useful for coarse mapping.
        op_name = (d.get("name") or "").lower()

        if "depthwise" in op_name:
            per_class[2]["F_c"] += 20e6
            per_class[2]["U_c"] += size_bytes
        elif "conv2d" in op_name:
            per_class[1]["F_c"] += 50e6
            per_class[1]["U_c"] += size_bytes
        elif ("matmul" in op_name) or ("fullyconnected" in op_name) or ("dense" in op_name):
            per_class[1]["F_c"] += 10e6
            per_class[1]["U_c"] += size_bytes
        elif any(x in op_name for x in ["add","mul","sub","div","relu","prelu","leakyrelu","sigmoid","tanh","gelu","elu","selu","swish","hard_swish"]):
            per_class[4]["F_c"] += 1e6
            per_class[4]["U_c"] += size_bytes
        elif any(x in op_name for x in ["maxpool","averagepool","avgpool","reduce","mean","sum","softmax","logsoftmax","argmax","argmin","batchnorm","layernorm","instancenorm","groupnorm"]):
            per_class[5]["F_c"] += 2e6
            per_class[5]["U_c"] += size_bytes
        elif any(x in op_name for x in ["gather","scatter","embedding"]):
            per_class[6]["F_c"] += 5e5
            per_class[6]["U_c"] += size_bytes
        elif any(x in op_name for x in ["reshape","transpose","concat","split","slice","pad","squeeze","unsqueeze","flatten","expand","tile","identity","cast","quantize","dequantize"]):
            per_class[7]["F_c"] += 1e5
            per_class[7]["U_c"] += size_bytes
        else:
            # Unknown → count as data movement (very cheap FLOPs; bytes dominate)
            per_class[7]["F_c"] += 5e4
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
    totalF, totalU = 0.0, 0.0
    for cid in range(1, 8):
        f, u, ai = per_class[cid]["F_c"], per_class[cid]["U_c"], per_class[cid]["AI_c"]
        print(f"{cid:<3} | {KERNEL_CLASSES[cid]:<30} | {f/1e9:<12.3f} | {u/1e6:<12.3f} | {ai:<12.3f}")
        totalF += f; totalU += u
    print("-" * 85)
    print(f"TOT | {'Total':<30} | {totalF/1e9:<12.3f} | {totalU/1e6:<12.3f} | {totalF/max(totalU,1e-9):<12.3f}")

def main():
    # Point this where your files landed (e.g., "models/tflite" or "models")
    models_dir = "models/tflite"
    output_file = "__ai_tflite_results_.json"

    if not os.path.exists(models_dir):
        print(f"Error: {models_dir} does not exist")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith(".tflite")]
    if not model_files:
        print("No .tflite files found")
        return

    all_results: Dict[str, Any] = {}
    for mf in model_files:
        model_path = os.path.join(models_dir, mf)
        name = mf.replace(".tflite", "")
        print(f"\nAnalyzing {mf}")
        res = analyze_tflite(model_path, name)
        if res:
            print_results(name, res)
            all_results[name] = res

    with open(output_file, "w") as f:
        json.dump({"summary": {"kernel_classes": KERNEL_CLASSES},
                   "per_model_results": all_results}, f, indent=2)
    print(f"\nSaved results to {output_file}")

if __name__ == "__main__":
    # Optional: quiet TF logs
    # os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    main()
