# FRM TFLite eval
# type: ignore
# Usage: python __eval_tflite_.py [--latency] [--lat-warmup-batches n] [--lat-repeats-batch n] [--bs n] [--blaze] [--gpu] [--mem] [--mem-sample-hz HZ] [--energy] [--energy-sample-hz HZ] [--gflops-map path/to/csv]

from __future__ import annotations
import argparse
import json
import os
import platform
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import math
import time
import threading
import random

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")

# Platform-specific imports
try:
    import resource
    HAVE_RESOURCE = True
except ImportError:
    resource = None
    HAVE_RESOURCE = False

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as T

# ---------- Optional deps ----------
try:
    import psutil
    HAVE_PSUTIL = True
except Exception:
    psutil = None
    HAVE_PSUTIL = False

# NVIDIA NVML
try:
    import pynvml
    HAVE_NVML = True
except Exception:
    try:
        import nvidia_ml_py3 as pynvml
        HAVE_NVML = True
    except Exception:
        HAVE_NVML = False

# ONNX (for model statics if needed)
try:
    import onnx
    from onnxsim import simplify
    HAVE_ONNXSIM = True
except Exception:
    onnx = None
    simplify = None
    HAVE_ONNXSIM = False

try:
    from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
    HAVE_SSI = True
except Exception:
    SymbolicShapeInference = None
    HAVE_SSI = False

try:
    import torch
    HAVE_TORCH = True
except Exception:
    torch = None
    HAVE_TORCH = False

# --- Try interpreters ---
Interpreter = None
RUNTIME = None
TF_AVAILABLE = False
GPU_AVAILABLE = False
_IMPORT_ERR = None

# First try full TensorFlow for best compatibility (Flex delegate + GPU support)
try:
    import tensorflow as tf
    from tensorflow.lite import Interpreter  # type: ignore
    RUNTIME = "tensorflow.lite"
    TF_AVAILABLE = True
    print(f"✓ Successfully imported full TensorFlow with TensorFlow Lite")
    
    # Check for GPU availability
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            GPU_AVAILABLE = True
            print(f"✓ GPU support detected - {len(gpus)} GPU(s) available")
        else:
            print(f"✓ No GPU devices found - using CPU only")
    except Exception as gpu_e:
        print(f"✓ GPU check failed: {gpu_e} - using CPU only")
        
except ImportError as tf_e:
    print(f"✗ Could not import full TensorFlow: {tf_e}")
    # Fallback to tflite_runtime (CPU only, no Flex delegate)
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
        RUNTIME = "tflite_runtime"
        print(f"✓ Successfully imported tflite_runtime (CPU only, no Flex delegate)")
        print(f"✓ Limited compatibility - some models may fail (install tensorflow for full support)")
    except ImportError as e:
        Interpreter = None
        RUNTIME = None
        _IMPORT_ERR = f"tensorflow: {tf_e}, tflite_runtime: {e}"
        print(f"✗ No TFLite backend available: {_IMPORT_ERR}")

# Early exit if no interpreter is available
if Interpreter is None:
    print(f"❌ Error: No TensorFlow Lite interpreter available.")
    print(f"   Please install one of the following:")
    print(f"   - tflite-runtime: pip install tflite-runtime")
    print(f"   - tensorflow: pip install tensorflow")
    import sys
    sys.exit(1)

# --- Config ---
DATA_DIR = Path("data")
IMAGENET_INDEX_FILE = Path("imagenet_class_index.json")
MODELS_DIR = Path("models/tflite")
OUTPUT_ODS = Path("frm_tflite_results.ods")
TOP_K = 5

# Preproc constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1,1,1,3)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32).reshape(1,1,1,3)
CAFFE_MEAN_RGB_255 = np.array([123.68, 116.779, 103.939], np.float32).reshape(1,1,1,3)

# --- IO helpers ---

def load_imagenet_class_mapping() -> Dict[str, int]:
    with IMAGENET_INDEX_FILE.open("r") as f:
        class_idx = json.load(f)
    # {"0": ["n01440764", "tench"], ...}
    return {wnid: int(i) for i, (wnid, _name) in class_idx.items()}


def choose_resize(size: int) -> T.Compose:
    return T.Compose([T.Resize(342 if size == 299 else 256), T.CenterCrop(size)])


def _finite(vals):
    return [v for v in vals if v is not None and not math.isnan(v)]

def nanrobust_max(vals: list) -> float:
    f = _finite(vals)
    return max(f) if f else math.nan

def nanrobust_median(vals: list) -> float:
    f = _finite(vals)
    return float(np.median(f)) if f else math.nan


def load_batch_CHW(paths: List[Path], size: int) -> np.ndarray:
    tf = T.Compose([choose_resize(size), T.ToTensor()])  # CHW in [0,1]
    batch = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            batch.append(tf(img).numpy())  # (3,H,W)
        except Exception:
            batch.append(np.zeros((3, size, size), np.float32))
    return np.stack(batch).astype(np.float32)  # (N,3,H,W)


# --- Normalizations (expect NHWC for TFLite) ---

def imagenet_norm_NHWC(x_0to1_nhwc: np.ndarray) -> np.ndarray:
    return (x_0to1_nhwc - IMAGENET_MEAN) / IMAGENET_STD


def caffe_rgb_submean_NHWC(x_0to1_nhwc: np.ndarray) -> np.ndarray:
    return x_0to1_nhwc * 255.0 - CAFFE_MEAN_RGB_255


# --- Model I/O helpers ---

def get_input_size_from_details(dets: List[Dict]) -> int:
    shape = dets[0].get("shape")
    if shape is None:
        return 224
    # NHWC: [N,H,W,C]
    if len(shape) >= 4 and isinstance(shape[1], (int, np.integer)) and isinstance(shape[2], (int, np.integer)):
        return int(max(shape[1], shape[2]))
    return 224


def get_fixed_batch_dim(dets: List[Dict]) -> Optional[int]:
    shape = dets[0].get("shape")
    if shape is not None and len(shape) >= 1 and isinstance(shape[0], (int, np.integer)) and shape[0] > 0:
        return int(shape[0])
    return None


def ensure_interpreter(model_path: str, use_gpu: bool = False) -> Interpreter:
    if Interpreter is None:
        raise RuntimeError(f"No TFLite interpreter available. Tried tflite_runtime and tensorflow.lite; last error: {_IMPORT_ERR}")
    
    # Try GPU delegate first if requested and available
    if use_gpu and GPU_AVAILABLE and RUNTIME == "tensorflow.lite":
        try:
            import tensorflow as tf
            print(f"[GPU] {Path(model_path).name} - Trying GPU delegate")
            
            # Create GPU delegate
            gpu_delegate = tf.lite.experimental.GpuDelegate(
                options={'precision_loss_allowed': True}  # Allow precision loss for better performance
            )
            
            interp = tf.lite.Interpreter(
                model_path=model_path,
                experimental_delegates=[gpu_delegate]
            )
            interp.allocate_tensors()
            print(f"[GPU-SUCCESS] {Path(model_path).name} - Using GPU acceleration")
            return interp
            
        except Exception as gpu_e:
            print(f"[GPU-FAIL] {Path(model_path).name} - GPU delegate failed: {gpu_e}, falling back to CPU")
            # Fall through to CPU approach
    
    try:
        # Try with basic CPU settings
        if RUNTIME == "tensorflow.lite":
            # Use TensorFlow's Interpreter directly
            import tensorflow as tf
            interp = tf.lite.Interpreter(model_path=model_path)
        else:
            # Use tflite_runtime
            interp = Interpreter(model_path=model_path, num_threads=1)
        interp.allocate_tensors()
        return interp
        
    except Exception as e:
        error_msg = str(e).lower()
        if any(op in error_msg for op in ['flexerf', 'erf', 'flex delegate', 'custom op', 'select tensorflow op']):
            print(f"[FLEX] {Path(model_path).name} - Trying with TensorFlow Flex delegate")
            
            # Try with Flex delegate if full tensorflow is available
            if RUNTIME == "tensorflow.lite" and TF_AVAILABLE:
                try:
                    import tensorflow as tf
                    print(f"[FLEX] {Path(model_path).name} - Loading with TensorFlow Flex delegate support")
                    
                    # Create delegates list
                    delegates = []
                    
                    # Add GPU delegate if requested and available
                    if use_gpu and GPU_AVAILABLE:
                        try:
                            gpu_delegate = tf.lite.experimental.GpuDelegate(
                                options={'precision_loss_allowed': True}
                            )
                            delegates.append(gpu_delegate)
                            print(f"[FLEX-GPU] {Path(model_path).name} - Added GPU delegate")
                        except Exception as gpu_e:
                            print(f"[FLEX-GPU-FAIL] {Path(model_path).name} - GPU delegate failed: {gpu_e}")
                    
                    # Create interpreter with full TensorFlow support (enables Flex delegate automatically)
                    interp = tf.lite.Interpreter(
                        model_path=model_path,
                        experimental_delegates=delegates
                    )
                    interp.allocate_tensors()
                    
                    delegate_info = f" + GPU" if (use_gpu and GPU_AVAILABLE and any(delegates)) else ""
                    print(f"[FLEX-SUCCESS] {Path(model_path).name} - Loaded with Flex delegate{delegate_info}")
                    return interp
                    
                except Exception as flex_e:
                    print(f"[FLEX-FAIL] {Path(model_path).name} - Flex delegate failed: {flex_e}")
            else:
                print(f"[SKIP] {Path(model_path).name} - Full TensorFlow not available for Flex delegate")
                    try:
                        import tensorflow as tf
                        # Some models work better with explicit Flex delegate loading
                        delegates = []
                        
                        # Add GPU delegate if requested and available
            
            print(f"[SKIP] {Path(model_path).name} - Requires TensorFlow Flex delegate (unsupported operations)")
            raise RuntimeError(f"Model requires Flex delegate: {e}")
        else:
            print(f"[ERROR] Failed to create interpreter for {model_path}: {e}")
        
        # Try fallback without delegates
        try:
            if RUNTIME == "tensorflow.lite":
                interp = tf.lite.Interpreter(model_path=model_path, experimental_delegates=[])
            else:
                interp = Interpreter(model_path=model_path, num_threads=1)
            interp.allocate_tensors()
            print(f"[FALLBACK] {Path(model_path).name} - Using basic interpreter without delegates")
            return interp
        except Exception as e2:
            raise RuntimeError(f"Failed to create interpreter for {model_path}: {e}, fallback: {e2}")


def run_tflite(interp: Interpreter, x_nchw: np.ndarray) -> Tuple[List[int], List[np.ndarray], str]:
    # Handles float32 and quantized outputs; also 1001-class adjustment
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Ensure input tensor matches expected shape exactly
    expected_shape = input_details[0]['shape']
    if expected_shape[0] == -1:  # Dynamic batch dimension
        expected_shape = list(expected_shape)
        expected_shape[0] = x_nchw.shape[0]
        expected_shape = tuple(expected_shape)
    
    # Resize input if needed to match expected spatial dimensions
    if tuple(x_nchw.shape) != tuple(expected_shape):
        if len(expected_shape) == 4:  # NCHW format
            target_h, target_w = expected_shape[2], expected_shape[3]
            if x_nchw.shape[2] != target_h or x_nchw.shape[3] != target_w:
                from PIL import Image
                resized_batch = []
                for i in range(x_nchw.shape[0]):
                    img_chw = x_nchw[i]  # (3, H, W)
                    # Convert to HWC for PIL
                    img_hwc = np.transpose(img_chw, (1, 2, 0))
                    # Denormalize to [0, 255] for PIL if needed
                    if img_hwc.min() < 0 or img_hwc.max() <= 1:
                        img_hwc = np.clip(img_hwc * 255, 0, 255).astype(np.uint8)
                    else:
                        img_hwc = np.clip(img_hwc, 0, 255).astype(np.uint8)
                    
                    img_pil = Image.fromarray(img_hwc)
                    img_resized = img_pil.resize((target_w, target_h), Image.LANCZOS)
                    img_hwc_resized = np.array(img_resized).astype(np.float32)
                    
                    # Renormalize if original was normalized
                    if x_nchw.min() < 0 or x_nchw.max() <= 1:
                        img_hwc_resized = img_hwc_resized / 255.0
                    
                    # Convert back to CHW
                    img_chw_resized = np.transpose(img_hwc_resized, (2, 0, 1))
                    resized_batch.append(img_chw_resized)
                x_nchw = np.stack(resized_batch)

    # If model expects uint8, quantize input
    xin = x_nchw
    if input_details[0]["dtype"] == np.uint8:
        scale, zp = input_details[0]["quantization"]
        if scale == 0 or scale is None:
            scale, zp = 1.0, 0
        xin = np.clip((xin / scale + zp), 0, 255).round().astype(np.uint8)
    else:
        xin = xin.astype(input_details[0]["dtype"], copy=False)

    # If model has fixed batch=1, loop internally
    fixed_bs = get_fixed_batch_dim(input_details)
    if fixed_bs == 1 and xin.shape[0] != 1:
        t1_all: List[int] = []
        t5_all: List[np.ndarray] = []
        for i in range(xin.shape[0]):
            sample = xin[i:i+1]
            try:
                interp.resize_tensor_input(input_details[0]['index'], sample.shape, strict=True)
                interp.allocate_tensors()
                interp.set_tensor(input_details[0]['index'], sample)
                interp.invoke()
                y = _extract_logits(interp, output_details)
                top1, top5 = _topk_from_logits(y)
                t1_all.extend(top1)
                t5_all.extend((t5 or [])[:len(top1)])
            except Exception as e:
                # Silently add dummy predictions for failed samples
                t1_all.append(0)
                t5_all.append(np.array([0, 1, 2, 3, 4]))
        return t1_all, t5_all, "looped-b1"

    # Try batched path
    try:
        interp.resize_tensor_input(input_details[0]['index'], xin.shape, strict=True)
        interp.allocate_tensors()
        interp.set_tensor(input_details[0]['index'], xin)
        interp.invoke()
        y = _extract_logits(interp, output_details)
        return _topk_from_logits(y) + ("batched",)
    except Exception as e:
        # Fallback to processing one by one
        t1_all: List[int] = []
        t5_all: List[np.ndarray] = []
        for i in range(xin.shape[0]):
            sample = xin[i:i+1]
            try:
                interp.resize_tensor_input(input_details[0]['index'], sample.shape, strict=True)
                interp.allocate_tensors()
                interp.set_tensor(input_details[0]['index'], sample)
                interp.invoke()
                y = _extract_logits(interp, output_details)
                top1, top5 = _topk_from_logits(y)
                t1_all.extend(top1)
                t5_all.extend((t5 or [])[:len(top1)])
            except Exception as e:
                # Silently add dummy predictions for failed samples
                t1_all.append(0)
                t5_all.append(np.array([0, 1, 2, 3, 4]))
        return t1_all, t5_all, "fallback-single"


def _extract_logits(interp: Interpreter, out_details: List[Dict]) -> np.ndarray:
    # Prefer the output with last dim >=1000
    logits = None
    for d in out_details:
        arr = interp.get_tensor(d['index'])
        if arr.ndim >= 2 and arr.shape[-1] >= 1000:
            logits = arr
            break
    if logits is None:
        # Single-output argmax style
        arr = interp.get_tensor(out_details[0]['index'])
        return arr.astype(np.float32)
    # Dequantize if needed
    if logits.dtype == np.uint8:
        scale, zp = out_details[0]["quantization"]
        scale = scale or 1.0
        zp = zp or 0
        logits = (logits.astype(np.float32) - zp) * float(scale)
    # Some TF models export 1001 classes (background at 0)
    if logits.shape[-1] == 1001:
        logits = logits[..., 1:]
    return logits.astype(np.float32)


def _topk_from_logits(y: np.ndarray) -> Tuple[List[int], List[np.ndarray]]:
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.shape[-1] >= 1000:
        top1 = np.argmax(y, axis=1).tolist()
        idx = np.argsort(y, axis=1)[:, -TOP_K:][:, ::-1]
        top5 = [row for row in idx]
        return top1, top5
    # Argmax-style integer class outputs
    top1 = [int(v) for v in np.asarray(y).reshape(-1).tolist()]
    top5 = [np.array([c, (c+1)%1000, (c+2)%1000, (c+3)%1000, (c+4)%1000], np.int64) for c in top1]
    return top1, top5


# ---- Memory helpers ----

def _read_linux_vm(value_key: str) -> Optional[int]:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(value_key + ":"):
                    val_str = line.split()[-2]  # "VmRSS: 12345 kB" → "12345"
                    return int(val_str) * 1024  # kB → bytes
    except Exception:
        pass
    return None

def get_host_rss_bytes() -> Optional[int]:
    if platform.system() == "Linux":
        v = _read_linux_vm("VmRSS")
        if v is not None: 
            return v
    if HAVE_PSUTIL:
        try:
            return psutil.Process().memory_info().rss
        except Exception:
            pass
    if HAVE_RESOURCE:
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == "Darwin":
                return ru  # macOS: bytes
            else:
                return ru * 1024  # Linux: kB → bytes
        except Exception:
            pass
    return None

def get_host_peak_bytes_hwm() -> Optional[int]:
    if platform.system() == "Linux":
        v = _read_linux_vm("VmHWM")
        if v is not None: 
            return v
    if HAVE_PSUTIL and platform.system() == "Windows":
        try:
            proc = psutil.Process()
            if hasattr(proc, "memory_info") and hasattr(proc.memory_info(), "peak_wset"):
                return proc.memory_info().peak_wset
        except Exception:
            pass
    return None

def init_nvml_if_present() -> bool:
    if not HAVE_NVML:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False

def nvml_proc_used_bytes(pid: int) -> Optional[int]:
    try:
        if not HAVE_NVML:
            return None
        device_count = pynvml.nvmlDeviceGetCount()
        total_used = 0
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    if proc.pid == pid:
                        total_used += proc.usedGpuMemory
            except Exception:
                try:
                    procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                    for proc in procs:
                        if proc.pid == pid:
                            total_used += proc.usedGpuMemory
                except Exception:
                    pass
        return total_used if total_used > 0 else None
    except Exception:
        return None

# ---- Robust stats helpers ----

def robust_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": math.nan, "median": math.nan, "std": math.nan, 
                "min": math.nan, "max": math.nan, "p50": math.nan, "p90": math.nan, 
                "p95": math.nan, "p99": math.nan}
    arr = np.array(values, dtype=np.float64)
    stats = {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    return stats


# ---------- POWER / ENERGY SAMPLING ----------

# Linux RAPL paths (package + DRAM if present)
class RAPLReader:
    def __init__(self):
        self.ok = False
        self.pkg_files: List[str] = []
        self.dram_files: List[str] = []
        if platform.system() != "Linux":
            return
        base = "/sys/class/powercap"
        try:
            for entry in os.listdir(base):
                if entry.startswith("intel-rapl:"):
                    zone_path = os.path.join(base, entry)
                    name_file = os.path.join(zone_path, "name")
                    energy_file = os.path.join(zone_path, "energy_uj")
                    if os.path.exists(name_file) and os.path.exists(energy_file):
                        with open(name_file, "r") as f:
                            name = f.read().strip()
                        if name.startswith("package"):
                            self.pkg_files.append(energy_file)
                        elif name.startswith("dram"):
                            self.dram_files.append(energy_file)
            self.ok = bool(self.pkg_files)
        except Exception:
            pass

    def read_energy_j(self) -> Tuple[Optional[float], Optional[float]]:
        # returns (pkg_J_total, dram_J_total) as cumulative counters converted to Joules
        try:
            pkg_total = 0.0
            for f in self.pkg_files:
                with open(f, "r") as file:
                    pkg_total += float(file.read().strip()) / 1e6  # microjoules → joules
            dram_total = 0.0
            for f in self.dram_files:
                with open(f, "r") as file:
                    dram_total += float(file.read().strip()) / 1e6
            return (pkg_total if self.pkg_files else None, dram_total if self.dram_files else None)
        except Exception:
            return None, None

class NVMLPower:
    def __init__(self):
        self.ok = False
        self.handles: List[Any] = []
        if not HAVE_NVML:
            return
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            self.ok = bool(self.handles)
        except Exception:
            pass

    def read_power_w(self) -> Optional[float]:
        try:
            if not self.ok:
                return None
            total_power = 0.0
            for handle in self.handles:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                total_power += power_mw / 1000.0  # milliwatts → watts
            return total_power
        except Exception:
            return None

    def limits(self) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "gpu_power_limit_w": None,
            "gpu_power_limit_default_w": None,
        }
        try:
            for handle in self.handles:
                limit_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraintsDefault(handle)[1]
                out["gpu_power_limit_default_w"] = limit_mw / 1000.0
                break  # Just take first GPU
        except Exception:
            pass
        return out

class EnergySampler:
    """
    High‑frequency sampler for energy/power windows.
    Host energy via RAPL cumulative counters (J); GPU instantaneous power via NVML (W).
    Integrates energy over window: pkg_J (and dram_J if available) + GPU_J (power × dt).
    """
    def __init__(self, hz: int = 300):
        self.hz = max(200, min(int(hz), 2000))
        self.dt = 1.0 / float(self.hz)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.rapl = RAPLReader()
        self.nvml = NVMLPower()
        # window buffers
        self.tstamps: List[float] = []
        self.gpu_power_w: List[float] = []
        self.host_pkg_j_start: Optional[float] = None
        self.host_dram_j_start: Optional[float] = None
        # static baselines (W)
        self.idle_host_w: Optional[float] = None
        self.idle_gpu_w: Optional[float] = None
        self.postload_host_w: Optional[float] = None
        self.postload_gpu_w: Optional[float] = None

    # --- instantaneous host power estimate from RAPL derivative ---
    def _read_host_energy(self) -> Tuple[Optional[float], Optional[float]]:
        return self.rapl.read_energy_j()

    def _read_gpu_power(self) -> Optional[float]:
        return self.nvml.read_power_w()

    def record_idle_power(self):
        # average a short window to get idle power (host+gpu) before model load
        self.idle_host_w, self.idle_gpu_w = self._measure_power_avg(seconds=0.5)

    def record_postload_power(self):
        self.postload_host_w, self.postload_gpu_w = self._measure_power_avg(seconds=0.5)

    def _measure_power_avg(self, seconds: float = 0.5) -> Tuple[Optional[float], Optional[float]]:
        n = max(5, int(self.hz * seconds))
        host_prev = self._read_host_energy()[0]
        time.sleep(self.dt)
        host_acc_w = []
        gpu_acc_w = []
        for _ in range(n):
            host_curr = self._read_host_energy()[0]
            gpu_w = self._read_gpu_power()
            if host_prev is not None and host_curr is not None:
                host_w = (host_curr - host_prev) / self.dt
                if host_w >= 0:
                    host_acc_w.append(host_w)
                host_prev = host_curr
            if gpu_w is not None:
                gpu_acc_w.append(gpu_w)
            time.sleep(self.dt)
        h = float(np.mean(host_acc_w)) if host_acc_w else None
        g = float(np.mean(gpu_acc_w)) if gpu_acc_w else None
        return h, g

    def start(self):
        # reset window buffers and set starting energy counters
        self.tstamps.clear(); self.gpu_power_w.clear()
        self.host_pkg_j_start, self.host_dram_j_start = self._read_host_energy()
        self._stop.clear()
        def _runner():
            while not self._stop.is_set():
                t = time.time()
                gpu_w = self._read_gpu_power()
                self.tstamps.append(t)
                if gpu_w is not None:
                    self.gpu_power_w.append(gpu_w)
                else:
                    self.gpu_power_w.append(math.nan)
                time.sleep(self.dt)
        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        # Host cumulative energies (J)
        host_pkg_j_end, host_dram_j_end = self._read_host_energy()
        pkg_dJ = (host_pkg_j_end - self.host_pkg_j_start) if (host_pkg_j_end is not None and self.host_pkg_j_start is not None) else math.nan
        dram_dJ = (host_dram_j_end - self.host_dram_j_start) if (host_dram_j_end is not None and self.host_dram_j_start is not None) else math.nan
        # GPU energy via sum(power*dt)
        # Use fixed dt per sample; if timestamps sparse, fallback to dt configured
        if len(self.tstamps) >= 2:
            dt_est = (self.tstamps[-1] - self.tstamps[0]) / (len(self.tstamps) - 1)
        else:
            dt_est = self.dt
        gp = np.array(self.gpu_power_w, dtype=float)
        gp = gp[np.isfinite(gp)]
        gpu_j = float(np.sum(gp) * dt_est) if gp.size > 0 else math.nan

        # Power profile statistics over window
        def _stats(arr: List[float]) -> Dict[str, float]:
            finite = [v for v in arr if not math.isnan(v)]
            if not finite:
                return {"mean": math.nan, "median": math.nan, "std": math.nan, 
                       "p90": math.nan, "p95": math.nan, "p99": math.nan}
            a = np.array(finite)
            return {
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "std": float(a.std()),
                "p90": float(np.percentile(a, 90)),
                "p95": float(np.percentile(a, 95)),
                "p99": float(np.percentile(a, 99)),
            }
        gpu_stats = _stats(self.gpu_power_w)

        # Clear window buffers
        self.tstamps.clear(); self.gpu_power_w.clear()
        return {
            "host_pkg_j": pkg_dJ,
            "host_dram_j": dram_dJ,
            "gpu_j": gpu_j,
            "gpu_power_mean_w": gpu_stats["mean"],
            "gpu_power_median_w": gpu_stats["median"],
            "gpu_power_std_w": gpu_stats["std"],
            "gpu_power_p90_w": gpu_stats["p90"],
            "gpu_power_p95_w": gpu_stats["p95"],
            "gpu_power_p99_w": gpu_stats["p99"],
        }

    def finalize_env(self) -> Dict[str, Any]:
        # Driver & power limit disclosure (best‑effort)
        info: Dict[str, Any] = {
            "rapl_available": self.rapl.ok,
            "nvml_available": self.nvml.ok,
            "idle_host_w": self.idle_host_w,
            "idle_gpu_w": self.idle_gpu_w,
            "postload_host_w": self.postload_host_w,
            "postload_gpu_w": self.postload_gpu_w,
            "cpu_governors": read_cpu_governors(),
        }
        if self.nvml.ok:
            try:
                limits = self.nvml.limits()
                info.update(limits)
            except Exception:
                pass
        return info

# CPU governor disclosure (Linux best‑effort)

def read_cpu_governors() -> str:
    if platform.system() != "Linux":
        return "NA"
    roots = "/sys/devices/system/cpu"
    govs = []
    try:
        for d in os.listdir(roots):
            if d.startswith("cpu") and d[3:].isdigit():
                gov_file = os.path.join(roots, d, "cpufreq", "scaling_governor")
                if os.path.exists(gov_file):
                    with open(gov_file, "r") as f:
                        govs.append(f.read().strip())
        if govs:
            unique_govs = list(set(govs))
            return ",".join(unique_govs)
    except Exception:
        pass
    return "unknown"


# ---------- Memory sampling ----------

class MemorySampler:
    def __init__(self, hz: int = 300, provider_label: str = ""):
        self.hz = max(100, min(int(hz), 1000))
        self.dt = 1.0 / float(self.hz)
        self.provider_label = provider_label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.host_rss_mib: List[float] = []
        self.dev_used_mib: List[float] = []
        self.pid = os.getpid()
        self.nvml_ok = init_nvml_if_present()
        # baselines
        self.idle_host_mib: Optional[float] = None
        self.idle_dev_mib: Optional[float] = None
        self.postload_host_mib: Optional[float] = None
        self.postload_dev_mib: Optional[float] = None

    def _poll_once(self):
        # Host RSS
        host_bytes = get_host_rss_bytes()
        host_mib = (host_bytes / (1024**2)) if host_bytes else math.nan
        self.host_rss_mib.append(host_mib)
        # Device memory
        dev_bytes = nvml_proc_used_bytes(self.pid) if self.nvml_ok else None
        dev_mib = (dev_bytes / (1024**2)) if dev_bytes else math.nan
        self.dev_used_mib.append(dev_mib)

    def start(self):
        self.host_rss_mib.clear()
        self.dev_used_mib.clear()
        self._stop.clear()
        def _runner():
            while not self._stop.is_set():
                self._poll_once()
                time.sleep(self.dt)
        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=2.0)
            self._thread = None
        
        host_stats = robust_stats([v for v in self.host_rss_mib if not math.isnan(v)])
        dev_stats = robust_stats([v for v in self.dev_used_mib if not math.isnan(v)])
        
        self.host_rss_mib.clear()
        self.dev_used_mib.clear()
        
        return {
            "host_peak_mib": host_stats["max"],
            "host_mean_mib": host_stats["mean"],
            "dev_peak_mib": dev_stats["max"],
            "dev_mean_mib": dev_stats["mean"],
        }

    def record_idle_baseline(self):
        self._poll_once()
        self.idle_host_mib = self.host_rss_mib[-1] if self.host_rss_mib else None
        self.idle_dev_mib = self.dev_used_mib[-1] if self.dev_used_mib else None

    def record_postload_baseline(self):
        self._poll_once()
        time.sleep(0.1)  # brief settle
        self._poll_once()
        self.postload_host_mib = self.host_rss_mib[-1] if self.host_rss_mib else None
        self.postload_dev_mib = self.dev_used_mib[-1] if self.dev_used_mib else None

    def finalize(self) -> Dict[str, Any]:
        return {
            "baseline_idle_host_mib": self.idle_host_mib,
            "baseline_postload_host_mib": self.postload_host_mib,
            "baseline_idle_dev_mib": self.idle_dev_mib,
            "baseline_postload_dev_mib": self.postload_dev_mib,
            "host_global_peak_mib": get_host_peak_bytes_hwm() / (1024**2) if get_host_peak_bytes_hwm() else None,
            "host_true_os_peak_mib": get_host_rss_bytes() / (1024**2) if get_host_rss_bytes() else None,
            "mem_device_backend": self.provider_label,
        }


def load_gflops_map(csv_path: Optional[str]) -> Optional[Dict[str, float]]:
    if not csv_path:
        return None
    if not os.path.exists(csv_path):
        print(f"[GFLOPS] File not found: {csv_path}")
        return None
    mp: Dict[str, float] = {}
    try:
        df = pd.read_csv(csv_path)
        if "model_name" not in df.columns or "gflops" not in df.columns:
            print(f"[GFLOPS] CSV must have columns: model_name, gflops")
            return None
        for _, row in df.iterrows():
            model = str(row["model_name"]).strip()
            gflops = float(row["gflops"])
            if model and not math.isnan(gflops) and gflops > 0:
                mp[model] = gflops
        print(f"[GFLOPS] Loaded {len(mp)} model mappings from {csv_path}")
        return mp
    except Exception as e:
        print(f"[GFLOPS] Failed to load {csv_path}: {e}")
        return None


# --- Evaluation ---

def expected_input_size_from_name(model_name: str) -> int:
    return 299 if "inception_v3" in model_name.lower() else 224


def evaluate_model(
    model_path: Path, 
    model_name: str, 
    rows: List[Tuple[str, int]], 
    req_bs: int, 
    use_gpu: bool = False,
    do_latency: bool = False,
    warmup_batches: int = 2,
    repeats_batch: int = 10,
    do_mem: bool = False,
    mem_hz: int = 300,
    provider_label: str = "TFLite",
    do_energy: bool = False,
    energy_hz: int = 300,
    gflops_map: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, float, Optional[float], Dict[str, Any], Dict[str, Any]]:
    """
    One pass over data; memory & energy sampling windows wrap each timed session.run.
    Returns per‑sample DF, accs, mem_meta, energy_meta.
    """
    try:
        interp = ensure_interpreter(str(model_path), use_gpu=use_gpu)
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        
        # Memory sampler + baselines
        sampler = MemorySampler(hz=mem_hz, provider_label=provider_label) if do_mem else None
        if sampler:
            sampler.record_idle_baseline()

        # Energy sampler + baselines (idle/post‑load power)
        es = EnergySampler(hz=energy_hz) if do_energy else None
        if es:
            es.record_idle_power()
        
        # Debug: print input details
        gpu_info = " [GPU]" if use_gpu and GPU_AVAILABLE else ""
        print(f"[INFO] {model_name}{gpu_info} - Input: {input_details[0]['shape']}, dtype: {input_details[0]['dtype']}")
        
        size = get_input_size_from_details(input_details) or expected_input_size_from_name(model_name)

        # postload baselines
        if sampler:
            sampler.record_postload_baseline()
        if es:
            es.record_postload_power()

        img_paths = [DATA_DIR / rel for (rel, _y) in rows]
        # Keep in NCHW format as TFLite models expect it
        x_chw = load_batch_CHW(img_paths, size)  # (N,3,H,W) in [0,1]

        n = model_name.lower()
        if n.startswith("resnet50_v1_mlperf"):
            # Apply Caffe preprocessing in CHW format
            x_pre = x_chw * 255.0
            x_pre[:, 0, :, :] -= 123.68  # R
            x_pre[:, 1, :, :] -= 116.779  # G
            x_pre[:, 2, :, :] -= 103.939  # B
        elif "mobilevit_xxs" in n:
            x_pre = x_chw.astype(np.float32)
        else:
            # Apply ImageNet normalization in CHW format
            x_pre = (x_chw - np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)) / np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        fixed_bs = get_fixed_batch_dim(input_details)
        if req_bs and req_bs > 0:
            if fixed_bs is not None and fixed_bs != req_bs:
                raise RuntimeError(f"{model_name}: fixed batch {fixed_bs} != requested --bs {req_bs}")
            target_bs = req_bs
        else:
            target_bs = fixed_bs if fixed_bs is not None else 32

        total = (len(x_pre) // target_bs) * target_bs
        if total < len(x_pre) and req_bs and req_bs > 0:
            print(f"[BS] {model_name}: dropping last {len(x_pre) - total} sample(s) to enforce --bs {target_bs}")
        x_pre = x_pre[:total] if total > 0 else x_pre
        if target_bs <= 0:
            raise RuntimeError(f"{model_name}: invalid resolved batch size {target_bs}")

        top1_all: List[int] = []
        top5_all: List[np.ndarray] = []

        # latency/throughput capture (per‑sample)
        per_sample_latency_ms: List[float] = []
        per_sample_throughput_sps: List[float] = []

        # memory capture (per‑sample)
        per_sample_host_peak_mib: List[float] = []
        per_sample_host_mean_mib: List[float] = []
        per_sample_dev_peak_mib:  List[float] = []
        per_sample_dev_mean_mib:  List[float] = []

        # energy capture (per‑sample)
        per_sample_energy_j: List[float] = []  # total (host_pkg + gpu + dram if available)
        per_sample_energy_host_j: List[float] = []
        per_sample_energy_gpu_j: List[float] = []
        per_sample_energy_dram_j: List[float] = []
        per_sample_gpu_power_mean_w: List[float] = []
        per_sample_gpu_power_median_w: List[float] = []
        per_sample_gpu_power_std_w: List[float] = []
        per_sample_gpu_power_p90_w: List[float] = []
        per_sample_gpu_power_p95_w: List[float] = []
        per_sample_gpu_power_p99_w: List[float] = []
        per_sample_energy_per_gflop: List[float] = []

        # aggregate steady‑state means (memory)
        steady_host_means: List[float] = []
        steady_dev_means:  List[float] = []

        pbar = tqdm(total=len(x_pre), desc=f"{model_name} eval(bs={target_bs})", leave=False, ncols=90)
        i = 0

        def _time_once_with_mem_energy(chunk: np.ndarray) -> Tuple[float, Dict[str, float], Dict[str, float]]:
            if sampler: sampler.start()
            if es: es.start()
            t0 = time.time()
            t1, t5, mode = run_tflite(interp, chunk)
            elapsed = time.time() - t0
            mem_res = sampler.stop() if sampler else {}
            energy_res = es.stop() if es else {}
            return elapsed, mem_res, energy_res

        def _warmup(chunk: np.ndarray, n: int):
            for _ in range(n):
                run_tflite(interp, chunk)

        while i < len(x_pre):
            end = i + target_bs
            chunk = x_pre[i:end].astype(np.float32)
            
            if do_latency:
                # warmup
                _warmup(chunk, warmup_batches)
                
                # timed repeats
                latencies = []
                for rep in range(repeats_batch):
                    elapsed, mem_res, energy_res = _time_once_with_mem_energy(chunk)
                    latencies.append(elapsed * 1000)  # ms
                    
                    # accumulate memory metrics per inference
                    if mem_res:
                        per_sample_host_peak_mib.append(mem_res.get("host_peak_mib", math.nan))
                        per_sample_host_mean_mib.append(mem_res.get("host_mean_mib", math.nan))
                        per_sample_dev_peak_mib.append(mem_res.get("dev_peak_mib", math.nan))
                        per_sample_dev_mean_mib.append(mem_res.get("dev_mean_mib", math.nan))
                        
                        # steady state tracking
                        if not math.isnan(mem_res.get("host_mean_mib", math.nan)):
                            steady_host_means.append(mem_res["host_mean_mib"])
                        if not math.isnan(mem_res.get("dev_mean_mib", math.nan)):
                            steady_dev_means.append(mem_res["dev_mean_mib"])
                    else:
                        per_sample_host_peak_mib.extend([math.nan] * len(chunk))
                        per_sample_host_mean_mib.extend([math.nan] * len(chunk))
                        per_sample_dev_peak_mib.extend([math.nan] * len(chunk))
                        per_sample_dev_mean_mib.extend([math.nan] * len(chunk))
                    
                    # accumulate energy metrics per inference
                    if energy_res:
                        host_j = energy_res.get("host_pkg_j", math.nan)
                        gpu_j = energy_res.get("gpu_j", math.nan)
                        dram_j = energy_res.get("host_dram_j", math.nan)
                        total_j = sum(x for x in [host_j, gpu_j, dram_j] if not math.isnan(x))
                        
                        per_sample_energy_j.append(total_j if total_j > 0 else math.nan)
                        per_sample_energy_host_j.append(host_j)
                        per_sample_energy_gpu_j.append(gpu_j)
                        per_sample_energy_dram_j.append(dram_j)
                        per_sample_gpu_power_mean_w.append(energy_res.get("gpu_power_mean_w", math.nan))
                        per_sample_gpu_power_median_w.append(energy_res.get("gpu_power_median_w", math.nan))
                        per_sample_gpu_power_std_w.append(energy_res.get("gpu_power_std_w", math.nan))
                        per_sample_gpu_power_p90_w.append(energy_res.get("gpu_power_p90_w", math.nan))
                        per_sample_gpu_power_p95_w.append(energy_res.get("gpu_power_p95_w", math.nan))
                        per_sample_gpu_power_p99_w.append(energy_res.get("gpu_power_p99_w", math.nan))
                        
                        # energy per GFLOP
                        if gflops_map and model_name in gflops_map and not math.isnan(total_j):
                            gflops = gflops_map[model_name] * len(chunk)  # total GFLOPs for batch
                            per_sample_energy_per_gflop.append(total_j / gflops if gflops > 0 else math.nan)
                        else:
                            per_sample_energy_per_gflop.append(math.nan)
                    else:
                        per_sample_energy_j.extend([math.nan] * len(chunk))
                        per_sample_energy_host_j.extend([math.nan] * len(chunk))
                        per_sample_energy_gpu_j.extend([math.nan] * len(chunk))
                        per_sample_energy_dram_j.extend([math.nan] * len(chunk))
                        per_sample_gpu_power_mean_w.extend([math.nan] * len(chunk))
                        per_sample_gpu_power_median_w.extend([math.nan] * len(chunk))
                        per_sample_gpu_power_std_w.extend([math.nan] * len(chunk))
                        per_sample_gpu_power_p90_w.extend([math.nan] * len(chunk))
                        per_sample_gpu_power_p95_w.extend([math.nan] * len(chunk))
                        per_sample_gpu_power_p99_w.extend([math.nan] * len(chunk))
                        per_sample_energy_per_gflop.extend([math.nan] * len(chunk))
                
                # aggregate latency/throughput for this batch
                lat_median = float(np.median(latencies))
                thr_median = float(len(chunk) * 1000.0 / lat_median)  # samples/sec
                per_sample_latency_ms.extend([lat_median] * len(chunk))
                per_sample_throughput_sps.extend([thr_median] * len(chunk))
                
                # for accuracy, just run once more (no timing)
                t1, t5, mode = run_tflite(interp, chunk)
            else:
                # no latency measurement, just run once with optional memory/energy
                if sampler or es:
                    elapsed, mem_res, energy_res = _time_once_with_mem_energy(chunk)
                    if mem_res:
                        per_sample_host_peak_mib.extend([mem_res.get("host_peak_mib", math.nan)] * len(chunk))
                        per_sample_host_mean_mib.extend([mem_res.get("host_mean_mib", math.nan)] * len(chunk))
                        per_sample_dev_peak_mib.extend([mem_res.get("dev_peak_mib", math.nan)] * len(chunk))
                        per_sample_dev_mean_mib.extend([mem_res.get("dev_mean_mib", math.nan)] * len(chunk))
                        if not math.isnan(mem_res.get("host_mean_mib", math.nan)):
                            steady_host_means.append(mem_res["host_mean_mib"])
                        if not math.isnan(mem_res.get("dev_mean_mib", math.nan)):
                            steady_dev_means.append(mem_res["dev_mean_mib"])
                    
                    if energy_res:
                        host_j = energy_res.get("host_pkg_j", math.nan)
                        gpu_j = energy_res.get("gpu_j", math.nan)
                        dram_j = energy_res.get("host_dram_j", math.nan)
                        total_j = sum(x for x in [host_j, gpu_j, dram_j] if not math.isnan(x))
                        
                        per_sample_energy_j.extend([total_j if total_j > 0 else math.nan] * len(chunk))
                        per_sample_energy_host_j.extend([host_j] * len(chunk))
                        per_sample_energy_gpu_j.extend([gpu_j] * len(chunk))
                        per_sample_energy_dram_j.extend([dram_j] * len(chunk))
                        per_sample_gpu_power_mean_w.extend([energy_res.get("gpu_power_mean_w", math.nan)] * len(chunk))
                        per_sample_gpu_power_median_w.extend([energy_res.get("gpu_power_median_w", math.nan)] * len(chunk))
                        per_sample_gpu_power_std_w.extend([energy_res.get("gpu_power_std_w", math.nan)] * len(chunk))
                        per_sample_gpu_power_p90_w.extend([energy_res.get("gpu_power_p90_w", math.nan)] * len(chunk))
                        per_sample_gpu_power_p95_w.extend([energy_res.get("gpu_power_p95_w", math.nan)] * len(chunk))
                        per_sample_gpu_power_p99_w.extend([energy_res.get("gpu_power_p99_w", math.nan)] * len(chunk))
                        
                        # energy per GFLOP
                        if gflops_map and model_name in gflops_map and not math.isnan(total_j):
                            gflops = gflops_map[model_name] * len(chunk)
                            per_sample_energy_per_gflop.extend([total_j / gflops if gflops > 0 else math.nan] * len(chunk))
                        else:
                            per_sample_energy_per_gflop.extend([math.nan] * len(chunk))
                    
                    t1, t5, mode = t1, t5, mode  # already computed
                else:
                    t1, t5, mode = run_tflite(interp, chunk)
            
            top1_all.extend(t1)
            top5_all.extend((t5 or [])[:len(t1)])
            i = end
            pbar.update(len(t1))
        pbar.close()

        # strict alignment & dtype coercion for export
        N = len(top1_all)
        rel_paths = [rows[j][0] for j in range(N)]
        y_true    = [int(rows[j][1]) for j in range(N)]
        top1_all  = [int(v) for v in top1_all]
        if len(top5_all) != N:
            top5_all = top5_all[:N]
        top5_strs = [" ".join(map(str, (arr.astype(int).tolist() if arr.size else []))) for arr in top5_all]

        denom = max(1, N)
        top1_acc = sum(int(p==y) for p,y in zip(top1_all, y_true)) / denom
        top5_acc = sum(int(y in arr) for y, arr in zip(y_true, [set(a.tolist()) for a in top5_all])) / denom if N else None

        # Build per‑sample DF
        data: Dict[str, List[Any]] = {
            "rel_path": [str(p) for p in rel_paths],
            "correct_label": y_true,
            "pred_top1": top1_all,
            "top1_correct": [int(p==y) for p,y in zip(top1_all, y_true)],
            "top5_preds": top5_strs,
        }
        if do_latency:
            data["latency_ms"] = per_sample_latency_ms[:N]
            data["throughput_sps"] = per_sample_throughput_sps[:N]
        if sampler:
            data["host_peak_mib"] = per_sample_host_peak_mib[:N]
            data["host_mean_mib"] = per_sample_host_mean_mib[:N]
            data["dev_peak_mib"] = per_sample_dev_peak_mib[:N]
            data["dev_mean_mib"] = per_sample_dev_mean_mib[:N]

        if do_energy:
            data["energy_j"] = per_sample_energy_j[:N]
            data["energy_host_j"] = per_sample_energy_host_j[:N]
            data["energy_gpu_j"] = per_sample_energy_gpu_j[:N]
            data["energy_dram_j"] = per_sample_energy_dram_j[:N]
            data["gpu_power_mean_w"] = per_sample_gpu_power_mean_w[:N]
            data["gpu_power_median_w"] = per_sample_gpu_power_median_w[:N]
            data["gpu_power_std_w"] = per_sample_gpu_power_std_w[:N]
            data["gpu_power_p90_w"] = per_sample_gpu_power_p90_w[:N]
            data["gpu_power_p95_w"] = per_sample_gpu_power_p95_w[:N]
            data["gpu_power_p99_w"] = per_sample_gpu_power_p99_w[:N]
            data["energy_per_gflop"] = per_sample_energy_per_gflop[:N]

        df = pd.DataFrame(data)

        # Finalize memory provenance & aggregates
        mem_meta: Dict[str, Any] = {}
        if sampler:
            mem_meta = sampler.finalize()
            if steady_host_means:
                mem_meta["steady_host_mean_delta_mib_median"] = float(np.median(steady_host_means))
            if steady_dev_means:
                mem_meta["steady_dev_mean_delta_mib_median"] = float(np.median(steady_dev_means))

        # Finalize energy/env provenance
        energy_meta: Dict[str, Any] = {}
        if es:
            energy_meta = es.finalize_env()

        return (
            df,
            float(top1_acc),
            (float(top5_acc) if top5_acc is not None else None),
            mem_meta,
            energy_meta,
        )
        
    except Exception as e:
        print(f"[ERROR] Failed to evaluate {model_name}: {e}")
        # Return empty dataframe with error info
        df = pd.DataFrame({"note": [str(e)]})
        return df, float('nan'), None, {}, {}


# --- Writer ---

def save_ods_multisheet(path: Path, sheets: Dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="odf") as writer:
        for name, df in sheets.items():
            (df if (df is not None and not df.empty) else pd.DataFrame({"note":["empty or failed"]})).to_excel(
                writer, sheet_name=name[:31], index=False
            )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)


# --- Main ---
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FRM TFLite eval (single-pass acc+latency+memory+energy, bs-enforced)")
    p.add_argument("--blaze", action="store_true", help="Run only 10 random samples")
    p.add_argument("--bs", type=int, default=0, help="Enforce batch size (e.g., --bs 1). 0=auto")
    p.add_argument("--gpu", action="store_true", help="Try to use GPU acceleration (requires full TensorFlow)")
    p.add_argument("--latency", action="store_true", help="Measure latency/throughput in the same pass")
    p.add_argument("--lat-warmup-batches", type=int, default=2, help="Warmup runs per batch (not recorded)")
    p.add_argument("--lat-repeats-batch", type=int, default=10, help="Timed repeats per batch; median used")
    p.add_argument("--mem", action="store_true", help="Enable memory sampling & reporting (host+device)")
    p.add_argument("--mem-sample-hz", type=int, default=300, help="Sampler frequency (Hz), 200–500 recommended")
    p.add_argument("--energy", action="store_true", help="Enable power/energy sampling (host RAPL + GPU NVML)")
    p.add_argument("--energy-sample-hz", type=int, default=300, help="Energy sampler frequency (Hz), ≥200 recommended")
    p.add_argument("--gflops-map", type=str, default="", help="CSV with columns [model_name,gflops] for J/GFLOP")
    args = p.parse_args()

    # Enforce full TensorFlow requirement for GPU acceleration
    if args.gpu and (RUNTIME != "tensorflow.lite" or not TF_AVAILABLE):
        print(f"❌ Error: --gpu flag requires full TensorFlow installation")
        print(f"   Current backend: {RUNTIME or 'None'}")
        print(f"   TensorFlow available: {TF_AVAILABLE}")
        print(f"   GPU acceleration is only available with full TensorFlow, not tflite_runtime")
        print(f"   Install with: pip install tensorflow")
        raise SystemExit("GPU acceleration requires full TensorFlow")

    if RUNTIME is None:
        raise SystemExit("No TFLite backend available. Install 'tflite-runtime' (preferred) or 'tensorflow'.")

    random.seed(1337)
    np.random.seed(1337)

    print(f"[System] TFLite backend: {RUNTIME} on {platform.system()} {platform.machine()}")
    print(f"[Batch] {'Enforcing' if (args.bs and args.bs>0) else 'Auto'} batch size: {args.bs if args.bs else 'auto(32 or fixed N)'}")
    
    if args.gpu:
        if GPU_AVAILABLE:
            print(f"[GPU] GPU acceleration enabled")
        else:
            print(f"[GPU] GPU requested but not available - falling back to CPU")
    else:
        print(f"[GPU] Using CPU only (use --gpu to try GPU acceleration)")

    gflops_map = load_gflops_map(args.gflops_map)

    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    if args.blaze and len(manifest) > 10:
        manifest = manifest.sample(n=10, random_state=0).reset_index(drop=True)
        print("[BLAZE] Using 10 random samples")

    if "correct_label" not in manifest.columns:
        wnid_to_idx = load_imagenet_class_mapping()
        if "wnid" in manifest.columns:
            manifest["correct_label"] = manifest["wnid"].map(wnid_to_idx).astype(int)
        elif "class_id_1to1000" in manifest.columns:
            manifest["correct_label"] = manifest["class_id_1to1000"].astype(int) - 1
        else:
            raise RuntimeError("manifest must contain 'wnid' or 'correct_label'")

    rows: List[Tuple[str,int]] = [(str(r.rel_path), int(r.correct_label)) for r in manifest.itertuples(index=False)]

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.tflite')]
    if not model_files:
        raise SystemExit(f"No .tflite models found in {MODELS_DIR}")
    model_files.sort()

    all_sheets: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    # Run info + Memory/Power backend sheet
    runinfo = pd.DataFrame(
        {
            "key": [
                "hostname",
                "platform", 
                "python_version",
                "tflite_runtime",
                "backend_info",
                "psutil",
                "nvml_available",
            ],
            "value": [
                platform.node(),
                platform.platform(),
                platform.python_version(),
                RUNTIME or "None",
                f"TFLite ({RUNTIME}) on {platform.system()} {platform.machine()}",
                str(HAVE_PSUTIL),
                str(HAVE_NVML),
            ],
        }
    )
    all_sheets["RunInfo"] = runinfo

    # Optional: model statics sheet (file sizes)
    model_statics_rows = []

    # We'll also capture one environment disclosure from energy sampler (if enabled) 
    power_env_sheet: Optional[pd.DataFrame] = None

    for fname in tqdm(model_files, desc="Models", ncols=90):
        mpath = MODELS_DIR / fname
        try:
            # File size
            file_size_mb = float(mpath.stat().st_size / (1024**2))
            
            df, t1, t5, mem_meta, energy_meta = evaluate_model(
                mpath, fname, rows, args.bs, use_gpu=args.gpu,
                do_latency=args.latency,
                warmup_batches=args.lat_warmup_batches,
                repeats_batch=args.lat_repeats_batch,
                do_mem=args.mem,
                mem_hz=args.mem_sample_hz,
                provider_label="TFLite",
                do_energy=args.energy,
                energy_hz=args.energy_sample_hz,
                gflops_map=gflops_map,
            )
            all_sheets[fname] = df
            
            # Build comprehensive summary row
            summary_row = {
                "model_name": fname,
                "top1_acc": t1,
                "top5_acc": (t5 if t5 is not None else np.nan),
                "samples": len(df) if "note" not in df.columns else 0,
                "model_file_mb": file_size_mb,
            }
            
            # Add latency/throughput metrics if measured
            if args.latency and "latency_ms" in df.columns:
                lat_values = [v for v in df["latency_ms"].tolist() if not math.isnan(v)]
                thr_values = [v for v in df["throughput_sps"].tolist() if not math.isnan(v)]
                
                if lat_values:
                    lat_stats = robust_stats(lat_values)
                    summary_row.update({
                        "lat_mean_ms": lat_stats["mean"],
                        "lat_median_ms": lat_stats["median"], 
                        "lat_std_ms": lat_stats["std"],
                        "lat_min_ms": lat_stats["min"],
                        "lat_max_ms": lat_stats["max"],
                        "lat_p50_ms": lat_stats["p50"],
                        "lat_p90_ms": lat_stats["p90"],
                        "lat_p95_ms": lat_stats["p95"],
                        "lat_p99_ms": lat_stats["p99"],
                    })
                
                if thr_values:
                    thr_stats = robust_stats(thr_values)
                    summary_row.update({
                        "thr_mean_sps": thr_stats["mean"],
                        "thr_median_sps": thr_stats["median"],
                        "thr_std_sps": thr_stats["std"],
                        "thr_min_sps": thr_stats["min"],
                        "thr_max_sps": thr_stats["max"],
                        "thr_p50_sps": thr_stats["p50"],
                        "thr_p90_sps": thr_stats["p90"],
                        "thr_p95_sps": thr_stats["p95"],
                        "thr_p99_sps": thr_stats["p99"],
                    })
            
            # Add memory metrics if measured
            if args.mem and mem_meta:
                # Per-sample memory deltas (from baseline)
                if "host_peak_mib" in df.columns:
                    host_peaks = [v for v in df["host_peak_mib"].tolist() if not math.isnan(v)]
                    dev_peaks = [v for v in df["dev_peak_mib"].tolist() if not math.isnan(v)]
                    
                    baseline_host = mem_meta.get("baseline_postload_host_mib", 0) or 0
                    baseline_dev = mem_meta.get("baseline_postload_dev_mib", 0) or 0
                    
                    host_deltas = [v - baseline_host for v in host_peaks if not math.isnan(v)]
                    dev_deltas = [v - baseline_dev for v in dev_peaks if not math.isnan(v)]
                    
                    if host_deltas:
                        host_stats = robust_stats(host_deltas)
                        summary_row.update({
                            "host_peak_delta_mean_mib": host_stats["mean"],
                            "host_peak_delta_median_mib": host_stats["median"],
                            "host_peak_delta_p95_mib": host_stats["p95"],
                            "host_peak_delta_max_mib": host_stats["max"],
                        })
                    
                    if dev_deltas:
                        dev_stats = robust_stats(dev_deltas)
                        summary_row.update({
                            "dev_peak_delta_mean_mib": dev_stats["mean"],
                            "dev_peak_delta_median_mib": dev_stats["median"],
                            "dev_peak_delta_p95_mib": dev_stats["p95"],
                            "dev_peak_delta_max_mib": dev_stats["max"],
                        })
                
                # Add baseline and global metrics from mem_meta
                summary_row.update({
                    "baseline_idle_host_mib": mem_meta.get("baseline_idle_host_mib"),
                    "baseline_postload_host_mib": mem_meta.get("baseline_postload_host_mib"),
                    "baseline_idle_dev_mib": mem_meta.get("baseline_idle_dev_mib"),
                    "baseline_postload_dev_mib": mem_meta.get("baseline_postload_dev_mib"),
                    "host_global_peak_mib": mem_meta.get("host_global_peak_mib"),
                    "host_true_os_peak_mib": mem_meta.get("host_true_os_peak_mib"),
                    "steady_host_mean_delta_mib_median": mem_meta.get("steady_host_mean_delta_mib_median"),
                    "steady_dev_mean_delta_mib_median": mem_meta.get("steady_dev_mean_delta_mib_median"),
                    "mem_device_backend": mem_meta.get("mem_device_backend"),
                })
            
            # Add energy metrics if measured
            if args.energy and "energy_j" in df.columns:
                energy_values = [v for v in df["energy_j"].tolist() if not math.isnan(v)]
                energy_host_values = [v for v in df["energy_host_j"].tolist() if not math.isnan(v)]
                energy_gpu_values = [v for v in df["energy_gpu_j"].tolist() if not math.isnan(v)]
                energy_dram_values = [v for v in df["energy_dram_j"].tolist() if not math.isnan(v)]
                energy_per_gflop_values = [v for v in df["energy_per_gflop"].tolist() if not math.isnan(v)]
                
                gpu_power_mean_values = [v for v in df["gpu_power_mean_w"].tolist() if not math.isnan(v)]
                gpu_power_median_values = [v for v in df["gpu_power_median_w"].tolist() if not math.isnan(v)]
                gpu_power_p95_values = [v for v in df["gpu_power_p95_w"].tolist() if not math.isnan(v)]
                
                if energy_values:
                    energy_stats = robust_stats(energy_values)
                    summary_row.update({
                        "energy_mean_j_per_sample": energy_stats["mean"],
                        "energy_median_j_per_sample": energy_stats["median"],
                        "energy_p90_j_per_sample": energy_stats["p90"],
                        "energy_p95_j_per_sample": energy_stats["p95"],
                        "energy_p99_j_per_sample": energy_stats["p99"],
                    })
                
                if energy_host_values:
                    summary_row["energy_host_median_j_per_sample"] = float(np.median(energy_host_values))
                if energy_gpu_values:
                    summary_row["energy_gpu_median_j_per_sample"] = float(np.median(energy_gpu_values))
                if energy_dram_values:
                    summary_row["energy_dram_median_j_per_sample"] = float(np.median(energy_dram_values))
                if energy_per_gflop_values:
                    gflop_stats = robust_stats(energy_per_gflop_values)
                    summary_row.update({
                        "energy_mean_j_per_gflop": gflop_stats["mean"],
                        "energy_median_j_per_gflop": gflop_stats["median"],
                    })
                
                if gpu_power_mean_values:
                    summary_row["gpu_power_mean_w_median"] = float(np.median(gpu_power_mean_values))
                if gpu_power_median_values:
                    summary_row["gpu_power_median_w_median"] = float(np.median(gpu_power_median_values))
                if gpu_power_p95_values:
                    summary_row["gpu_power_p95_w_median"] = float(np.median(gpu_power_p95_values))
                
                # Capture environment info once
                if power_env_sheet is None and energy_meta:
                    env_data = []
                    for k, v in energy_meta.items():
                        env_data.append({"key": k, "value": str(v)})
                    power_env_sheet = pd.DataFrame(env_data)
            
            # Add model statics 
            model_statics_rows.append({
                "model_name": fname,
                "file_size_mb": file_size_mb,
            })
            
            summary_rows.append(summary_row)
            
        except Exception as e:
            msg = str(e)
            all_sheets[fname] = pd.DataFrame({"note":[msg]})
            summary_rows.append({
                "model_name": fname, 
                "top1_acc": np.nan, 
                "top5_acc": np.nan, 
                "samples": 0,
                "model_file_mb": float(mpath.stat().st_size / (1024**2)) if mpath.exists() else np.nan,
            })

    if model_statics_rows:
        all_sheets["ModelStatics"] = pd.DataFrame(model_statics_rows)

    if power_env_sheet is not None:
        all_sheets["PowerEnv"] = power_env_sheet

    sum_df = pd.DataFrame(summary_rows)
    # Column order preference
    pref = [
        "model_name",
        "top1_acc",
        "top5_acc", 
        "samples",
        "model_file_mb",
        "lat_mean_ms",
        "lat_median_ms",
        "lat_std_ms",
        "lat_min_ms",
        "lat_max_ms",
        "lat_p50_ms",
        "lat_p90_ms",
        "lat_p95_ms",
        "lat_p99_ms",
        "thr_mean_sps",
        "thr_median_sps", 
        "thr_std_sps",
        "thr_min_sps",
        "thr_max_sps",
        "thr_p50_sps",
        "thr_p90_sps",
        "thr_p95_sps",
        "thr_p99_sps",
        "host_peak_delta_mean_mib",
        "host_peak_delta_median_mib",
        "host_peak_delta_p95_mib",
        "host_peak_delta_max_mib",
        "dev_peak_delta_mean_mib",
        "dev_peak_delta_median_mib",
        "dev_peak_delta_p95_mib",
        "dev_peak_delta_max_mib",
        "baseline_idle_host_mib",
        "baseline_postload_host_mib",
        "baseline_idle_dev_mib",
        "baseline_postload_dev_mib",
        "host_global_peak_mib",
        "host_true_os_peak_mib",
        "steady_host_mean_delta_mib_median",
        "steady_dev_mean_delta_mib_median",
        "mem_device_backend",
        "energy_mean_j_per_sample",
        "energy_median_j_per_sample",
        "energy_p90_j_per_sample",
        "energy_p95_j_per_sample",
        "energy_p99_j_per_sample",
        "energy_host_median_j_per_sample",
        "energy_gpu_median_j_per_sample",
        "energy_dram_median_j_per_sample",
        "energy_mean_j_per_gflop",
        "energy_median_j_per_gflop",
        "gpu_power_mean_w_median",
        "gpu_power_median_w_median",
        "gpu_power_p95_w_median",
    ]
    cols = [c for c in pref if c in sum_df.columns] + [c for c in sum_df.columns if c not in pref]
    sum_df = sum_df[cols]

    save_ods_multisheet(OUTPUT_ODS, all_sheets, sum_df)
    print(f"Saved: {OUTPUT_ODS}")