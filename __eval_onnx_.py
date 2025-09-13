# FRM ONNX eval
# type: ignore
# Usage: python __eval_onnx_.py [--latency] [--lat-warmup-batches n] [--lat-repeats-batch n] [--bs n] [--blaze] [--mem] [--mem-sample-hz HZ] [--energy] [--energy-sample-hz HZ] [--gflops-map path/to/csv]

import os, json, random, platform, argparse, time, math, threading, csv
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image

import onnxruntime as ort
import torchvision.transforms as T
from tqdm import tqdm

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
    HAVE_NVML = False

# LEVIT SHAPE RESET
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

# --- Config ---
DATA_DIR = "data"
MANIFEST_FILE = os.path.join(DATA_DIR, "manifest.csv")
IMAGENET_INDEX_FILE = "imagenet_class_index.json"
MODELS_DIR = os.path.join("models", "onnx")
OUTPUT_ODS = "frm_onnx_results.ods"
TOP_K = 5

# [OPTION]: optional model exclude list
EXCLUDE_MODELS: Optional[List[str]] = None  # e.g., ["levit_128s.onnx"]

# --- Preproc helpers ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)
CAFFE_MEAN_RGB_255 = np.array([123.68, 116.779, 103.939], np.float32)

_resize_224 = T.Compose([T.Resize(256), T.CenterCrop(224)])
_resize_299 = T.Compose([T.Resize(342), T.CenterCrop(299)])
_to_tensor  = T.ToTensor()

def _finite(vals):
    return [v for v in vals if v is not None and not math.isnan(v)]

def nanrobust_max(vals: list) -> float:
    f = _finite(vals)
    return max(f) if f else math.nan

def nanrobust_median(vals: list) -> float:
    f = _finite(vals)
    return float(np.median(f)) if f else math.nan

# --- ImageNet helpers ---

def load_imagenet_class_mapping() -> Tuple[Dict[str, int], Dict[int, str]]:
    with open(IMAGENET_INDEX_FILE, "r") as f:
        class_idx = json.load(f)
    wnid_to_idx, idx_to_class = {}, {}
    for idx, (wnid, class_name) in class_idx.items():
        wnid_to_idx[wnid] = int(idx)
        idx_to_class[int(idx)] = class_name
    return wnid_to_idx, idx_to_class

# --- Shapes / inputs ---

def get_input_size_from_shape(shape: List) -> int:
    if len(shape) >= 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return max(shape[2], shape[3])
    return 224

def get_fixed_batch_dim(shape: List) -> Optional[int]:
    if len(shape) >= 1 and isinstance(shape[0], int) and shape[0] > 0:
        return int(shape[0])
    return None

def choose_resize(size: int):
    return _resize_299 if size == 299 else _resize_224

# --- Data loading ---

def load_batch(paths: List[str], size: int) -> np.ndarray:
    tf = T.Compose([choose_resize(size), _to_tensor])
    batch = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            batch.append(tf(img).numpy())
        except Exception:
            batch.append(np.zeros((3, size, size), np.float32))
    return np.stack(batch).astype(np.float32)

# --- Normalization variants ---

def imagenet_norm(x: np.ndarray) -> np.ndarray:
    m = IMAGENET_MEAN.reshape(1,3,1,1); s = IMAGENET_STD.reshape(1,3,1,1)
    return (x - m) / s

def caffe_rgb_submean(x_0to1: np.ndarray) -> np.ndarray:
    x = (x_0to1 * 255.0).astype(np.float32)
    m = CAFFE_MEAN_RGB_255.reshape(1,3,1,1)
    return x - m

def rgb_no_norm(x_0to1: np.ndarray) -> np.ndarray:
    return x_0to1.astype(np.float32)

# --- Top‑K ---

def get_topk(logits: np.ndarray, k: int = TOP_K) -> np.ndarray:
    if logits.ndim == 1: logits = logits.reshape(1, -1)
    return np.argsort(logits, axis=1)[:, -k:][:, ::-1]

# --- Run ORT ---

def run_onnx(session: ort.InferenceSession, input_name: str, batch: np.ndarray):
    outs = session.run(None, {input_name: batch})
    pick_logits = None
    for y in outs:
        if isinstance(y, np.ndarray) and y.ndim >= 2 and y.shape[-1] >= 1000:
            pick_logits = y; break
    if pick_logits is not None:
        y = pick_logits
        if y.ndim == 1: y = y.reshape(1, -1)
        if y.shape[1] == 1001: y = y[:, 1:]
        top1 = np.argmax(y, axis=1).tolist()
        top5 = [row for row in get_topk(y, TOP_K)]
        return top1, top5, "logits"
    y = np.asarray(outs[0]).reshape(-1)
    top1 = [int(v) for v in y.tolist()]
    top5 = [np.array([v, (v+1)%1000, (v+2)%1000, (v+3)%1000, (v+4)%1000], np.int64) for v in top1]
    return top1, top5, "argmax"

# ---------- LeViT load repair (bytes) ----------

def simplify_if_needed(model_path: str, fname: str, input_size: int = 224) -> Optional[bytes]:
    if "levit" not in fname.lower() or onnx is None: return None
    try:
        m = onnx.load(model_path)
        if HAVE_SSI and SymbolicShapeInference is not None:
            try:
                m = SymbolicShapeInference.infer_shapes(m, auto_merge=True, guess_output_rank=True, verbose=0)
                print(f"[{fname}] symbolic shape inference ✓")
            except Exception as e:
                print(f"[{fname}] symbolic shape inference failed: {e}")
        if HAVE_ONNXSIM and simplify is not None:
            try:
                inp = m.graph.input[0].name  # type: ignore
                sm, ok = simplify(
                    m,
                    overwrite_input_shapes={inp: [1,3,input_size,input_size]},
                    test_input_shapes={inp: [1,3,input_size,input_size]},
                )
                if ok:
                    print(f"[{fname}] onnxsim simplified ✓")
                    m = sm
            except Exception as e:
                print(f"[{fname}] onnxsim failed: {e}")
        return m.SerializeToString()
    except Exception as e:
        print(f"[{fname}] simplify_if_needed failed early: {e}")
        return None

# ---------- Provider selection ----------

def select_ort_providers() -> Tuple[List[str], str]:
    avail = ort.get_available_providers()
    prefs = [
        ("TensorrtExecutionProvider", "TensorRT"),
        ("CUDAExecutionProvider", "CUDA"),
        ("ROCMExecutionProvider", "ROCm"),
        ("DmlExecutionProvider", "DirectML"),
        ("OpenVINOExecutionProvider", "OpenVINO"),
        ("CoreMLExecutionProvider", "CoreML"),
    ]
    chosen, label = "CPUExecutionProvider", "CPU"
    for key, lab in prefs:
        if key in avail:
            chosen, label = key, lab; break
    providers = [chosen] if chosen == "CPUExecutionProvider" else [chosen, "CPUExecutionProvider"]
    if chosen in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
        if HAVE_TORCH and torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            count = torch.cuda.device_count()
            extra = f" (x{count})" if count and count > 1 else ""
            info = f"{label} on NVIDIA {name}{extra}"
        else:
            info = f"{label} (GPU)"
    elif chosen == "ROCMExecutionProvider":
        info = "ROCm (AMD GPU)"
    elif chosen == "DmlExecutionProvider":
        info = "DirectML (GPU)"
    elif chosen == "OpenVINOExecutionProvider":
        info = "OpenVINO (Intel CPU/iGPU/NPU)"
    elif chosen == "CoreMLExecutionProvider":
        info = "CoreML (Apple)"
    else:
        cpu = platform.processor() or platform.machine()
        cores = os.cpu_count() or 1
        info = f"CPU ({cpu}, {cores} cores)"
    print(f"[System] ORT providers available: {avail}")
    print(f"[System] Using backend: {info}")
    return providers, info

# ---------- ONNX model statics ----------

def onnx_param_bytes(path: str) -> Optional[int]:
    try:
        import onnx
        from onnx import numpy_helper
        m = onnx.load(path)
        total = 0
        for init in m.graph.initializer:
            arr = numpy_helper.to_array(init)
            total += int(arr.nbytes)
        return total
    except Exception:
        return None

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
            if os.path.isdir(base):
                for d in os.listdir(base):
                    if d.startswith("intel-rapl:"):
                        dom = os.path.join(base, d)
                        efile = os.path.join(dom, "energy_uj")
                        if os.path.exists(efile):
                            self.pkg_files.append(efile)
                        # check subdomains for dram
                        for sd in os.listdir(dom):
                            if sd.startswith(d) and os.path.isdir(os.path.join(dom, sd)):
                                de = os.path.join(dom, sd, "energy_uj")
                                nfile = os.path.join(dom, sd, "name")
                                try:
                                    name = open(nfile, "r").read().strip().lower()
                                except Exception:
                                    name = ""
                                if os.path.exists(de) and ("dram" in name or "mem" in name):
                                    self.dram_files.append(de)
            self.ok = bool(self.pkg_files)
        except Exception:
            self.ok = False

    def read_energy_j(self) -> Tuple[Optional[float], Optional[float]]:
        # returns (pkg_J_total, dram_J_total) as cumulative counters converted to Joules
        try:
            pkg = 0
            for f in self.pkg_files:
                val = int(open(f, "r").read().strip())  # microjoules
                pkg += val
            dram = 0
            for f in self.dram_files:
                val = int(open(f, "r").read().strip())
                dram += val
            pkg_j = pkg / 1e6
            dram_j = (dram / 1e6) if self.dram_files else None
            return pkg_j, dram_j
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
            n = pynvml.nvmlDeviceGetCount()
            for i in range(n):
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            self.ok = n > 0
        except Exception:
            self.ok = False

    def read_power_w(self) -> Optional[float]:
        try:
            if not self.ok:
                return None
            total_mw = 0
            for h in self.handles:
                mw = pynvml.nvmlDeviceGetPowerUsage(h)  # milliwatts
                if mw is not None and mw != pynvml.NVML_VALUE_NOT_AVAILABLE:
                    total_mw += int(mw)
            return (total_mw / 1000.0)
        except Exception:
            return None

    def limits(self) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "gpu_power_limit_w": None,
            "gpu_power_limit_default_w": None,
        }
        try:
            if not self.ok: return out
            h = self.handles[0]
            out["gpu_power_limit_w"] = pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
            out["gpu_power_limit_default_w"] = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0
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
            t0 = time.perf_counter()
            host_now, _ = self._read_host_energy()
            if host_prev is not None and host_now is not None:
                dE = host_now - host_prev
                dt = self.dt
                if dE >= 0 and dt > 0:
                    host_acc_w.append(dE / dt)
            host_prev = host_now
            gpw = self._read_gpu_power()
            if gpw is not None:
                gpu_acc_w.append(gpw)
            # busy‑wait to target Hz
            t1 = time.perf_counter()
            remain = self.dt - (t1 - t0)
            if remain > 0: time.sleep(remain)
        h = float(np.mean(host_acc_w)) if host_acc_w else None
        g = float(np.mean(gpu_acc_w)) if gpu_acc_w else None
        return h, g

    def start(self):
        # reset window buffers and set starting energy counters
        self.tstamps.clear(); self.gpu_power_w.clear()
        self.host_pkg_j_start, self.host_dram_j_start = self._read_host_energy()
        self._stop.clear()
        def _runner():
            last = time.perf_counter()
            while not self._stop.is_set():
                t0 = time.perf_counter()
                self.tstamps.append(t0)
                gpw = self._read_gpu_power()
                self.gpu_power_w.append(gpw if gpw is not None else math.nan)
                # sleep to respect sampling rate
                t1 = time.perf_counter()
                remain = self.dt - (t1 - t0)
                if remain > 0:
                    time.sleep(remain)
                last = t0
        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        # Host cumulative energies (J)
        host_pkg_j_end, host_dram_j_end = self._read_host_energy()
        pkg_dJ = (host_pkg_j_end - self.host_pkg_j_start) if (host_pkg_j_end is not None and self.host_pkg_j_start is not None) else math.nan
        dram_dJ = (host_dram_j_end - self.host_dram_j_start) if (host_dram_j_end is not None and self.host_dram_j_start is not None) else math.nan
        # GPU energy via sum(power*dt)
        # Use fixed dt per sample; if timestamps sparse, fallback to dt configured
        if len(self.tstamps) >= 2:
            dt_est = float(np.median(np.diff(self.tstamps)))
        else:
            dt_est = self.dt
        gp = np.array(self.gpu_power_w, dtype=float)
        gp = gp[np.isfinite(gp)]
        gpu_j = float(np.sum(gp) * dt_est) if gp.size > 0 else math.nan

        # Power profile statistics over window
        def _stats(arr: List[float]) -> Dict[str, float]:
            arr = [v for v in arr if v is not None and np.isfinite(v)]
            if not arr:
                return {k: math.nan for k in ["mean","median","std","p90","p95","p99"]}
            a = np.array(arr, dtype=float)
            return {
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "std": float(a.std(ddof=1)) if a.size>1 else 0.0,
                "p90": float(np.percentile(a,90)),
                "p95": float(np.percentile(a,95)),
                "p99": float(np.percentile(a,99)),
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
                info.update(self.nvml.limits())
                info["nvidia_driver_version"] = pynvml.nvmlSystemGetDriverVersion().decode() if hasattr(pynvml.nvmlSystemGetDriverVersion(), 'decode') else str(pynvml.nvmlSystemGetDriverVersion())
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
                p = os.path.join(roots, d, "cpufreq", "scaling_governor")
                if os.path.exists(p):
                    try:
                        g = open(p, "r").read().strip()
                        govs.append(g)
                    except Exception:
                        pass
        if govs:
            # majority label
            vals, counts = np.unique(np.array(govs), return_counts=True)
            return f"{vals[np.argmax(counts)]} (per‑CPU: {','.join(govs[:16])}{'...' if len(govs)>16 else ''})"
    except Exception:
        pass
    return "unknown"

# ---------- Memory sampling (unchanged from previous) ----------

def _read_linux_vm(value_key: str) -> Optional[int]:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(value_key + ":"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kB = int(parts[1])
                        return kB * 1024
    except Exception:
        pass
    return None

def get_host_rss_bytes() -> Optional[int]:
    if platform.system() == "Linux":
        v = _read_linux_vm("VmRSS")
        if v is not None: return v
    if HAVE_PSUTIL:
        try:
            return psutil.Process().memory_info().rss
        except Exception:
            pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            return int(ru)
        else:
            return int(ru) * 1024
    except Exception:
        return None

def get_host_peak_bytes_hwm() -> Optional[int]:
    if platform.system() == "Linux":
        v = _read_linux_vm("VmHWM")
        if v is not None: return v
    if HAVE_PSUTIL and platform.system() == "Windows":
        try:
            mi = psutil.Process().memory_info()
            peak = getattr(mi, "peak_wset", None)
            if peak is not None:
                return int(peak)
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
        n = pynvml.nvmlDeviceGetCount()
        total = 0
        found = False
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(h)
            except Exception:
                try:
                    procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3(h)
                except Exception:
                    procs = []
            for p in procs:
                if int(p.pid) == pid:
                    used = getattr(p, "usedGpuMemory", 0)
                    if used is None or used == pynvml.NVML_VALUE_NOT_AVAILABLE:
                        continue
                    total += int(used)
                    found = True
        if found:
            return total
        return None
    except Exception:
        return None

class MemorySampler:
    def __init__(self, hz: int = 300, provider_label: str = ""):
        self.hz = max(50, min(int(hz), 2000))
        self.dt = 1.0 / float(self.hz)
        self.pid = os.getpid()
        self.provider_label = provider_label
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.host_samples: List[int] = []
        self.dev_samples: List[int] = []
        self._have_nvml = init_nvml_if_present() if ("CUDA" in provider_label or "TensorRT" in provider_label) else False
        self.idle_host: Optional[int] = None
        self.idle_dev: Optional[int] = None
        self.postload_host: Optional[int] = None
        self.postload_dev: Optional[int] = None
        self.global_host_peak: int = 0
        self.global_dev_peak: int = 0
        self.true_host_peak_bytes: Optional[int] = None

    def _poll_once(self):
        h = get_host_rss_bytes()
        if h is not None:
            self.host_samples.append(h)
            if h > self.global_host_peak:
                self.global_host_peak = h
        d = None
        if self._have_nvml:
            d = nvml_proc_used_bytes(self.pid)
            if d is not None:
                self.dev_samples.append(d)
                if d > self.global_dev_peak:
                    self.global_dev_peak = d

    def start(self):
        self.host_samples.clear()
        self.dev_samples.clear()
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
            self._thread.join(timeout=1.0)
            self._thread = None
        def stats(arr: List[int]) -> Tuple[float, float]:
            if not arr: return (math.nan, math.nan)
            a = np.array(arr, dtype=np.float64)
            return float(a.max() / (1024*1024)), float(a.mean() / (1024*1024))
        host_peak_mib, host_mean_mib = stats(self.host_samples)
        dev_peak_mib,  dev_mean_mib  = stats(self.dev_samples)
        self.host_samples.clear()
        self.dev_samples.clear()
        return {
            "host_window_peak_mib": host_peak_mib,
            "host_window_mean_mib": host_mean_mib,
            "dev_window_peak_mib": dev_peak_mib,
            "dev_window_mean_mib": dev_mean_mib,
        }

    def record_idle_baseline(self):
        self.idle_host = get_host_rss_bytes()
        self.idle_dev  = nvml_proc_used_bytes(self.pid) if self._have_nvml else None

    def record_postload_baseline(self):
        self.postload_host = get_host_rss_bytes()
        self.postload_dev  = nvml_proc_used_bytes(self.pid) if self._have_nvml else None
        self.true_host_peak_bytes = get_host_peak_bytes_hwm()

    def finalize(self) -> Dict[str, Any]:
        out = {
            "host_global_peak_mib": (self.global_host_peak / (1024*1024)),
            "dev_global_peak_mib":  (self.global_dev_peak  / (1024*1024)) if self.global_dev_peak else math.nan,
            "host_true_os_peak_mib": (self.true_host_peak_bytes / (1024*1024)) if self.true_host_peak_bytes else math.nan,
            "baselines": {
                "idle_host_mib": (self.idle_host / (1024*1024)) if self.idle_host else math.nan,
                "idle_dev_mib":  (self.idle_dev  / (1024*1024)) if self.idle_dev  else math.nan,
                "postload_host_mib": (self.postload_host / (1024*1024)) if self.postload_host else math.nan,
                "postload_dev_mib":  (self.postload_dev  / (1024*1024)) if self.postload_dev  else math.nan,
            },
            "device_backend": ("NVIDIA+NVML" if self._have_nvml else ("Unknown/No-device" if "CPU" in self.provider_label else f"No per-process VRAM for {self.provider_label}")),
        }
        return out

# ---- Robust stats helpers ----

def robust_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {k: math.nan for k in ["count","mean","median","std","min","max","p50","p90","p95","p99"]}
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

# ---------- Single‑pass evaluation (acc + latency/throughput + memory + energy) ----------

def evaluate_model(
    model_path: str,
    model_name: str,
    rows: List[Tuple[str,int]],
    providers: List[str],
    req_bs: int,
    do_latency: bool,
    warmup_batches: int,
    repeats_batch: int,
    do_mem: bool,
    mem_hz: int,
    provider_label: str,
    do_energy: bool,
    energy_hz: int,
    gflops_map: Optional[Dict[str, float]],
) -> Tuple[pd.DataFrame, float, Optional[float], Dict[str, Any], Dict[str, Any]]:
    """
    One pass over data; memory & energy sampling windows wrap each timed session.run.
    Returns per‑sample DF, accs, mem_meta, energy_meta.
    """
    onnx_bytes = simplify_if_needed(model_path, model_name, 224)
    try:
        session = ort.InferenceSession(onnx_bytes or model_path, providers=providers, sess_options=ort.SessionOptions())
    except Exception:
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        session = ort.InferenceSession(onnx_bytes or model_path, providers=providers, sess_options=so)

    # Memory sampler + baselines
    sampler = MemorySampler(hz=mem_hz, provider_label=provider_label) if do_mem else None
    if sampler:
        sampler.record_idle_baseline()

    # Energy sampler + baselines (idle/post‑load power)
    es = EnergySampler(hz=energy_hz) if do_energy else None
    if es:
        es.record_idle_power()

    inp = session.get_inputs()[0]
    size = get_input_size_from_shape(inp.shape)

    # postload baselines
    if sampler:
        sampler.record_postload_baseline()
    if es:
        es.record_postload_power()

    img_paths = [os.path.join(DATA_DIR, rel) for (rel, _y) in rows]
    x0 = load_batch(img_paths, size)

    lower = model_name.lower()
    if lower == "resnet50_v1.onnx":
        x = caffe_rgb_submean(x0)
    elif lower == "mobilevit_xxs.onnx":
        x = rgb_no_norm(x0)
    else:
        x = imagenet_norm(x0)

    fixed_bs = get_fixed_batch_dim(inp.shape)
    if req_bs and req_bs > 0:
        if fixed_bs is not None and fixed_bs != req_bs:
            raise RuntimeError(f"{model_name}: fixed batch {fixed_bs} != requested --bs {req_bs}")
        target_bs = req_bs
    else:
        target_bs = fixed_bs if fixed_bs is not None else 32

    total = (len(x) // target_bs) * target_bs
    if total < len(x) and req_bs and req_bs > 0:
        print(f"[BS] {model_name}: dropping last {len(x) - total} sample(s) to enforce --bs {target_bs}")
    x = x[:total] if total > 0 else x
    if target_bs <= 0: raise RuntimeError(f"{model_name}: invalid resolved batch size {target_bs}")

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

    pbar = tqdm(total=len(x), desc=f"{model_name} eval(bs={target_bs})", leave=False, ncols=90)
    i = 0

    def _time_once_with_mem_energy(chunk: np.ndarray) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        if sampler: sampler.start()
        if es: es.start()
        t0 = time.perf_counter_ns()
        session.run(None, {inp.name: chunk})
        t1 = time.perf_counter_ns()
        w_mem = sampler.stop() if sampler else {}
        w_en  = es.stop() if es else {}
        return (t1 - t0) / 1e6, w_mem, w_en  # ms, mem window, energy window

    def _warmup(chunk: np.ndarray, n: int):
        for _ in range(n):
            session.run(None, {inp.name: chunk})

    while i < len(x):
        end = i + target_bs
        chunk = x[i:end].astype(np.float32)

        # --- Warmup (not recorded) ---
        if do_latency and warmup_batches > 0:
            _warmup(chunk, warmup_batches)

        # --- Timed run(s) + predictions ---
        times_ms: List[float] = []
        host_peaks: List[float] = []
        dev_peaks:  List[float] = []
        host_means: List[float] = []
        dev_means:  List[float] = []

        # energy window aggregates across repeats
        batch_energy_total_j: List[float] = []
        batch_energy_host_j:  List[float] = []
        batch_energy_gpu_j:   List[float] = []
        batch_energy_dram_j:  List[float] = []
        batch_gpu_power_means:   List[float] = []
        batch_gpu_power_medians: List[float] = []
        batch_gpu_power_stds:    List[float] = []
        batch_gpu_power_p90s:    List[float] = []
        batch_gpu_power_p95s:    List[float] = []
        batch_gpu_power_p99s:    List[float] = []

        # First timed run (recorded) used for timing/mem/energy; accuracy from a separate run right after
        if do_latency or do_mem or do_energy:
            ms, w_mem, w_en = _time_once_with_mem_energy(chunk)
            times_ms.append(ms)
            if sampler and w_mem:
                ph = w_mem.get("host_window_peak_mib", math.nan)
                mh = w_mem.get("host_window_mean_mib", math.nan)
                pdv = w_mem.get("dev_window_peak_mib",  math.nan)
                mdv = w_mem.get("dev_window_mean_mib",  math.nan)
                host_peaks.append(ph); host_means.append(mh)
                dev_peaks.append(pdv);  dev_means.append(mdv)
            if es and w_en:
                host_j = w_en.get("host_pkg_j", math.nan)
                dram_j = w_en.get("host_dram_j", math.nan)
                gpu_j  = w_en.get("gpu_j", math.nan)
                tot_j  = float(np.nansum([host_j, dram_j, gpu_j]))
                batch_energy_total_j.append(tot_j)
                batch_energy_host_j.append(host_j)
                batch_energy_gpu_j.append(gpu_j)
                batch_energy_dram_j.append(dram_j)
                batch_gpu_power_means.append(w_en.get("gpu_power_mean_w", math.nan))
                batch_gpu_power_medians.append(w_en.get("gpu_power_median_w", math.nan))
                batch_gpu_power_stds.append(w_en.get("gpu_power_std_w", math.nan))
                batch_gpu_power_p90s.append(w_en.get("gpu_power_p90_w", math.nan))
                batch_gpu_power_p95s.append(w_en.get("gpu_power_p95_w", math.nan))
                batch_gpu_power_p99s.append(w_en.get("gpu_power_p99_w", math.nan))

        # accuracy from fresh run to avoid timing cache oddities
        t1_preds, t5_preds, _mode = run_onnx(session, inp.name, chunk)

        # Additional repeats for robust timing/mem/energy
        if do_latency:
            for _ in range(max(0, repeats_batch - 1)):
                ms, w_mem, w_en = _time_once_with_mem_energy(chunk)
                times_ms.append(ms)
                if sampler and w_mem:
                    ph = w_mem.get("host_window_peak_mib", math.nan)
                    mh = w_mem.get("host_window_mean_mib", math.nan)
                    pdv = w_mem.get("dev_window_peak_mib",  math.nan)
                    mdv = w_mem.get("dev_window_mean_mib",  math.nan)
                    host_peaks.append(ph); host_means.append(mh)
                    dev_peaks.append(pdv);  dev_means.append(mdv)
                if es and w_en:
                    host_j = w_en.get("host_pkg_j", math.nan)
                    dram_j = w_en.get("host_dram_j", math.nan)
                    gpu_j  = w_en.get("gpu_j", math.nan)
                    tot_j  = float(np.nansum([host_j, dram_j, gpu_j]))
                    batch_energy_total_j.append(tot_j)
                    batch_energy_host_j.append(host_j)
                    batch_energy_gpu_j.append(gpu_j)
                    batch_energy_dram_j.append(dram_j)
                    batch_gpu_power_means.append(w_en.get("gpu_power_mean_w", math.nan))
                    batch_gpu_power_medians.append(w_en.get("gpu_power_median_w", math.nan))
                    batch_gpu_power_stds.append(w_en.get("gpu_power_std_w", math.nan))
                    batch_gpu_power_p90s.append(w_en.get("gpu_power_p90_w", math.nan))
                    batch_gpu_power_p95s.append(w_en.get("gpu_power_p95_w", math.nan))
                    batch_gpu_power_p99s.append(w_en.get("gpu_power_p99_w", math.nan))

        # Medians (paper‑grade) and per‑sample assignment
        if do_latency and times_ms:
            median_ms = float(np.median(times_ms))
            per_item = median_ms / len(chunk)
            sps = len(chunk) / (median_ms / 1000.0) if median_ms > 0 else math.nan
            per_sample_latency_ms.extend([per_item] * len(chunk))
            per_sample_throughput_sps.extend([sps] * len(chunk))

        if es and batch_energy_total_j:
            # median across repeats
            med_tot_j   = float(np.nanmedian(batch_energy_total_j))
            med_host_j  = float(np.nanmedian(batch_energy_host_j))
            med_gpu_j   = float(np.nanmedian(batch_energy_gpu_j))
            med_dram_j  = float(np.nanmedian(batch_energy_dram_j)) if any(np.isfinite(batch_energy_dram_j)) else math.nan
            # per‑sample energy
            e_per_samp  = med_tot_j / len(chunk) if len(chunk) else math.nan
            e_host_samp = med_host_j / len(chunk) if len(chunk) else math.nan
            e_gpu_samp  = med_gpu_j / len(chunk) if len(chunk) else math.nan
            e_dram_samp = med_dram_j / len(chunk) if len(chunk) else math.nan
            # power profile per window (use medians of repeat‑level stats)
            gpu_pw_mean = float(np.nanmedian(batch_gpu_power_means)) if batch_gpu_power_means else math.nan
            gpu_pw_med  = float(np.nanmedian(batch_gpu_power_medians)) if batch_gpu_power_medians else math.nan
            gpu_pw_std  = float(np.nanmedian(batch_gpu_power_stds)) if batch_gpu_power_stds else math.nan
            gpu_pw_p90  = float(np.nanmedian(batch_gpu_power_p90s)) if batch_gpu_power_p90s else math.nan
            gpu_pw_p95  = float(np.nanmedian(batch_gpu_power_p95s)) if batch_gpu_power_p95s else math.nan
            gpu_pw_p99  = float(np.nanmedian(batch_gpu_power_p99s)) if batch_gpu_power_p99s else math.nan

            # J/GFLOP if available
            model_gflops = None
            if gflops_map is not None:
                model_gflops = gflops_map.get(model_name) or gflops_map.get(os.path.splitext(model_name)[0])
            e_per_gflop = (e_per_samp / model_gflops) if (model_gflops and model_gflops > 0) else math.nan

            per_sample_energy_j.extend([e_per_samp] * len(chunk))
            per_sample_energy_host_j.extend([e_host_samp] * len(chunk))
            per_sample_energy_gpu_j.extend([e_gpu_samp] * len(chunk))
            per_sample_energy_dram_j.extend([e_dram_samp] * len(chunk))
            per_sample_gpu_power_mean_w.extend([gpu_pw_mean] * len(chunk))
            per_sample_gpu_power_median_w.extend([gpu_pw_med] * len(chunk))
            per_sample_gpu_power_std_w.extend([gpu_pw_std] * len(chunk))
            per_sample_gpu_power_p90_w.extend([gpu_pw_p90] * len(chunk))
            per_sample_gpu_power_p95_w.extend([gpu_pw_p95] * len(chunk))
            per_sample_gpu_power_p99_w.extend([gpu_pw_p99] * len(chunk))
            per_sample_energy_per_gflop.extend([e_per_gflop] * len(chunk))

        # Memory deltas (subtract postload baselines)
        if sampler and (host_peaks or host_means):
            idle_h = (sampler.idle_host or 0) / (1024*1024)
            post_h = (sampler.postload_host or sampler.idle_host or 0) / (1024*1024)
            base_h = max(idle_h, post_h)
            idle_d = (sampler.idle_dev  if sampler.idle_dev  is not None else math.nan)
            post_d = (sampler.postload_dev if sampler.postload_dev is not None else idle_d)
            base_d_mib = (max(idle_d, post_d) / (1024*1024)) if (not math.isnan(idle_d) and not math.isnan(post_d)) else math.nan
            batch_host_peak = nanrobust_max(host_peaks)
            batch_host_mean = nanrobust_median(host_means)
            batch_dev_peak  = nanrobust_max(dev_peaks) if dev_peaks else math.nan
            batch_dev_mean  = nanrobust_median(dev_means) if dev_means else math.nan
            dh_peak = max(0.0, batch_host_peak - base_h) if not math.isnan(batch_host_peak) else math.nan
            dh_mean = max(0.0, batch_host_mean - base_h) if not math.isnan(batch_host_mean) else math.nan
            dd_peak = max(0.0, batch_dev_peak  - base_d_mib) if (not math.isnan(batch_dev_peak) and not math.isnan(base_d_mib)) else math.nan
            dd_mean = max(0.0, batch_dev_mean  - base_d_mib) if (not math.isnan(batch_dev_mean) and not math.isnan(base_d_mib)) else math.nan
            per_sample_host_peak_mib.extend([dh_peak] * len(chunk))
            per_sample_host_mean_mib.extend([dh_mean] * len(chunk))
            per_sample_dev_peak_mib.extend([dd_peak] * len(chunk))
            per_sample_dev_mean_mib.extend([dd_mean] * len(chunk))
            if not math.isnan(dh_mean): steady_host_means.append(dh_mean)
            if not math.isnan(dd_mean): steady_dev_means.append(dd_mean)

        # accumulate accuracy
        top1_all.extend(t1_preds)
        top5_all.extend((t5_preds or [])[:len(t1_preds)])
        i = end
        pbar.update(len(t1_preds))
    pbar.close()

    # strict alignment & dtype coercion for export
    N = len(top1_all)
    rel_paths = [rows[j][0] for j in range(N)]
    y_true    = [int(rows[j][1]) for j in range(N)]
    top1_all  = [int(v) for v in top1_all]
    if len(top5_all) != N:
        top5_all = [np.array([], dtype=np.int64)] * N
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
        data["latency_ms_median"] = per_sample_latency_ms
        data["throughput_samples_per_sec"] = per_sample_throughput_sps
    if sampler:
        data["host_peak_delta_mib"] = per_sample_host_peak_mib
        data["host_mean_delta_mib"] = per_sample_host_mean_mib

        if getattr(sampler, "_have_nvml", False):
            data["dev_peak_delta_mib"] = per_sample_dev_peak_mib
            data["dev_mean_delta_mib"] = per_sample_dev_mean_mib

    if do_energy:
        data["energy_j_per_sample"] = per_sample_energy_j
        data["energy_host_j_per_sample"] = per_sample_energy_host_j
        data["energy_gpu_j_per_sample"] = per_sample_energy_gpu_j
        data["energy_dram_j_per_sample"] = per_sample_energy_dram_j
        data["gpu_power_mean_w"] = per_sample_gpu_power_mean_w
        data["gpu_power_median_w"] = per_sample_gpu_power_median_w
        data["gpu_power_std_w"] = per_sample_gpu_power_std_w
        data["gpu_power_p90_w"] = per_sample_gpu_power_p90_w
        data["gpu_power_p95_w"] = per_sample_gpu_power_p95_w
        data["gpu_power_p99_w"] = per_sample_gpu_power_p99_w
        data["energy_per_gflop"] = per_sample_energy_per_gflop

    df = pd.DataFrame(data)

    # Finalize memory provenance & aggregates
    mem_meta: Dict[str, Any] = {}
    if sampler:
        mem_meta = sampler.finalize()
        # add steady-state medians
        mem_meta["steady_host_mean_delta_mib_median"] = (
            float(np.nanmedian(steady_host_means)) if steady_host_means else math.nan
        )
        mem_meta["steady_dev_mean_delta_mib_median"] = (
            float(np.nanmedian(steady_dev_means)) if steady_dev_means else math.nan
        )

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

# --- ODS writer ---
def save_ods_multisheet(path: str, sheets: Dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="odf") as writer:
        for name, df in sheets.items():
            safe = name[:31]
            (df if (df is not None and not df.empty) else pd.DataFrame({"note": ["empty or failed"]})).to_excel(
                writer, sheet_name=safe, index=False
            )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

def load_gflops_map(csv_path: Optional[str]) -> Optional[Dict[str, float]]:
    if not csv_path:
        return None
    if not os.path.exists(csv_path):
        print(f"[GFLOPs] Map not found: {csv_path}")
        return None
    mp: Dict[str, float] = {}
    try:
        with open(csv_path, "r", newline="") as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                # accept common column names
                name = row.get("model_name") or row.get("model") or row.get("name")
                gfl = row.get("gflops") or row.get("GFLOPs") or row.get("flops_g")
                if name and gfl:
                    try:
                        mp[name] = float(gfl)
                    except Exception:
                        pass
        print(f"[GFLOPs] Loaded entries: {len(mp)}")
        return mp
    except Exception as e:
        print(f"[GFLOPs] Failed to load: {e}")
        return None

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FRM ONNX eval (single-pass acc+latency+memory+energy, bs-enforced)")
    parser.add_argument("--blaze", action="store_true", help="Run only 10 random samples")
    parser.add_argument("--bs", type=int, default=0, help="Batch size to enforce. 0=auto (fixed N or 32)")
    parser.add_argument("--latency", action="store_true", help="Measure latency/throughput in the same pass")
    parser.add_argument("--lat-warmup-batches", type=int, default=2, help="Warmup runs per batch (not recorded)")
    parser.add_argument("--lat-repeats-batch", type=int, default=10, help="Timed repeats per batch; median used")
    parser.add_argument("--mem", action="store_true", help="Enable memory sampling & reporting (host+device)")
    parser.add_argument("--mem-sample-hz", type=int, default=300, help="Sampler frequency (Hz), 200–500 recommended")
    parser.add_argument("--energy", action="store_true", help="Enable power/energy sampling (host RAPL + GPU NVML)")
    parser.add_argument("--energy-sample-hz", type=int, default=300, help="Energy sampler frequency (Hz), ≥200 recommended")
    parser.add_argument("--gflops-map", type=str, default="", help="CSV with columns [model_name,gflops] for J/GFLOP")
    args = parser.parse_args()

    random.seed(1337)
    np.random.seed(1337)

    providers, backend_info = select_ort_providers()
    print(f"[Batch] {'Enforcing' if (args.bs and args.bs>0) else 'Auto'} batch size: {args.bs if args.bs else 'auto(32 or fixed N)'}")

    gflops_map = load_gflops_map(args.gflops_map)

    manifest = pd.read_csv(MANIFEST_FILE)
    wnid_to_idx, _ = load_imagenet_class_mapping()
    manifest = manifest[manifest["wnid"].isin(wnid_to_idx.keys())].copy()
    manifest["label"] = manifest["wnid"].map(wnid_to_idx).astype(int)

    if args.blaze and len(manifest) > 10:
        manifest = manifest.sample(n=10, random_state=random.randint(0, 999999)).reset_index(drop=True)
        print("[BLAZE] Using 10 random samples")

    rows_all: List[Tuple[str, int]] = list(zip(manifest["rel_path"].tolist(), manifest["label"].tolist()))

    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".onnx")]
    if EXCLUDE_MODELS is not None:
        excl = set(EXCLUDE_MODELS)
        model_files = [f for f in model_files if f not in excl]
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
                "onnxruntime_version",
                "providers",
                "backend_info",
                "psutil",
                "nvml_available",
            ],
            "value": [
                platform.node(),
                platform.platform(),
                platform.python_version(),
                ort.__version__,
                ",".join(ort.get_available_providers()),
                backend_info,
                str(HAVE_PSUTIL),
                str(HAVE_NVML),
            ],
        }
    )
    all_sheets["RunInfo"] = runinfo

    # Optional: ONNX model statics sheet
    model_statics_rows = []

    # We'll also capture one environment disclosure from energy sampler (if enabled)
    power_env_sheet: Optional[pd.DataFrame] = None

    for fname in tqdm(model_files, desc="Models", ncols=90):
        if fname.endswith(".simp.onnx") or fname.endswith(".shape.onnx"):
            continue
        mpath = os.path.join(MODELS_DIR, fname)
        try:
            # statics
            fsize_mb = (os.path.getsize(mpath) / (1024 * 1024.0)) if os.path.exists(mpath) else math.nan
            pbytes = onnx_param_bytes(mpath)
            pbytes_mb = (pbytes / (1024 * 1024.0)) if pbytes is not None else math.nan

            df, t1, t5, mem_meta, energy_meta = evaluate_model(
                mpath,
                fname,
                rows_all,
                providers,
                args.bs,
                do_latency=args.latency,
                warmup_batches=args.lat_warmup_batches,
                repeats_batch=args.lat_repeats_batch,
                do_mem=args.mem,
                mem_hz=args.mem_sample_hz,
                provider_label=backend_info,
                do_energy=args.energy,
                energy_hz=args.energy_sample_hz,
                gflops_map=gflops_map,
            )
            base = os.path.splitext(fname)[0]
            all_sheets[base] = df

            # summary row with latency/memory/energy distribution (per-sample)
            row: Dict[str, Any] = {
                "model_name": fname,
                "top1_acc": f"{t1:.4f}",
                "top5_acc": (f"{t5:.4f}" if t5 is not None else ""),
                "samples": len(df),
                "model_file_mb": f"{fsize_mb:.2f}",
                "param_bytes_mb": (f"{pbytes_mb:.2f}" if not math.isnan(pbytes_mb) else ""),
            }
            # latency summary
            if args.latency and "latency_ms_median" in df.columns:
                vals = df["latency_ms_median"].astype(float).tolist()
                stats = robust_stats(vals)
                row.update(
                    {
                        "lat_mean_ms": stats["mean"],
                        "lat_median_ms": stats["median"],
                        "lat_std_ms": stats["std"],
                        "lat_min_ms": stats["min"],
                        "lat_max_ms": stats["max"],
                        "lat_p50_ms": stats["p50"],
                        "lat_p90_ms": stats["p90"],
                        "lat_p95_ms": stats["p95"],
                        "lat_p99_ms": stats["p99"],
                    }
                )
                if "throughput_samples_per_sec" in df.columns:
                    tvals = df["throughput_samples_per_sec"].astype(float).tolist()
                    tstats = robust_stats(tvals)
                    row.update(
                        {
                            "thr_mean_sps": tstats["mean"],
                            "thr_median_sps": tstats["median"],
                            "thr_std_sps": tstats["std"],
                            "thr_min_sps": tstats["min"],
                            "thr_max_sps": tstats["max"],
                            "thr_p50_sps": tstats["p50"],
                            "thr_p90_sps": tstats["p90"],
                            "thr_p95_sps": tstats["p95"],
                            "thr_p99_sps": tstats["p99"],
                        }
                    )
            # memory summary
            if args.mem and "host_peak_delta_mib" in df.columns:
                hp = df["host_peak_delta_mib"].astype(float).tolist()
                hm = df["host_mean_delta_mib"].astype(float).tolist()
                dp = df["dev_peak_delta_mib"].astype(float).tolist() if "dev_peak_delta_mib" in df.columns else []
                dm = df["dev_mean_delta_mib"].astype(float).tolist() if "dev_mean_delta_mib" in df.columns else []
                hp_s = robust_stats(hp)
                hm_s = robust_stats(hm)
                dp_s = robust_stats(dp) if dp else {k: math.nan for k in ["mean", "median", "std", "min", "max", "p50", "p90", "p95", "p99"]}
                dm_s = robust_stats(dm) if dm else {k: math.nan for k in ["mean", "median", "std", "min", "max", "p50", "p90", "p95", "p99"]}

                row.update(
                    {
                        # Host
                        "host_peak_delta_mean_mib": hp_s["mean"],
                        "host_peak_delta_median_mib": hp_s["median"],
                        "host_peak_delta_p95_mib": hp_s["p95"],
                        "host_peak_delta_max_mib": hp_s["max"],
                        "host_steady_mean_delta_median_mib": hm_s["median"],  # steady-state ΔRSS
                        # Device
                        "dev_peak_delta_mean_mib": dp_s["mean"],
                        "dev_peak_delta_median_mib": dp_s["median"],
                        "dev_peak_delta_p95_mib": dp_s["p95"],
                        "dev_peak_delta_max_mib": dp_s["max"],
                        "dev_steady_mean_delta_median_mib": dm_s["median"],
                    }
                )
                # Baselines & global peaks for traceability
                row.update(
                    {
                        "baseline_idle_host_mib": mem_meta.get("baselines", {}).get("idle_host_mib", math.nan),
                        "baseline_postload_host_mib": mem_meta.get("baselines", {}).get("postload_host_mib", math.nan),
                        "baseline_idle_dev_mib": mem_meta.get("baselines", {}).get("idle_dev_mib", math.nan),
                        "baseline_postload_dev_mib": mem_meta.get("baselines", {}).get("postload_dev_mib", math.nan),
                        "host_global_peak_mib": mem_meta.get("host_global_peak_mib", math.nan),
                        "host_true_os_peak_mib": mem_meta.get("host_true_os_peak_mib", math.nan),
                        "dev_global_peak_mib": mem_meta.get("dev_global_peak_mib", math.nan),
                        "steady_host_mean_delta_mib_median": mem_meta.get("steady_host_mean_delta_mib_median", math.nan),
                        "steady_dev_mean_delta_mib_median": mem_meta.get("steady_dev_mean_delta_mib_median", math.nan),
                        "mem_device_backend": mem_meta.get("device_backend", ""),
                    }
                )

            # energy summary
            if args.energy and "energy_j_per_sample" in df.columns:
                ej = df["energy_j_per_sample"].astype(float).tolist()
                ej_s = robust_stats(ej)
                row.update(
                    {
                        "energy_mean_j_per_sample": ej_s["mean"],
                        "energy_median_j_per_sample": ej_s["median"],
                        "energy_p95_j_per_sample": ej_s["p95"],
                    }
                )
                if "energy_host_j_per_sample" in df.columns:
                    eh = df["energy_host_j_per_sample"].astype(float).tolist()
                    row["energy_host_median_j_per_sample"] = robust_stats(eh)["median"]
                if "energy_gpu_j_per_sample" in df.columns:
                    eg = df["energy_gpu_j_per_sample"].astype(float).tolist()
                    row["energy_gpu_median_j_per_sample"] = robust_stats(eg)["median"]
                if "energy_dram_j_per_sample" in df.columns:
                    ed = df["energy_dram_j_per_sample"].astype(float).tolist()
                    row["energy_dram_median_j_per_sample"] = robust_stats(ed)["median"]

                if "energy_per_gflop" in df.columns:
                    epg = [v for v in df["energy_per_gflop"].astype(float).tolist() if np.isfinite(v)]
                    if epg:
                        epg_s = robust_stats(epg)
                        row.update(
                            {
                                "energy_mean_j_per_gflop": epg_s["mean"],
                                "energy_median_j_per_gflop": epg_s["median"],
                            }
                        )

                # GPU power profile (medians of window statistics already assigned per-sample)
                if "gpu_power_mean_w" in df.columns:
                    gp_mean = [v for v in df["gpu_power_mean_w"].astype(float).tolist() if np.isfinite(v)]
                    gp_med = [v for v in df["gpu_power_median_w"].astype(float).tolist() if np.isfinite(v)]
                    gp_p95 = [v for v in df["gpu_power_p95_w"].astype(float).tolist() if np.isfinite(v)]
                    if gp_mean:
                        row["gpu_power_mean_w_median"] = float(np.median(gp_mean))
                    if gp_med:
                        row["gpu_power_median_w_median"] = float(np.median(gp_med))
                    if gp_p95:
                        row["gpu_power_p95_w_median"] = float(np.median(gp_p95))

                # Capture one environment disclosure sheet (same across models typically)
                if power_env_sheet is None and energy_meta:
                    # flatten dict to 2-cols
                    kv = []
                    for k, v in energy_meta.items():
                        if isinstance(v, dict):
                            for kk, vv in v.items():
                                kv.append((f"{k}.{kk}", vv))
                        else:
                            kv.append((k, v))
                    power_env_sheet = pd.DataFrame(kv, columns=["key", "value"])
                    all_sheets["PowerEnv"] = power_env_sheet

            summary_rows.append(row)
            model_statics_rows.append(
                {
                    "model_name": fname,
                    "model_file_mb": f"{fsize_mb:.2f}",
                    "param_bytes_mb": (f"{pbytes_mb:.2f}" if not math.isnan(pbytes_mb) else ""),
                }
            )
        except Exception as e:
            base = os.path.splitext(fname)[0]
            err_df = pd.DataFrame({"Status": ["load_failed"], "Error": [str(e)], "Model": [fname]})
            all_sheets[base] = err_df
            summary_rows.append({"model_name": fname, "top1_acc": "", "top5_acc": "", "samples": 0})

    if model_statics_rows:
        all_sheets["ModelStatics"] = pd.DataFrame(model_statics_rows)

    sum_df = pd.DataFrame(summary_rows)
    # Column order preference
    pref = [
        "model_name",
        "top1_acc",
        "top5_acc",
        "samples",
        "model_file_mb",
        "param_bytes_mb",
        # latency
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
        # memory (host/device)
        "host_peak_delta_mean_mib",
        "host_peak_delta_median_mib",
        "host_peak_delta_p95_mib",
        "host_peak_delta_max_mib",
        "host_steady_mean_delta_median_mib",
        "dev_peak_delta_mean_mib",
        "dev_peak_delta_median_mib",
        "dev_peak_delta_p95_mib",
        "dev_peak_delta_max_mib",
        "dev_steady_mean_delta_median_mib",
        # baselines & provenance
        "baseline_idle_host_mib",
        "baseline_postload_host_mib",
        "baseline_idle_dev_mib",
        "baseline_postload_dev_mib",
        "host_global_peak_mib",
        "host_true_os_peak_mib",
        "dev_global_peak_mib",
        "steady_host_mean_delta_mib_median",
        "steady_dev_mean_delta_mib_median",
        "mem_device_backend",
        # energy
        "energy_mean_j_per_sample",
        "energy_median_j_per_sample",
        "energy_p95_j_per_sample",
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
    print(f"Saved {OUTPUT_ODS}")