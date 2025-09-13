# FRM PyTorch eval
# type: ignore
# Usage: python __eval_torch_.py [--latency] [--lat-warmup-batches n] [--lat-repeats-batch n] [--bs n] [--blaze] [--mem] [--mem-sample-hz HZ] [--energy] [--energy-sample-hz HZ] [--gflops-map path/to/csv]

from __future__ import annotations
import argparse
import json
import math
import os
import platform
import random
import re
import threading
import time
import warnings
from pathlib import Path

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

try:
    import timm
    HAVE_TIMM = True
except Exception:
    timm = None
    HAVE_TIMM = False

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

# Resource module (Unix-like systems only)
try:
    import resource
    HAVE_RESOURCE = True
except Exception:
    resource = None
    HAVE_RESOURCE = False

# --- Config ---
DATA_DIR = Path("data")
IMAGENET_INDEX_FILE = Path("imagenet_class_index.json")
MODELS_DIR = Path("models/torch")
OUTPUT_ODS = Path("frm_torch_results.ods")
TOP_K = 5

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CAFFE_MEAN_RGB_255 = torch.tensor([123.68, 116.779, 103.939]).view(1, 3, 1, 1)

TOP_K = 5

def _finite(vals):
    return [v for v in vals if v is not None and not math.isnan(v)]

def nanrobust_max(vals: list) -> float:
    f = _finite(vals)
    return max(f) if f else math.nan

def nanrobust_median(vals: list) -> float:
    f = _finite(vals)
    return float(np.median(f)) if f else math.nan

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
            for d in os.listdir(base):
                if d.startswith("intel-rapl:"):
                    zone_dir = os.path.join(base, d)
                    name_file = os.path.join(zone_dir, "name")
                    if os.path.exists(name_file):
                        with open(name_file, "r") as f:
                            name = f.read().strip()
                        energy_file = os.path.join(zone_dir, "energy_uj")
                        if os.path.exists(energy_file):
                            if name.startswith("package-"):
                                self.pkg_files.append(energy_file)
                            elif name == "dram":
                                self.dram_files.append(energy_file)
            self.ok = bool(self.pkg_files)
        except Exception:
            pass

    def read_energy_j(self) -> Tuple[Optional[float], Optional[float]]:
        # returns (pkg_J_total, dram_J_total) as cumulative counters converted to Joules
        try:
            pkg_total = 0.0
            for f in self.pkg_files:
                with open(f, "r") as fp:
                    pkg_total += float(fp.read().strip()) / 1e6  # µJ → J
            dram_total = 0.0
            for f in self.dram_files:
                with open(f, "r") as fp:
                    dram_total += float(fp.read().strip()) / 1e6
            return (pkg_total, dram_total if self.dram_files else None)
        except Exception:
            return (None, None)

class NVMLPower:
    def __init__(self):
        self.ok = False
        self.handles: List[Any] = []
        if not HAVE_NVML:
            return
        try:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            self.ok = bool(self.handles)
        except Exception:
            pass

    def read_power_w(self) -> Optional[float]:
        try:
            total = 0.0
            for h in self.handles:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
                total += power_mw / 1000.0
            return total
        except Exception:
            return None

    def limits(self) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {
            "gpu_power_limit_w": None,
            "gpu_power_limit_default_w": None,
        }
        try:
            if self.handles:
                out["gpu_power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(self.handles[0])[1] / 1000.0
                out["gpu_power_limit_default_w"] = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handles[0]) / 1000.0
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
            t_start = time.perf_counter()
            host_curr = self._read_host_energy()[0]
            gpu_curr = self._read_gpu_power()
            if host_prev is not None and host_curr is not None:
                host_w = (host_curr - host_prev) / self.dt
                if host_w >= 0: 
                    host_acc_w.append(host_w)
                host_prev = host_curr
            if gpu_curr is not None:
                gpu_acc_w.append(gpu_curr)
            elapsed = time.perf_counter() - t_start
            remaining = max(0, self.dt - elapsed)
            if remaining > 0:
                time.sleep(remaining)
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
                t_start = time.perf_counter()
                self.tstamps.append(t_start)
                gpu_w = self._read_gpu_power()
                if gpu_w is not None:
                    self.gpu_power_w.append(gpu_w)
                elapsed = time.perf_counter() - t_start
                remaining = max(0, self.dt - elapsed)
                if remaining > 0:
                    time.sleep(remaining)
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
            dt_est = (self.tstamps[-1] - self.tstamps[0]) / max(1, len(self.tstamps) - 1)
        else:
            dt_est = self.dt
        gp = np.array(self.gpu_power_w, dtype=float)
        gp = gp[np.isfinite(gp)]
        gpu_j = float(np.sum(gp) * dt_est) if gp.size > 0 else math.nan

        # Power profile statistics over window
        def _stats(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"mean": math.nan, "median": math.nan, "std": math.nan, "p90": math.nan, "p95": math.nan, "p99": math.nan}
            a = np.array(arr, dtype=float)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return {"mean": math.nan, "median": math.nan, "std": math.nan, "p90": math.nan, "p95": math.nan, "p99": math.nan}
            return {
                "mean": float(a.mean()),
                "median": float(np.median(a)),
                "std": float(a.std()) if a.size > 1 else 0.0,
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
            limits = self.nvml.limits()
            info.update(limits)
            info["gpu_count"] = len(self.nvml.handles)
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
                gf = os.path.join(roots, d, "cpufreq", "scaling_governor")
                if os.path.exists(gf):
                    with open(gf, "r") as f:
                        gov = f.read().strip()
                    if gov not in govs:
                        govs.append(gov)
        if govs:
            return ",".join(sorted(govs))
    except Exception:
        pass
    return "unknown"

# ---------- Memory sampling (unchanged from previous) ----------

def _read_linux_vm(value_key: str) -> Optional[int]:
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith(value_key + ":"):
                    return int(line.split()[1]) * 1024  # kB → bytes
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
                return ru  # bytes on macOS
            else:
                return ru * 1024  # kB → bytes on Linux
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
            p = psutil.Process()
            mem = p.memory_info()
            return getattr(mem, "peak_wset", None)
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
        if not init_nvml_if_present():
            return None
        count = pynvml.nvmlDeviceGetCount()
        total_used = 0
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                for p in procs:
                    if p.pid == pid:
                        total_used += p.usedGpuMemory
            except pynvml.NVMLError:
                try:
                    procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(h)
                    for p in procs:
                        if p.pid == pid:
                            total_used += p.usedGpuMemory
                except pynvml.NVMLError:
                    pass
        return total_used if total_used > 0 else None
    except Exception:
        return None

class MemorySampler:
    def __init__(self, hz: int = 300, provider_label: str = ""):
        self.hz = max(200, min(int(hz), 500))
        self.dt = 1.0 / float(self.hz)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.provider_label = provider_label
        
        # Current process PID
        self.pid = os.getpid()
        
        # Buffers
        self.host_rss_mib: List[float] = []
        self.dev_used_mib: List[float] = []
        
        # Baselines
        self.idle_host_mib: Optional[float] = None
        self.postload_host_mib: Optional[float] = None
        self.idle_dev_mib: Optional[float] = None
        self.postload_dev_mib: Optional[float] = None

    def _poll_once(self):
        h = get_host_rss_bytes()
        if h is not None:
            self.host_rss_mib.append(h / (1024 * 1024))
        
        d = nvml_proc_used_bytes(self.pid)
        if d is not None:
            self.dev_used_mib.append(d / (1024 * 1024))

    def start(self):
        self.host_rss_mib.clear()
        self.dev_used_mib.clear()
        self._stop.clear()
        def _runner():
            while not self._stop.is_set():
                t_start = time.perf_counter()
                self._poll_once()
                elapsed = time.perf_counter() - t_start
                remaining = max(0, self.dt - elapsed)
                if remaining > 0:
                    time.sleep(remaining)
        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, float]:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
            self._thread = None
        
        # Final poll
        self._poll_once()
        
        # Host memory stats
        h_arr = np.array(self.host_rss_mib, dtype=float)
        h_arr = h_arr[np.isfinite(h_arr)]
        
        # Device memory stats  
        d_arr = np.array(self.dev_used_mib, dtype=float)
        d_arr = d_arr[np.isfinite(d_arr)]
        
        def _array_stats(arr: np.ndarray) -> Dict[str, float]:
            if arr.size == 0:
                return {"mean": math.nan, "median": math.nan, "max": math.nan, "min": math.nan, "p95": math.nan}
            return {
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "max": float(arr.max()),
                "min": float(arr.min()),
                "p95": float(np.percentile(arr, 95)),
            }
        
        host_stats = _array_stats(h_arr)
        dev_stats = _array_stats(d_arr)
        
        # Clear buffers
        self.host_rss_mib.clear()
        self.dev_used_mib.clear()
        
        return {
            "host_mean_mib": host_stats["mean"],
            "host_median_mib": host_stats["median"],
            "host_max_mib": host_stats["max"],
            "host_min_mib": host_stats["min"],
            "host_p95_mib": host_stats["p95"],
            "dev_mean_mib": dev_stats["mean"],
            "dev_median_mib": dev_stats["median"],
            "dev_max_mib": dev_stats["max"],
            "dev_min_mib": dev_stats["min"],
            "dev_p95_mib": dev_stats["p95"],
        }

    def record_idle_baseline(self):
        h = get_host_rss_bytes()
        self.idle_host_mib = h / (1024 * 1024) if h is not None else None
        d = nvml_proc_used_bytes(self.pid)
        self.idle_dev_mib = d / (1024 * 1024) if d is not None else None

    def record_postload_baseline(self):
        h = get_host_rss_bytes()
        self.postload_host_mib = h / (1024 * 1024) if h is not None else None
        d = nvml_proc_used_bytes(self.pid)
        self.postload_dev_mib = d / (1024 * 1024) if d is not None else None

    def finalize(self) -> Dict[str, Any]:
        final_host = get_host_rss_bytes()
        final_dev = nvml_proc_used_bytes(self.pid)
        os_peak = get_host_peak_bytes_hwm()
        
        return {
            "provider_label": self.provider_label,
            "idle_host_mib": self.idle_host_mib,
            "postload_host_mib": self.postload_host_mib,
            "idle_dev_mib": self.idle_dev_mib,
            "postload_dev_mib": self.postload_dev_mib,
            "final_host_mib": final_host / (1024 * 1024) if final_host else None,
            "final_dev_mib": final_dev / (1024 * 1024) if final_dev else None,
            "os_peak_mib": os_peak / (1024 * 1024) if os_peak else None,
        }

# ---- Robust stats helpers ----

def robust_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": math.nan, "median": math.nan, "std": math.nan, "min": math.nan, "max": math.nan, "p50": math.nan, "p90": math.nan, "p95": math.nan, "p99": math.nan}
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

# --- IO helpers ---

def load_imagenet_class_mapping() -> Dict[str, int]:
    with IMAGENET_INDEX_FILE.open("r") as f:
        class_idx = json.load(f)
    # class_idx: {"0": ["n01440764", "tench"], ...}
    wnid_to_idx = {wnid: int(i) for i, (wnid, _name) in class_idx.items()}
    return wnid_to_idx


def choose_resize(size: int) -> T.Compose:
    return T.Compose([T.Resize(342 if size == 299 else 256), T.CenterCrop(size)])


def load_batch(paths: List[Path], size: int) -> torch.Tensor:
    tf = T.Compose([choose_resize(size), T.ToTensor()])
    batch = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            batch.append(tf(img))
        except Exception:
            batch.append(torch.zeros(3, size, size))
    return torch.stack(batch).float()


# --- Normalizations ---

def imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def caffe_rgb_submean(x_0to1: torch.Tensor) -> torch.Tensor:
    return x_0to1 * 255.0 - CAFFE_MEAN_RGB_255


# --- Builders ---

def build_model_for_file(fname: str) -> torch.nn.Module | None:
    n = fname.lower()
    if n.startswith("resnet18"): return tvm.resnet18(weights=None)
    if n.startswith("resnet50"): return tvm.resnet50(weights=None)
    if n.startswith("squeezenet1_1"): return tvm.squeezenet1_1(weights=None)
    if n.startswith("mobilenet_v2"): return tvm.mobilenet_v2(weights=None)
    if n.startswith("mobilenet_v3_small"): return tvm.mobilenet_v3_small(weights=None)
    if n.startswith("densenet121"): return tvm.densenet121(weights=None)
    if n.startswith("efficientnet_b0"): return tvm.efficientnet_b0(weights=None)
    if n.startswith("inception_v3"): return tvm.inception_v3(weights=None, aux_logits=False, init_weights=False)
    if n.startswith("mnasnet1.0") or n.startswith("mnasnet1_0"): return tvm.mnasnet1_0(weights=None)
    if n.startswith("convnext_tiny"):
        try:
            return tvm.convnext_tiny(weights=None)
        except Exception:
            pass
    if HAVE_TIMM and timm is not None:
        if n.startswith("deit_tiny_distilled_patch16_224"): return timm.create_model("deit_tiny_distilled_patch16_224", pretrained=False, num_classes=1000)
        if n.startswith("levit_128s"): return timm.create_model("levit_128s", pretrained=False, num_classes=1000)
        if n.startswith("mobilevit_xxs"): return timm.create_model("mobilevit_xxs", pretrained=False, num_classes=1000)
    return None


# --- DenseNet remapper ---
_DNET_PATTS = (
    (re.compile(r"\.norm\.(\d+)"), r".norm\1"),
    (re.compile(r"\.conv\.(\d+)"), r".conv\1"),
    (re.compile(r"\.relu\.(\d+)"), r".relu\1"),
    (re.compile(r"\.pool\.(\d+)"), r".pool\1"),
)


def remap_densenet_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        for patt, repl in _DNET_PATTS:
            nk = patt.sub(repl, nk)
        out[nk] = v
    return out


# --- loaders ---

def _load_state_dict_legacy_ok(path: Path):
    last_err = None
    # 1) PyTorch 2.x fast path
    try:
        return torch.load(path, map_location="cpu", weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        pass
    except Exception as e:
        last_err = e
    # 2) Plain
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        last_err = e
    # 3) latin1
    try:
        return torch.load(path, map_location="cpu", encoding="latin1")
    except Exception as e:
        last_err = e
    # 4) Legacy rebuild patch
    try:
        import torch._utils as _tu
        _orig = getattr(_tu, "_rebuild_tensor_v2", None)
        if _orig is not None:
            def _patched(storage, storage_offset, size, stride, requires_grad, backward_hooks):
                return _orig(storage, storage_offset, size, stride, requires_grad, backward_hooks)
            _tu._rebuild_tensor_v2 = _patched  # type: ignore
            try:
                return torch.load(path, map_location="cpu")
            finally:
                _tu._rebuild_tensor_v2 = _orig
    except Exception as e:
        last_err = e
    raise RuntimeError(f"Failed to load checkpoint: {last_err}")


def robust_load_model(model_path: Path, model_name: str, device: torch.device):
    lower = model_name.lower()

    # MNASNet: try state_dict first; fallback to JIT
    if "mnasnet" in lower:
        try:
            sd = _load_state_dict_legacy_ok(model_path)
            if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model", "net", "module")):
                for key in ("state_dict", "model", "net", "module"):
                    if key in sd: sd = sd[key]; break
            if isinstance(sd, dict):
                m = build_model_for_file(model_name)
                if m is None: raise RuntimeError("No MNASNet builder")
                clean = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
                original = m._load_from_state_dict
                def patched(state_dict, prefix, local_metadata, strict, missing, unexpected, errors):
                    md = dict(local_metadata or {})
                    md['version'] = 2
                    return original(state_dict, prefix, md, strict, missing, unexpected, errors)
                m._load_from_state_dict = patched
                try:
                    m.load_state_dict(clean, strict=False)
                finally:
                    m._load_from_state_dict = original
                m.eval(); return m, False
            else:
                raise ValueError("Not a state_dict, trying JIT")
        except Exception as e1:
            try:
                jm = torch.jit.load(str(model_path), map_location=device)
                jm.eval(); return jm, True
            except Exception as e2:
                raise RuntimeError(f"MNASNet load failed - state_dict error: {e1}, JIT error: {e2}")

    # Generic path (incl. DenseNet)
    m = build_model_for_file(model_name)
    if m is None:
        raise RuntimeError(f"No builder for {model_name}.")

    sd = _load_state_dict_legacy_ok(model_path)
    if isinstance(sd, dict) and any(k in sd for k in ("state_dict", "model", "net", "module")):
        for key in ("state_dict", "model", "net", "module"):
            if key in sd: sd = sd[key]; break

    if not isinstance(sd, dict):
        try:
            jm = torch.jit.load(str(model_path), map_location=device)
            jm.eval(); return jm, True
        except Exception as e:
            raise RuntimeError(f"Checkpoint is not a state_dict and JIT failed: {e}")

    if "densenet121" in lower:
        sd = remap_densenet_keys(sd)
        tgt = m.state_dict()
        filtered = {k: v for k, v in sd.items() if k in tgt and tuple(tgt[k].shape) == tuple(v.shape)}
        # Load with strict=False to tolerate tracking buffers
        m.load_state_dict(filtered, strict=False)
        return m, False

    clean = { (k[7:] if k.startswith("module.") else k): v for k, v in sd.items() }
    m.load_state_dict(clean, strict=False)
    return m, False


# --- Eval ---

def expected_input_size(model_name: str) -> int:
    return 299 if "inception_v3" in model_name.lower() else 224

def evaluate_model(
    model_path: Path,
    model_name: str,
    rows: List[Tuple[str, int]],
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
    device: torch.device,
) -> Tuple[pd.DataFrame, float, Optional[float], Dict[str, Any], Dict[str, Any]]:
    """
    One pass over data; memory & energy sampling windows wrap each timed inference.
    Returns per‑sample DF, accs, mem_meta, energy_meta.
    """
    model, _ = robust_load_model(model_path, model_name, device)
    model.eval().to(device)

    # Memory sampler + baselines
    sampler = MemorySampler(hz=mem_hz, provider_label=provider_label) if do_mem else None
    if sampler:
        sampler.record_idle_baseline()

    # Energy sampler + baselines (idle/post‑load power)
    es = EnergySampler(hz=energy_hz) if do_energy else None
    if es:
        es.record_idle_power()

    size = expected_input_size(model_name)

    # postload baselines
    if sampler:
        sampler.record_postload_baseline()
    if es:
        es.record_postload_power()

    img_paths = [DATA_DIR / rel for (rel, _y) in rows]
    x0 = load_batch(img_paths, size)

    n = model_name.lower()
    if n.startswith("resnet50_v1"):
        x = caffe_rgb_submean(x0)
    elif "mobilevit_xxs" in n:
        x = x0.float()
    else:
        x = imagenet_norm(x0)

    target_bs = req_bs if (req_bs and req_bs > 0) else 32
    total = (len(x) // target_bs) * target_bs
    if total < len(x) and req_bs and req_bs > 0:
        print(f"[{model_name}] Truncating {len(x)} → {total} (enforced bs={req_bs})")
    x = x[:total] if total > 0 else x
    if target_bs <= 0: 
        target_bs = 1

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

    def _time_once_with_mem_energy(chunk: torch.Tensor) -> Tuple[float, Dict[str, float], Dict[str, float]]:
        if sampler: sampler.start()
        if es: es.start()
        
        # Ensure GPU operations are synchronized for accurate timing
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(chunk.to(device))
        
        # Synchronize again to ensure inference is complete
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        mem_res = sampler.stop() if sampler else {}
        energy_res = es.stop() if es else {}
        return lat_ms, mem_res, energy_res

    def _warmup(chunk: torch.Tensor, n: int):
        for _ in range(n):
            with torch.no_grad():
                _ = model(chunk.to(device))
        # Synchronize after warmup for GPU
        if device.type == "cuda":
            torch.cuda.synchronize()

    while i < len(x):
        chunk = x[i : i + target_bs]
        if len(chunk) == 0:
            break

        # Warmup
        if do_latency and warmup_batches > 0:
            _warmup(chunk, warmup_batches)

        # Timing + sampling (if enabled)
        if do_latency or do_mem or do_energy:
            times_ms = []
            mem_results = []
            energy_results = []
            
            n_reps = repeats_batch if do_latency else 1
            for _ in range(n_reps):
                lat_ms, mem_res, energy_res = _time_once_with_mem_energy(chunk)
                times_ms.append(lat_ms)
                if mem_res: mem_results.append(mem_res)
                if energy_res: energy_results.append(energy_res)
            
            # Record latency/throughput (median of repeats)
            if do_latency and times_ms:
                lat_median = nanrobust_median(times_ms)
                thr_median = len(chunk) / (lat_median / 1000.0) if lat_median > 0 else math.nan
                per_sample_latency_ms.extend([lat_median] * len(chunk))
                per_sample_throughput_sps.extend([thr_median] * len(chunk))
            
            # Record memory (aggregate of all repeats)
            if do_mem and mem_results:
                host_peaks = [r.get("host_max_mib", math.nan) for r in mem_results]
                host_means = [r.get("host_mean_mib", math.nan) for r in mem_results]
                dev_peaks = [r.get("dev_max_mib", math.nan) for r in mem_results]
                dev_means = [r.get("dev_mean_mib", math.nan) for r in mem_results]
                
                host_peak_med = nanrobust_median(host_peaks)
                host_mean_med = nanrobust_median(host_means)
                dev_peak_med = nanrobust_median(dev_peaks)
                dev_mean_med = nanrobust_median(dev_means)
                
                per_sample_host_peak_mib.extend([host_peak_med] * len(chunk))
                per_sample_host_mean_mib.extend([host_mean_med] * len(chunk))
                per_sample_dev_peak_mib.extend([dev_peak_med] * len(chunk))
                per_sample_dev_mean_mib.extend([dev_mean_med] * len(chunk))
                
                # Steady-state tracking
                if not math.isnan(host_mean_med):
                    steady_host_means.append(host_mean_med)
                if not math.isnan(dev_mean_med):
                    steady_dev_means.append(dev_mean_med)
            
            # Record energy (aggregate of all repeats)
            if do_energy and energy_results:
                host_energies = [r.get("host_pkg_j", math.nan) for r in energy_results]
                gpu_energies = [r.get("gpu_j", math.nan) for r in energy_results]
                dram_energies = [r.get("host_dram_j", math.nan) for r in energy_results]
                
                host_j_med = nanrobust_median(host_energies)
                gpu_j_med = nanrobust_median(gpu_energies)
                dram_j_med = nanrobust_median(dram_energies)
                
                # Total energy (sum available components)
                total_j_components = []
                if not math.isnan(host_j_med):
                    total_j_components.append(host_j_med)
                if not math.isnan(gpu_j_med):
                    total_j_components.append(gpu_j_med)
                if not math.isnan(dram_j_med):
                    total_j_components.append(dram_j_med)
                total_j = sum(total_j_components) if total_j_components else math.nan
                
                per_sample_energy_j.extend([total_j] * len(chunk))
                per_sample_energy_host_j.extend([host_j_med] * len(chunk))
                per_sample_energy_gpu_j.extend([gpu_j_med] * len(chunk))
                per_sample_energy_dram_j.extend([dram_j_med] * len(chunk))
                
                # GPU power stats
                gpu_power_means = [r.get("gpu_power_mean_w", math.nan) for r in energy_results]
                gpu_power_medians = [r.get("gpu_power_median_w", math.nan) for r in energy_results]
                gpu_power_stds = [r.get("gpu_power_std_w", math.nan) for r in energy_results]
                gpu_power_p90s = [r.get("gpu_power_p90_w", math.nan) for r in energy_results]
                gpu_power_p95s = [r.get("gpu_power_p95_w", math.nan) for r in energy_results]
                gpu_power_p99s = [r.get("gpu_power_p99_w", math.nan) for r in energy_results]
                
                per_sample_gpu_power_mean_w.extend([nanrobust_median(gpu_power_means)] * len(chunk))
                per_sample_gpu_power_median_w.extend([nanrobust_median(gpu_power_medians)] * len(chunk))
                per_sample_gpu_power_std_w.extend([nanrobust_median(gpu_power_stds)] * len(chunk))
                per_sample_gpu_power_p90_w.extend([nanrobust_median(gpu_power_p90s)] * len(chunk))
                per_sample_gpu_power_p95_w.extend([nanrobust_median(gpu_power_p95s)] * len(chunk))
                per_sample_gpu_power_p99_w.extend([nanrobust_median(gpu_power_p99s)] * len(chunk))
                
                # Energy per GFLOP
                if gflops_map and model_name in gflops_map:
                    gflops_per_sample = gflops_map[model_name] / len(chunk)  # Assuming GFLOPS is per batch
                    energy_per_gflop = total_j / gflops_per_sample if gflops_per_sample > 0 and not math.isnan(total_j) else math.nan
                    per_sample_energy_per_gflop.extend([energy_per_gflop] * len(chunk))
                else:
                    per_sample_energy_per_gflop.extend([math.nan] * len(chunk))
        
        # Get predictions (separate from timing to avoid interference)
        with torch.no_grad():
            y = model(chunk.to(device))
            if isinstance(y, (list, tuple)):
                y = y[0]
            
            # Synchronize for GPU operations
            if device.type == "cuda":
                torch.cuda.synchronize()
                
            t1 = torch.argmax(y, dim=1).tolist()
            t5_indices = torch.topk(y, k=TOP_K, dim=1).indices.cpu().numpy()
            
            top1_all.extend(t1)
            for row in t5_indices:
                top5_all.append(row)

        # Fill missing measurement arrays if not doing those measurements
        if not do_latency:
            per_sample_latency_ms.extend([math.nan] * len(chunk))
            per_sample_throughput_sps.extend([math.nan] * len(chunk))
        
        if not do_mem:
            per_sample_host_peak_mib.extend([math.nan] * len(chunk))
            per_sample_host_mean_mib.extend([math.nan] * len(chunk))
            per_sample_dev_peak_mib.extend([math.nan] * len(chunk))
            per_sample_dev_mean_mib.extend([math.nan] * len(chunk))
        
        if not do_energy:
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

        pbar.update(len(chunk))
        i += target_bs

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
    if do_mem:
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
        mem_meta["steady_host_mean_median_mib"] = nanrobust_median(steady_host_means)
        mem_meta["steady_dev_mean_median_mib"] = nanrobust_median(steady_dev_means)

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


# --- Writer ---

def save_ods_multisheet(path: Path, sheets: Dict[str, pd.DataFrame], summary_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="odf") as writer:
        for name, df in sheets.items():
            (df if (df is not None and not df.empty) else pd.DataFrame({"note": ["empty or failed"]})).to_excel(
                writer, sheet_name=name[:31], index=False
            )
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

def load_gflops_map(csv_path: Optional[str]) -> Optional[Dict[str, float]]:
    if not csv_path:
        return None
    if not os.path.exists(csv_path):
        print(f"[Warning] GFLOPS map file not found: {csv_path}")
        return None
    mp: Dict[str, float] = {}
    try:
        df = pd.read_csv(csv_path)
        if "model_name" not in df.columns or "gflops" not in df.columns:
            print(f"[Warning] GFLOPS map CSV must have columns 'model_name' and 'gflops'")
            return None
        for _, row in df.iterrows():
            model_name = str(row["model_name"]).strip()
            gflops = float(row["gflops"])
            if model_name and not math.isnan(gflops) and gflops > 0:
                mp[model_name] = gflops
        print(f"[GFLOPS] Loaded {len(mp)} model mappings from {csv_path}")
        return mp
    except Exception as e:
        print(f"[Warning] Failed to load GFLOPS map: {e}")
        return None


# --- Main ---
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FRM Torch eval (single-pass acc+latency+memory+energy, bs-enforced)")
    p.add_argument("--blaze", action="store_true", help="Run only 10 random samples")
    p.add_argument("--bs", type=int, default=0, help="Batch size to enforce. 0=auto (fixed N or 32)")
    p.add_argument("--latency", action="store_true", help="Measure latency/throughput in the same pass")
    p.add_argument("--lat-warmup-batches", type=int, default=2, help="Warmup runs per batch (not recorded)")
    p.add_argument("--lat-repeats-batch", type=int, default=10, help="Timed repeats per batch; median used")
    p.add_argument("--mem", action="store_true", help="Enable memory sampling & reporting (host+device)")
    p.add_argument("--mem-sample-hz", type=int, default=300, help="Sampler frequency (Hz), 200–500 recommended")
    p.add_argument("--energy", action="store_true", help="Enable power/energy sampling (host RAPL + GPU NVML)")
    p.add_argument("--energy-sample-hz", type=int, default=300, help="Energy sampler frequency (Hz), ≥200 recommended")
    p.add_argument("--gflops-map", type=str, default="", help="CSV with columns [model_name,gflops] for J/GFLOP")
    args = p.parse_args()

    random.seed(1337)
    np.random.seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backend_info = f"PyTorch {torch.__version__} on {device}"
    print(f"[System] Using device: {device}")
    print(f"[Batch] {'Enforcing' if (args.bs and args.bs>0) else 'Auto'} batch size: {args.bs if args.bs else 'auto(32)'}")

    gflops_map = load_gflops_map(args.gflops_map)

    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    if args.blaze:
        manifest = manifest.sample(n=min(10, len(manifest)), random_state=0).reset_index(drop=True)

    if "correct_label" not in manifest.columns:
        wnid_to_idx = load_imagenet_class_mapping()
        if "wnid" in manifest.columns:
            manifest["correct_label"] = manifest["wnid"].map(wnid_to_idx).astype(int)
        elif "class_id_1to1000" in manifest.columns:
            manifest["correct_label"] = manifest["class_id_1to1000"].astype(int) - 1
        else:
            raise RuntimeError("manifest must contain 'wnid' or 'correct_label'")

    rows: List[Tuple[str, int]] = [(str(r.rel_path), int(r.correct_label)) for r in manifest.itertuples(index=False)]

    files = [f for f in os.listdir(MODELS_DIR) if f.lower().endswith((".pth", ".pt"))]
    if not files:
        raise SystemExit(f"No .pth/.pt models found in {MODELS_DIR}")

    all_sheets: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Dict[str, Any]] = []

    # Run info + Memory/Power backend sheet
    runinfo = pd.DataFrame(
        {
            "key": [
                "hostname",
                "platform", 
                "python_version",
                "torch_version",
                "backend_info",
                "psutil",
                "nvml_available",
                "device",
            ],
            "value": [
                platform.node(),
                platform.platform(),
                platform.python_version(),
                torch.__version__,
                backend_info,
                str(HAVE_PSUTIL),
                str(HAVE_NVML),
                str(device),
            ],
        }
    )
    all_sheets["RunInfo"] = runinfo

    # Optional: Model statics sheet
    model_statics_rows = []

    # We'll also capture one environment disclosure from energy sampler (if enabled)
    power_env_sheet: Optional[pd.DataFrame] = None

    for fname in tqdm(files, desc="Models", ncols=90):
        mpath = MODELS_DIR / fname
        try:
            df, top1, top5, mem_meta, energy_meta = evaluate_model(
                mpath, fname, rows, args.bs,
                args.latency, args.lat_warmup_batches, args.lat_repeats_batch,
                args.mem, args.mem_sample_hz, f"PyTorch-{device}",
                args.energy, args.energy_sample_hz, gflops_map, device
            )
            all_sheets[fname] = df

            # Model file size
            model_file_mb = mpath.stat().st_size / (1024 * 1024)

            # Build summary row
            row: Dict[str, Any] = {
                "model_name": fname,
                "top1_acc": top1,
                "top5_acc": top5,
                "samples": len(df),
                "model_file_mb": model_file_mb,
            }

            # Latency/throughput stats
            if args.latency and "latency_ms" in df.columns:
                lat_vals = [v for v in df["latency_ms"].tolist() if not math.isnan(v)]
                thr_vals = [v for v in df["throughput_sps"].tolist() if not math.isnan(v)]
                lat_stats = robust_stats(lat_vals)
                thr_stats = robust_stats(thr_vals)
                row.update({f"lat_{k}": v for k, v in lat_stats.items()})
                row.update({f"thr_{k}": v for k, v in thr_stats.items()})

            # Memory stats
            if args.mem and mem_meta:
                # Per-sample memory deltas
                if "host_peak_mib" in df.columns:
                    baseline_host = mem_meta.get("postload_host_mib", 0) or 0
                    baseline_dev = mem_meta.get("postload_dev_mib", 0) or 0
                    
                    host_deltas = [(v - baseline_host) for v in df["host_peak_mib"].tolist() if not math.isnan(v)]
                    dev_deltas = [(v - baseline_dev) for v in df["dev_peak_mib"].tolist() if not math.isnan(v)]
                    
                    if host_deltas:
                        host_delta_stats = robust_stats(host_deltas)
                        row.update({f"host_peak_delta_{k}_mib": v for k, v in host_delta_stats.items()})
                    
                    if dev_deltas:
                        dev_delta_stats = robust_stats(dev_deltas)
                        row.update({f"dev_peak_delta_{k}_mib": v for k, v in dev_delta_stats.items()})

                # Baseline memory values
                row.update({
                    "baseline_idle_host_mib": mem_meta.get("idle_host_mib"),
                    "baseline_postload_host_mib": mem_meta.get("postload_host_mib"),
                    "baseline_idle_dev_mib": mem_meta.get("idle_dev_mib"),
                    "baseline_postload_dev_mib": mem_meta.get("postload_dev_mib"),
                    "host_global_peak_mib": mem_meta.get("final_host_mib"),
                    "host_true_os_peak_mib": mem_meta.get("os_peak_mib"),
                    "steady_host_mean_delta_mib_median": mem_meta.get("steady_host_mean_median_mib"),
                    "steady_dev_mean_delta_mib_median": mem_meta.get("steady_dev_mean_median_mib"),
                    "mem_device_backend": mem_meta.get("provider_label"),
                })

            # Energy stats
            if args.energy and "energy_j" in df.columns:
                energy_vals = [v for v in df["energy_j"].tolist() if not math.isnan(v)]
                energy_host_vals = [v for v in df["energy_host_j"].tolist() if not math.isnan(v)]
                energy_gpu_vals = [v for v in df["energy_gpu_j"].tolist() if not math.isnan(v)]
                energy_dram_vals = [v for v in df["energy_dram_j"].tolist() if not math.isnan(v)]
                energy_per_gflop_vals = [v for v in df["energy_per_gflop"].tolist() if not math.isnan(v)]
                
                gpu_power_mean_vals = [v for v in df["gpu_power_mean_w"].tolist() if not math.isnan(v)]
                gpu_power_median_vals = [v for v in df["gpu_power_median_w"].tolist() if not math.isnan(v)]
                gpu_power_p95_vals = [v for v in df["gpu_power_p95_w"].tolist() if not math.isnan(v)]

                if energy_vals:
                    energy_stats = robust_stats(energy_vals)
                    row.update({f"energy_{k}_j_per_sample": v for k, v in energy_stats.items()})
                
                row.update({
                    "energy_host_median_j_per_sample": nanrobust_median(energy_host_vals),
                    "energy_gpu_median_j_per_sample": nanrobust_median(energy_gpu_vals),
                    "energy_dram_median_j_per_sample": nanrobust_median(energy_dram_vals),
                    "energy_mean_j_per_gflop": nanrobust_median(energy_per_gflop_vals),
                    "energy_median_j_per_gflop": nanrobust_median(energy_per_gflop_vals),
                    "gpu_power_mean_w_median": nanrobust_median(gpu_power_mean_vals),
                    "gpu_power_median_w_median": nanrobust_median(gpu_power_median_vals),
                    "gpu_power_p95_w_median": nanrobust_median(gpu_power_p95_vals),
                })

                # Power environment info (capture once)
                if power_env_sheet is None and energy_meta:
                    power_env_data = []
                    for k, v in energy_meta.items():
                        power_env_data.append({"key": k, "value": str(v)})
                    power_env_sheet = pd.DataFrame(power_env_data)

            summary_rows.append(row)

            # Model statics
            model_statics_rows.append({
                "model_name": fname,
                "model_file_mb": model_file_mb,
            })
            
            # Clean up GPU memory after each model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        except Exception as e:
            msg = str(e)
            all_sheets[fname] = pd.DataFrame({"note": [msg]})
            summary_rows.append({
                "model_name": fname, 
                "top1_acc": math.nan, 
                "top5_acc": math.nan, 
                "samples": 0,
                "model_file_mb": math.nan,
            })
            
            # Clean up GPU memory even on error
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if model_statics_rows:
        all_sheets["ModelStatics"] = pd.DataFrame(model_statics_rows)

    if power_env_sheet is not None:
        all_sheets["PowerEnv"] = power_env_sheet

    sum_df = pd.DataFrame(summary_rows)
    # Column order preference (matching ONNX script)
    pref = [
        "model_name",
        "top1_acc", 
        "top5_acc",
        "samples",
        "model_file_mb",
        "lat_mean",
        "lat_median",
        "lat_std",
        "lat_min",
        "lat_max",
        "lat_p50",
        "lat_p90",
        "lat_p95",
        "lat_p99",
        "thr_mean",
        "thr_median",
        "thr_std",
        "thr_min",
        "thr_max",
        "thr_p50",
        "thr_p90",
        "thr_p95",
        "thr_p99",
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
    print(f"Saved: {OUTPUT_ODS}")