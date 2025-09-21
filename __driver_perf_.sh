#!/usr/bin/env bash
# Performance Benchmark Driver CPU/GPU Compute and Memory Bandwidth
# Usage: bash __driver_perf_.sh [--cpu] [--gpu] [--run N] [--tf32] [--gemm-n N] [--output output.json]
set -euo pipefail

trap 'echo -e "[\e[31m$(date +'%H:%M:%S')\e[0m] ❌ Error on line $LINENO"; exit 1' ERR

# --- Parse args ---
USE_CPU=0
USE_GPU=0
RUN_COUNT=5
OUTPUT_FILE=""
DETERMINISTIC_MODE=0
GEMM_SIZE=8192
ENABLE_TF32=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) 
      USE_CPU=1
      shift ;;
    --gpu)
      USE_GPU=1
      shift ;;
    --deterministic)
      DETERMINISTIC_MODE=1
      shift ;;
    --tf32)
      ENABLE_TF32=1
      shift ;;
    --gemm-n)
      if [[ $# -lt 2 ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: --gemm-n requires a positive integer" >&2
        exit 2
      fi
      GEMM_SIZE="$2"
      shift 2 ;;
    --run)
      if [[ $# -lt 2 ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: --run requires a positive integer" >&2
        exit 2
      fi
      RUN_COUNT="$2"
      shift 2 ;;
    --output)
      if [[ $# -lt 2 ]]; then
        echo "Error: --output requires a filename" >&2
        exit 2
      fi
      OUTPUT_FILE="$2"
      shift 2 ;;
    *) 
      echo "Unknown arg: $1" >&2
      echo "Usage: bash __driver_perf_.sh [--cpu] [--gpu] [--deterministic] [--tf32] [--gemm-n N] [--run N] [--output output.json]" >&2
      exit 2 ;;
  esac
done

# If no device specified, enable both
if [[ $USE_CPU -eq 0 && $USE_GPU -eq 0 ]]; then
  USE_CPU=1
  USE_GPU=1
fi

# Default output file
if [[ -z "$OUTPUT_FILE" ]]; then
  OUTPUT_FILE="perf_benchmark_$(date +'%Y%m%dT%H%M%SZ').json"
fi

# --- Helper functions ---
log()  { echo -e "[\e[32m$(date +'%H:%M:%S')\e[0m] $*"; }
warn() { echo -e "[\e[33m$(date +'%H:%M:%S')\e[0m] ⚠ $*" >&2; }

# --- System detection ---
OS_ID="unknown"; OS_VER=""; ARCH="$(uname -m)"
if [ -f /etc/os-release ]; then . /etc/os-release || true; OS_ID="${ID:-unknown}"; OS_VER="${VERSION_ID:-}"; fi
HAS_APT=0; command -v apt-get  >/dev/null 2>&1 && HAS_APT=1
HAS_DNF=0; command -v dnf      >/dev/null 2>&1 && HAS_DNF=1
HAS_PAC=0; command -v pacman   >/dev/null 2>&1 && HAS_PAC=1
HAS_NVIDIA=0; command -v nvidia-smi >/dev/null 2>&1 && HAS_NVIDIA=1
HAS_ROCM=0; command -v rocm-smi >/dev/null 2>&1 && HAS_ROCM=1
IS_PI=0; grep -qi 'raspberry pi' /proc/cpuinfo 2>/dev/null && IS_PI=1

log "Performance benchmark configuration:"
log "CPU: $USE_CPU | GPU: $USE_GPU | Runs: $RUN_COUNT | GEMM size: $GEMM_SIZE | Output: $OUTPUT_FILE"
log "Mode: $([ $DETERMINISTIC_MODE -eq 1 ] && echo "DETERMINISTIC (reproducible)" || echo "PEAK (fast kernels)")"
log "GPU precision: $([ $ENABLE_TF32 -eq 1 ] && echo "TF32 (tensor cores)" || echo "FP32-exact")"
log "OS: $OS_ID $OS_VER | ARCH: $ARCH | NVIDIA: $HAS_NVIDIA | ROCm: $HAS_ROCM | RPi: $IS_PI"

# --- Setup paths and environments ---
BASE_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
ROOT="$(pwd)"
SETUP_DIR="$ROOT/perf_setup_$BASE_TS"
VENV_PERF="$SETUP_DIR/.venv_perf"

mkdir -p "$SETUP_DIR"

# --- Device info and hygiene functions (adapted from __driver.sh) ---
create_device_info_script() {
cat > "$SETUP_DIR/device_info_collect.py" << 'PY'
import json, os, platform, subprocess, sys
from datetime import datetime

def sh(cmd):
  try:
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
  except Exception as e:
    return f"__ERR__ {e}"

out = {
  "timestamp_utc": datetime.utcnow().isoformat()+"Z",
  "python": sys.version,
  "platform": platform.platform(),
  "machine": platform.machine(),
  "processor": platform.processor(),
  "uname": sh("uname -a"),
}

if os.path.exists("/etc/os-release"):
  out["os_release"] = open("/etc/os-release").read()

out["lscpu"] = sh("lscpu")
p="/sys/devices/system/cpu/smt/control"
out["smt_control"] = open(p).read().strip() if os.path.exists(p) else ""

govs={}
for root,_,files in os.walk("/sys/devices/system/cpu"):
  if "cpufreq" in root and "scaling_governor" in files:
    try: 
      govs[root]=open(os.path.join(root,"scaling_governor")).read().strip()
    except: 
      pass
out["cpu_governors"]=govs
out["meminfo"]=sh("cat /proc/meminfo")
out["lspci_vga"]=sh("lspci | grep -i -E 'vga|3d|display'")

try:
  import nvidia_ml_py3 as pynvml
  pynvml.nvmlInit()
  n=pynvml.nvmlDeviceGetCount()
  g=[]
  for i in range(n):
    h=pynvml.nvmlDeviceGetHandleByIndex(i)
    name=pynvml.nvmlDeviceGetName(h)
    if isinstance(name,bytes): 
      name=name.decode()
    mem=pynvml.nvmlDeviceGetMemoryInfo(h)
    try:
      e=float(pynvml.nvmlDeviceGetEnforcedPowerLimit(h))/1000.0
      mn,mx=pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
      pl={"enforced_w":e,"min_w":mn/1000.0,"max_w":mx/1000.0}
    except Exception: 
      pl={}
    g.append({"index":i,"name":name,"memory_total_mb":int(mem.total/1048576),"power_limits":pl})
  out["nvidia_gpus"]=g
except Exception as e:
  out["nvidia_gpus"]=f"not_available: {e}"

out["versions_perf_env"]={}
for m in ["numpy","torch","nvidia_ml_py3"]:
  try:
    mod=__import__(m)
    out["versions_perf_env"][m]=getattr(mod,"__version__","unknown")
  except Exception: 
    out["versions_perf_env"][m]="not_installed"

# Add CUDA/cuBLAS/cuDNN version information
try:
  import torch
  cuda_info = {
    "cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "not_available",
    "cuda_available": torch.cuda.is_available()
  }
  
  # Try to get cuBLAS version (advanced)
  try:
    import ctypes
    import ctypes.util
    cublas_lib = ctypes.util.find_library('cublas')
    if cublas_lib:
      lib = ctypes.CDLL(cublas_lib)
      if hasattr(lib, 'cublasGetVersion'):
        version = ctypes.c_int()
        lib.cublasGetVersion(ctypes.byref(version))
        cuda_info["cublas_version"] = version.value
      else:
        cuda_info["cublas_version"] = "version_func_not_found"
    else:
      cuda_info["cublas_version"] = "library_not_found"
  except Exception as e:
    cuda_info["cublas_version"] = f"detection_failed: {e}"
    
  out["cuda_runtime_info"] = cuda_info
except Exception as e:
  out["cuda_runtime_info"] = f"torch_not_available: {e}"

print(json.dumps(out, indent=2))
PY
}

create_hygiene_script() {
cat > "$SETUP_DIR/hygiene_apply_and_snapshot.py" << 'PY'
import json, os, subprocess, re, sys
from datetime import datetime

def sh(cmd):
  try: 
    return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
  except Exception as e: 
    return f"__ERR__ {e}"

snap={"timestamp_utc": datetime.utcnow().isoformat()+"Z"}

# Check if deterministic mode is requested
deterministic_mode = len(sys.argv) > 1 and sys.argv[1] == "deterministic"
snap["deterministic_mode"] = deterministic_mode

# Set environment variables
if deterministic_mode:
    # Deterministic mode: enable all constraints for reproducibility
    snap["mode"] = "deterministic_reproducible"
    os.environ.update({
      "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1",
      "CUDA_LAUNCH_BLOCKING":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8",
      "PYTHONHASHSEED":"0","CUDA_DETERMINISTIC_OPS":"1","CUBLAS_DETERMINISTIC":"1"
    })
else:
    # Peak mode: do NOT force deterministic/cuBLAS constraints; allow fast kernels
    snap["mode"] = "peak_performance"
    os.environ.update({"PYTHONHASHSEED":"0"})

# Set CPU governors to performance (best effort)
for root,_,files in os.walk('/sys/devices/system/cpu'):
  if root.endswith('cpufreq') and 'scaling_governor' in files:
    try: 
      open(os.path.join(root,'scaling_governor'),'w').write('performance')
    except: 
      pass

# Turn off SMT (best effort)
try: 
  open('/sys/devices/system/cpu/smt/control','w').write('off')
except: 
  pass

# NVIDIA: persistence mode + lock clocks (no sudo; assume root)
if os.system("nvidia-smi >/dev/null 2>&1")==0:
  snap["nvidia_pm_on"]=sh('nvidia-smi -pm 1')
  sup=sh('nvidia-smi -q -d SUPPORTED_CLOCKS')
  snap["nvidia_supported"]=sup
  mem=re.findall(r"Memory\s*:\s*(\d+) MHz", sup) or ['']
  gr=re.findall(r"Graphics\s*:\s*(\d+) MHz", sup) or ['']
  if mem[-1] and gr[-1]:
    snap["lock_mem_clock"]=sh(f"nvidia-smi -lmc {mem[-1]}")
    snap["lock_gfx_clock"]=sh(f"nvidia-smi -lgc {gr[-1]},{gr[-1]}")
  
  # Record current clocks for monitoring
  snap["nvidia_clocks_pre"]=sh('nvidia-smi --query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw --format=csv,noheader')
  
  # Record ECC status and power limits (critical for paper results)
  snap["nvidia_ecc_power"]=sh('nvidia-smi -q -d ECC,POWER | sed -n "1,200p"')

# Check NUMA topology
snap["numa_topology"]=sh('numactl -H 2>/dev/null || echo "numactl not available"')

snap["ps_aux"]=sh('ps aux')
try: 
  snap["smt_control"]=open('/sys/devices/system/cpu/smt/control').read().strip()
except: 
  snap["smt_control"]=""
snap["governors"]=sh("bash -lc 'grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true'")
snap["nvidia_smi"]=sh('nvidia-smi')

print(json.dumps(snap, indent=2))
PY
}

# --- CPU performance benchmark functions ---
create_cpu_benchmark_scripts() {
  # STREAM Triad for memory bandwidth
cat > "$SETUP_DIR/stream.c" << 'C'
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N (80*1024*1024) // ~80M elements ~= 305 MB per array
double a[N], b[N], c[N];
int main(){
  #pragma omp parallel for
  for (long i=0;i<N;i++){ b[i]=1.0; c[i]=2.0; }
  
  double best_bw = 0.0;
  for(int k=0;k<15;k++){
    double t0=omp_get_wtime();
    #pragma omp parallel for
    for (long i=0;i<N;i++) a[i]=b[i]+3.0*c[i]; // STREAM triad
    double t=omp_get_wtime()-t0;
    double bytes = (sizeof(double)* (1 /*write a*/ + 1 /*read b*/ + 1 /*read c*/)) * (double)N;
    double bw = bytes/t/1e9;
    if (bw > best_bw) best_bw = bw;
    printf("Triad: %.2f GB/s\n", bw);
  }
  printf("BEST_BW: %.2f\n", best_bw);
}
C

  # CPU SGEMM for compute performance
cat > "$SETUP_DIR/cpu_sgemm_bench.py" << 'PY'
import numpy as np
import time
import json
import sys
import os

# Enable multi-threading for peak performance
nthreads = os.cpu_count()
os.environ.setdefault("OMP_NUM_THREADS", str(nthreads))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(nthreads))
os.environ.setdefault("MKL_NUM_THREADS", str(nthreads))
try:
    import mkl
    mkl.set_num_threads(nthreads)
except Exception:
    pass

# Record BLAS vendor information for paper reviewers
try:
    import io, contextlib, numpy as np, numpy.__config__ as cfg
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try: 
            cfg.show()
        except Exception: 
            pass
    blas_info = buf.getvalue() or 'unknown'
except Exception: 
    blas_info = 'unknown'

def benchmark_cpu_sgemm(n=8192, warmup=3, iters=10):
    """Benchmark CPU SGEMM performance"""
    results = []
    
    # Create matrices
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    
    # Record configuration
    config_info = {
        "omp_threads": os.environ.get("OMP_NUM_THREADS", "unknown"),
        "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unknown"),
        "mkl_threads": os.environ.get("MKL_NUM_THREADS", "unknown"),
        "cpu_count": os.cpu_count(),
        "blas_info": blas_info
    }
    
    # Warmup
    for _ in range(warmup):
        _ = a @ b
    
    # Benchmark
    best_tflops = 0.0
    timings = []
    for i in range(iters):
        t0 = time.perf_counter()
        c = a @ b
        t = time.perf_counter() - t0
        
        flops = 2 * (n ** 3)  # multiply + add
        tflops = flops / t / 1e12
        if tflops > best_tflops:
            best_tflops = tflops
        
        timings.append(t)
        results.append({
            "iteration": i + 1,
            "time_seconds": t,
            "tflops": tflops
        })
        
        print(f"CPU SGEMM iter {i+1}: {tflops:.3f} TFLOP/s")
    
    print(f"BEST_TFLOPS: {best_tflops:.3f}")
    
    # Calculate per-run confidence interval
    import statistics, math
    mean_tflops = statistics.mean([r["tflops"] for r in results])
    if len(results) > 1:
        stdev_tflops = statistics.stdev([r["tflops"] for r in results])
        ci95_single_run = 1.96 * stdev_tflops / math.sqrt(len(results))
    else:
        stdev_tflops = 0.0
        ci95_single_run = 0.0
    
    return {
        "matrix_size": n,
        "dtype": "float32",
        "precision_label": "FP32-exact",
        "threading_config": config_info,
        "warmup_iterations": warmup,
        "benchmark_iterations": iters,
        "results": results,
        "timings_raw": timings,
        "best_tflops": best_tflops,
        "mean_tflops": mean_tflops,
        "std_tflops": stdev_tflops,
        "ci95_single_run": ci95_single_run,
        "units": {"compute": "TFLOP/s", "time": "seconds"}
    }

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    result = benchmark_cpu_sgemm(n)
    print("CPU_SGEMM_JSON:", json.dumps(result))
PY
}

# --- GPU performance benchmark functions ---
create_gpu_benchmark_scripts() {
  # GPU bandwidth and compute benchmark using PyTorch
cat > "$SETUP_DIR/gpu_perf_bench.py" << 'PY'
import torch
import time
import json
import sys
import subprocess
import os

# Configure precision based on command line argument
enable_tf32 = len(sys.argv) > 3 and sys.argv[3] == "enable_tf32"

if enable_tf32:
    # Enable TF32 for tensor core performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except:
        pass  # Older PyTorch versions
    PRECISION_LABEL = "TF32 (tensor cores)"
else:
    # Force FP32 exact precision (disable TF32 on Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    PRECISION_LABEL = "FP32-exact"

def tsec(fn, warmup=3, iters=10):
    """Time a function with warmup"""
    for _ in range(warmup): 
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    t0 = time.perf_counter()
    for _ in range(iters): 
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    return (time.perf_counter() - t0) / iters

def get_gpu_clocks():
    """Get current GPU clocks and thermals"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unavailable"

def benchmark_gpu_sgemm(n=8192, warmup=3, iters=10):
    """Benchmark GPU SGEMM performance"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device("cuda")
    
    # Record GPU state before benchmark
    clocks_pre = get_gpu_clocks()
    
    # Create matrices (cache-bust with fresh random data each time)
    A = torch.randn((n, n), device=device, dtype=torch.float32)
    B = torch.randn((n, n), device=device, dtype=torch.float32)
    
    results = []
    best_tflops = 0.0
    timings = []
    
    def gpu_gemm(): 
        return A @ B
    
    # Benchmark
    for i in range(iters):
        dt = tsec(gpu_gemm, warmup=(warmup if i == 0 else 1), iters=1)
        flops = 2 * (n ** 3)
        tflops = flops / dt / 1e12
        if tflops > best_tflops:
            best_tflops = tflops
        
        timings.append(dt)
        results.append({
            "iteration": i + 1,
            "time_seconds": dt,
            "tflops": tflops
        })
        
        print(f"GPU SGEMM iter {i+1}: {tflops:.3f} TFLOP/s")
    
    # Record GPU state after benchmark
    clocks_post = get_gpu_clocks()
    
    print(f"BEST_GPU_TFLOPS: {best_tflops:.3f}")
    
    # Calculate per-run confidence interval
    import statistics, math
    mean_tflops = sum(r["tflops"] for r in results) / len(results)
    if len(results) > 1:
        stdev_tflops = (sum((r["tflops"] - mean_tflops)**2 for r in results) / len(results))**0.5
        ci95_single_run = 1.96 * stdev_tflops / math.sqrt(len(results))
    else:
        stdev_tflops = 0.0
        ci95_single_run = 0.0
    
    return {
        "matrix_size": n,
        "dtype": "float32",
        "precision_label": PRECISION_LABEL,
        "device": str(device),
        "gpu_clocks_pre": clocks_pre,
        "gpu_clocks_post": clocks_post,
        "warmup_iterations": warmup,
        "benchmark_iterations": iters,
        "results": results,
        "timings_raw": timings,
        "best_tflops": best_tflops,
        "mean_tflops": mean_tflops,
        "std_tflops": stdev_tflops,
        "ci95_single_run": ci95_single_run,
        "units": {"compute": "TFLOP/s", "time": "seconds"}
    }

def benchmark_gpu_bandwidth(mbytes=8*1024**3, warmup=3, iters=20):
    """Benchmark GPU memory bandwidth using D2D copy"""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    device = torch.device("cuda")
    
    # Record GPU state before benchmark
    clocks_pre = get_gpu_clocks()
    
    # Create tensors for D2D copy (cache-bust with two pairs)
    X0 = torch.empty(mbytes // 8, device=device, dtype=torch.float32)  # Use 8 for two pairs
    Y0 = torch.empty_like(X0)
    X1 = torch.empty_like(X0)
    Y1 = torch.empty_like(X0)
    
    # For H2D/D2H
    H = torch.empty(mbytes // 4, device='cpu', pin_memory=True, dtype=torch.float32)
    X_full = torch.empty_like(H, device=device)
    
    toggle = [0]
    def d2d(): 
        # Alternate between two pairs to avoid cache effects (deterministic toggle)
        if toggle[0] == 0:
            Y0.copy_(X0, non_blocking=True)
        else:
            Y1.copy_(X1, non_blocking=True)
        toggle[0] ^= 1
    
    def h2d(): 
        X_full.copy_(H, non_blocking=True)
    def d2h(): 
        H.copy_(X_full, non_blocking=True)
    
    # Benchmark D2D (device-to-device) - effective bandwidth for 4x smaller tensors
    dt_d2d = tsec(d2d, warmup, iters)
    bw_d2d = (mbytes // 2) / dt_d2d / 1e9  # Adjust for smaller tensor size
    
    # Benchmark H2D and D2H (PCIe)
    dt_h2d = tsec(h2d, warmup, iters)
    dt_d2h = tsec(d2h, warmup, iters)
    bw_h2d = mbytes / dt_h2d / 1e9
    bw_d2h = mbytes / dt_d2h / 1e9
    
    # Record GPU state after benchmark
    clocks_post = get_gpu_clocks()
    
    print(f"GPU D2D: {bw_d2d:.1f} GB/s")
    print(f"GPU H2D: {bw_h2d:.1f} GB/s")
    print(f"GPU D2H: {bw_d2h:.1f} GB/s")
    print(f"BEST_GPU_BW: {bw_d2d:.1f}")
    
    return {
        "memory_size_bytes": mbytes,
        "d2d_bandwidth_gbs": bw_d2d,
        "h2d_bandwidth_gbs": bw_h2d,
        "d2h_bandwidth_gbs": bw_d2h,
        "best_bandwidth_gbs": bw_d2d,
        "gpu_clocks_pre": clocks_pre,
        "gpu_clocks_post": clocks_post,
        "warmup_iterations": warmup,
        "benchmark_iterations": iters,
        "units": {"bandwidth": "GB/s (1e9)"}
    }

def run_cuda_bandwidthtest():
    """Try to run CUDA bandwidthTest if available"""
    cuda_paths = [
        "/usr/local/cuda/samples/1_Utilities/bandwidthTest/bandwidthTest",
        "/usr/local/cuda-*/samples/1_Utilities/bandwidthTest/bandwidthTest"
    ]
    
    clocks_pre = get_gpu_clocks()
    
    for path in cuda_paths:
        if os.path.exists(path):
            try:
                result = subprocess.run([path], capture_output=True, text=True, timeout=60)
                clocks_post = get_gpu_clocks()
                return {
                    "stdout": result.stdout, 
                    "stderr": result.stderr, 
                    "returncode": result.returncode,
                    "gpu_clocks_pre": clocks_pre,
                    "gpu_clocks_post": clocks_post
                }
            except Exception as e:
                return {"error": str(e), "gpu_clocks_pre": clocks_pre}
    
    return {"error": "CUDA bandwidthTest not found", "gpu_clocks_pre": clocks_pre}

if __name__ == "__main__":
    test_type = sys.argv[1] if len(sys.argv) > 1 else "all"
    
    if test_type == "sgemm":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
        result = benchmark_gpu_sgemm(n)
        print("GPU_SGEMM_JSON:", json.dumps(result))
    elif test_type == "bandwidth":
        mbytes = int(sys.argv[2]) if len(sys.argv) > 2 else 8*1024**3
        result = benchmark_gpu_bandwidth(mbytes)
        print("GPU_BW_JSON:", json.dumps(result))
    elif test_type == "cuda_bw":
        result = run_cuda_bandwidthtest()
        print("CUDA_BW_JSON:", json.dumps(result))
    else:
        # Run all tests
        sgemm_result = benchmark_gpu_sgemm()
        bw_result = benchmark_gpu_bandwidth()
        cuda_bw_result = run_cuda_bandwidthtest()
        
        result = {
            "sgemm": sgemm_result,
            "bandwidth": bw_result,
            "cuda_bandwidthtest": cuda_bw_result
        }
        print("GPU_ALL_JSON:", json.dumps(result))
PY
}

# --- System packages installation ---
install_sysdeps() {
  log "Installing system packages…"
  if [ $HAS_APT -eq 1 ]; then
    apt-get update -y || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-venv python3-pip python3-dev build-essential \
      gcc g++ gfortran libomp-dev \
      git wget curl unzip pkg-config \
      libopenblas-dev liblapack-dev \
      pciutils dmidecode hwloc jq || true
  elif [ $HAS_DNF -eq 1 ]; then
    dnf install -y \
      python3 python3-virtualenv python3-pip python3-devel @development-tools \
      gcc gcc-c++ gcc-gfortran libomp-devel \
      git wget curl unzip openblas-devel lapack-devel \
      pciutils dmidecode lshw hwloc jq || true
  elif [ $HAS_PAC -eq 1 ]; then
    pacman -Sy --noconfirm \
      python python-virtualenv python-pip base-devel \
      gcc gfortran openmp \
      git wget curl unzip openblas lapack \
      pciutils dmidecode lshw hwloc jq || true
  else
    warn "No known package manager detected; assuming deps exist."
  fi
}

# --- Performance benchmark execution ---
run_performance_benchmark() {
  local run_num=$1
  local base_ts=$2
  
  TS="${base_ts}_run${run_num}"
  RUN_DIR="$ROOT/perf_runs/$TS"
  PROV_DIR="$RUN_DIR/provenance"
  LOG_DIR="$RUN_DIR/logs"
  RES_DIR="$RUN_DIR/results"
  
  mkdir -p "$PROV_DIR" "$LOG_DIR" "$RES_DIR"
  
  log "=== Starting performance benchmark run $run_num/$RUN_COUNT ==="

  # Copy helper scripts for this run
  cp "$SETUP_DIR"/*.py "$RUN_DIR/" 2>/dev/null || true
  cp "$SETUP_DIR"/*.c "$RUN_DIR/" 2>/dev/null || true
  
  # Collect device info + apply hygiene
  log "Collecting device info and applying system hygiene…"
  source "$VENV_PERF/bin/activate"
  python "$RUN_DIR/device_info_collect.py" | tee "$PROV_DIR/device_info.json" >/dev/null
  if [[ $DETERMINISTIC_MODE -eq 1 ]]; then
    python "$RUN_DIR/hygiene_apply_and_snapshot.py" deterministic | tee "$PROV_DIR/hygiene_post.json" >/dev/null
  else
    python "$RUN_DIR/hygiene_apply_and_snapshot.py" | tee "$PROV_DIR/hygiene_post.json" >/dev/null
  fi
  deactivate || true

  # Initialize results structure
  python3 -c "
import json
from datetime import datetime
result = {
  'run_number': $run_num,
  'timestamp_utc': datetime.utcnow().isoformat() + 'Z',
  'cpu_benchmarks': {},
  'gpu_benchmarks': {}
}
with open('$RES_DIR/perf_results.json', 'w') as f:
    json.dump(result, f, indent=2)
"

  # --- CPU benchmarks ---
  if [[ $USE_CPU -eq 1 ]]; then
    log "Running CPU performance benchmarks (run $run_num)…"
    
    # Compile and run STREAM Triad
    log "  CPU Memory Bandwidth (STREAM Triad)…"
    cd "$RUN_DIR"
    gcc -Ofast -march=native -mtune=native -fopenmp stream.c -o stream || {
      warn "Failed to compile STREAM benchmark"
    }
    
    if [ -f stream ]; then
      export OMP_NUM_THREADS=$(nproc)
      export OMP_PROC_BIND=close
      export OMP_PLACES=cores
      
      # Check for NUMA and run with interleaving if multi-socket
      if command -v numactl >/dev/null 2>&1; then
        NUMA_NODES=$(numactl -H | grep "available:" | awk '{print $2}' || echo "1")
        if [ "$NUMA_NODES" -gt 1 ]; then
          log "  Detected $NUMA_NODES NUMA nodes, using interleaved memory policy"
          numactl --interleave=all ./stream 2>&1 | tee "$LOG_DIR/cpu_stream.log"
        else
          ./stream 2>&1 | tee "$LOG_DIR/cpu_stream.log"
        fi
      else
        ./stream 2>&1 | tee "$LOG_DIR/cpu_stream.log"
      fi
      
      # Extract best bandwidth
      CPU_BW=$(grep "BEST_BW:" "$LOG_DIR/cpu_stream.log" | cut -d' ' -f2 || echo "0.0")
      log "  CPU Memory BW: $CPU_BW GB/s"
    else
      CPU_BW="0.0"
      warn "STREAM benchmark not available"
    fi
    
    # Run CPU SGEMM
    log "  CPU Compute Performance (SGEMM)…"
    source "$VENV_PERF/bin/activate"
    
    # Check for NUMA and run with interleaving if multi-socket (same as STREAM)
    if command -v numactl >/dev/null 2>&1; then
      NUMA_NODES=$(numactl -H | grep "available:" | awk '{print $2}' || echo "1")
      if [ "$NUMA_NODES" -gt 1 ]; then
        log "  Using NUMA interleaving for multi-socket SGEMM ($NUMA_NODES nodes)"
        numactl --interleave=all python cpu_sgemm_bench.py "$GEMM_SIZE" 2>&1 | tee "$LOG_DIR/cpu_sgemm.log"
      else
        python cpu_sgemm_bench.py "$GEMM_SIZE" 2>&1 | tee "$LOG_DIR/cpu_sgemm.log"
      fi
    else
      python cpu_sgemm_bench.py "$GEMM_SIZE" 2>&1 | tee "$LOG_DIR/cpu_sgemm.log"
    fi
    
    deactivate || true
    
    # Extract SGEMM results
    CPU_SGEMM_JSON=$(grep "CPU_SGEMM_JSON:" "$LOG_DIR/cpu_sgemm.log" | cut -d' ' -f2- || echo '{"best_tflops": 0.0}')
    CPU_TFLOPS=$(echo "$CPU_SGEMM_JSON" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin) if sys.stdin.read().strip() else {}
    print(data.get('best_tflops', 0.0))
except:
    print('0.0')
" 2>/dev/null || echo "0.0")
    
    log "  CPU Compute: $CPU_TFLOPS TFLOP/s"
    
    # Calculate intensity I* = P_d / BW_d
    CPU_INTENSITY=$(python3 -c "
bw_gbs = $CPU_BW
tflops = $CPU_TFLOPS
if bw_gbs > 0:
    bw_bytes_per_sec = bw_gbs * 1e9
    flops_per_sec = tflops * 1e12
    intensity = flops_per_sec / bw_bytes_per_sec
    print(f'{intensity:.2f}')
else:
    print('0.0')
" 2>/dev/null || echo "0.0")
    
    log "  CPU Intensity I*: $CPU_INTENSITY FLOP/byte"
    
    # Update results JSON
    python3 -c "
import json
with open('$RES_DIR/perf_results.json', 'r') as f:
    data = json.load(f)
data['cpu_benchmarks'] = {
    'memory_bandwidth_gbs': $CPU_BW,
    'compute_tflops': $CPU_TFLOPS,
    'intensity_flop_per_byte': $CPU_INTENSITY,
    'sgemm_details': $CPU_SGEMM_JSON
}
with open('$RES_DIR/perf_results.json', 'w') as f:
    json.dump(data, f, indent=2)
"
    cd "$ROOT"
  fi

  # --- GPU benchmarks ---
  if [[ $USE_GPU -eq 1 ]]; then
    log "Running GPU performance benchmarks (run $run_num)…"
    
    if [ $HAS_NVIDIA -eq 1 ]; then
      source "$VENV_PERF/bin/activate"
      cd "$RUN_DIR"
      
      # Prepare TF32 flag
      TF32_FLAG=""
      if [[ $ENABLE_TF32 -eq 1 ]]; then
        TF32_FLAG="enable_tf32"
      fi
      
      # Attempt clock management for deterministic results
      if [[ $DETERMINISTIC_MODE -eq 1 ]]; then
        log "  Attempting to lock GPU clocks for deterministic results..."
        MAX_GR_CLOCK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1 | tr -d ' ')
        MAX_MEM_CLOCK=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits | head -1 | tr -d ' ')
        
        if nvidia-smi -lgc $MAX_GR_CLOCK 2>/dev/null; then
          log "  ✓ Graphics clock locked to $MAX_GR_CLOCK MHz"
        else
          log "  ⚠ Could not lock graphics clock, using dynamic clocks"
        fi
        
        if nvidia-smi -lmc $MAX_MEM_CLOCK 2>/dev/null; then
          log "  ✓ Memory clock locked to $MAX_MEM_CLOCK MHz"  
        else
          log "  ⚠ Could not lock memory clock, using dynamic clocks"
        fi
      else
        log "  Using peak performance mode (dynamic clocks)"
        nvidia-smi -rgc 2>/dev/null || true
        nvidia-smi -rmc 2>/dev/null || true
      fi
      
      # Log current clocks before benchmarks
      CURRENT_CLOCKS=$(nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv,noheader,nounits | head -1)
      log "  Current clocks: $CURRENT_CLOCKS (graphics,memory MHz)"
      
      # GPU SGEMM
      log "  GPU Compute Performance (SGEMM)…"
      python gpu_perf_bench.py sgemm $GEMM_SIZE $TF32_FLAG 2>&1 | tee "$LOG_DIR/gpu_sgemm.log"
      
      # GPU Bandwidth
      log "  GPU Memory Bandwidth…"
      python gpu_perf_bench.py bandwidth 2>&1 | tee "$LOG_DIR/gpu_bandwidth.log"
      
      # Extract results
      GPU_SGEMM_JSON=$(grep "GPU_SGEMM_JSON:" "$LOG_DIR/gpu_sgemm.log" | cut -d' ' -f2- || echo '{"best_tflops": 0.0}')
      GPU_BW_JSON=$(grep "GPU_BW_JSON:" "$LOG_DIR/gpu_bandwidth.log" | cut -d' ' -f2- || echo '{"best_bandwidth_gbs": 0.0}')
      
      GPU_TFLOPS=$(echo "$GPU_SGEMM_JSON" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin) if sys.stdin.read().strip() else {}
    print(data.get('best_tflops', 0.0))
except:
    print('0.0')
" 2>/dev/null || echo "0.0")
      
      GPU_BW=$(echo "$GPU_BW_JSON" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin) if sys.stdin.read().strip() else {}
    print(data.get('best_bandwidth_gbs', 0.0))
except:
    print('0.0')
" 2>/dev/null || echo "0.0")
      
      log "  GPU Compute: $GPU_TFLOPS TFLOP/s"
      log "  GPU Memory BW: $GPU_BW GB/s"
      
      # Calculate GPU intensity
      GPU_INTENSITY=$(python3 -c "
bw_gbs = $GPU_BW
tflops = $GPU_TFLOPS
if bw_gbs > 0:
    bw_bytes_per_sec = bw_gbs * 1e9
    flops_per_sec = tflops * 1e12
    intensity = flops_per_sec / bw_bytes_per_sec
    print(f'{intensity:.2f}')
else:
    print('0.0')
" 2>/dev/null || echo "0.0")
      
      log "  GPU Intensity I*: $GPU_INTENSITY FLOP/byte"
      
      # Update results JSON
      python3 -c "
import json
with open('$RES_DIR/perf_results.json', 'r') as f:
    data = json.load(f)
data['gpu_benchmarks'] = {
    'memory_bandwidth_gbs': $GPU_BW,
    'compute_tflops': $GPU_TFLOPS,
    'intensity_flop_per_byte': $GPU_INTENSITY,
    'sgemm_details': $GPU_SGEMM_JSON,
    'bandwidth_details': $GPU_BW_JSON
}
with open('$RES_DIR/perf_results.json', 'w') as f:
    json.dump(data, f, indent=2)
"
      
      # Reset clocks to auto after benchmarks (cleanup)
      if [[ $DETERMINISTIC_MODE -eq 1 ]]; then
        log "  Resetting GPU clocks to automatic"
        nvidia-smi -rgc 2>/dev/null || log "  Note: Could not reset graphics clock"
        nvidia-smi -rmc 2>/dev/null || log "  Note: Could not reset memory clock"
      fi
      
      deactivate || true
      cd "$ROOT"
    else
      log "  GPU benchmarks skipped (no NVIDIA GPU detected)"
      python3 -c "
import json
with open('$RES_DIR/perf_results.json', 'r') as f:
    data = json.load(f)
data['gpu_benchmarks'] = {'error': 'No NVIDIA GPU detected'}
with open('$RES_DIR/perf_results.json', 'w') as f:
    json.dump(data, f, indent=2)
"
    fi
  fi

  log "=== Finished performance benchmark run $run_num/$RUN_COUNT ==="
  return 0
}

# --- Main execution: setup once, run multiple times ---
log "Setting up performance benchmark environment…"

# Install system dependencies
install_sysdeps

# Setup Python environment
if [ ! -d "$VENV_PERF" ]; then
  log "Creating Python environment for performance benchmarks…"
  python3 -m venv "$VENV_PERF"
  "$VENV_PERF/bin/python" -m pip install -U pip wheel setuptools
  "$VENV_PERF/bin/pip" install numpy torch nvidia-ml-py3
fi

# Create benchmark scripts
create_device_info_script
create_hygiene_script
create_cpu_benchmark_scripts
create_gpu_benchmark_scripts

# Execute benchmark runs
FAILED_RUNS=0
SUCCESS_COUNT=0
ALL_RESULTS=()

for ((i=1; i<=RUN_COUNT; i++)); do
  log "Starting performance benchmark run $i of $RUN_COUNT"
  if run_performance_benchmark "$i" "$BASE_TS"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    
    # Collect this run's results
    RESULT_FILE="$ROOT/perf_runs/${BASE_TS}_run${i}/results/perf_results.json"
    if [ -f "$RESULT_FILE" ]; then
      ALL_RESULTS+=("$RESULT_FILE")
    fi
    
    log "✓ Run $i completed successfully"
  else
    FAILED_RUNS=$((FAILED_RUNS + 1))
    warn "✗ Run $i failed"
  fi
done

# --- Aggregate results and create final JSON ---
log "Aggregating results across all runs…"

cat > "$SETUP_DIR/aggregate_results.py" << 'PY'
import json
import sys
from datetime import datetime
import statistics
import math

def ci95(vals):
    """Calculate 95% confidence interval"""
    if len(vals) < 2: 
        return 0.0
    return 1.96 * statistics.stdev(vals) / math.sqrt(len(vals))

def aggregate_results(result_files):
    """Aggregate performance results across multiple runs"""
    
    aggregated = {
        "benchmark_info": {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "total_runs": len(result_files),
            "benchmark_type": "cpu_gpu_performance",
            "script_version": "paper-grade",
            "units_specification": {
                "cpu_memory_bandwidth": "GB/s (STREAM Triad, 1e9 bytes/sec)",
                "cpu_compute": "GFLOP/s (single-precision SGEMM, best-effort BLAS)",
                "gpu_memory_bandwidth": "GB/s (device-to-device copy, 1e9 bytes/sec)",
                "gpu_compute": "TFLOP/s (single-precision SGEMM, FP32-exact or TF32)",
                "computational_intensity": "FLOP/byte (arithmetic intensity, roofline model)"
            },
            "statistical_notes": {
                "confidence_intervals": "95% confidence intervals based on normal distribution",
                "best_values": "Maximum observed performance across all runs",
                "aggregation_method": "Sample statistics across independent benchmark runs"
            }
        },
        "device_info": {},
        "cpu_results": {
            "runs": [],
            "aggregated": {}
        },
        "gpu_results": {
            "runs": [],
            "aggregated": {}
        }
    }
    
    cpu_bw_values = []
    cpu_tflops_values = []
    cpu_intensity_values = []
    
    gpu_bw_values = []
    gpu_tflops_values = []
    gpu_intensity_values = []
    
    for i, result_file in enumerate(result_files):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract device info from first run
            if i == 0:
                # Try to get device info from provenance
                prov_file = result_file.replace('/results/perf_results.json', '/provenance/device_info.json')
                try:
                    with open(prov_file, 'r') as f:
                        aggregated["device_info"] = json.load(f)
                except:
                    pass
            
            # CPU results
            if "cpu_benchmarks" in data and data["cpu_benchmarks"]:
                cpu_data = data["cpu_benchmarks"]
                aggregated["cpu_results"]["runs"].append(cpu_data)
                
                if "memory_bandwidth_gbs" in cpu_data:
                    cpu_bw_values.append(float(cpu_data["memory_bandwidth_gbs"]))
                if "compute_tflops" in cpu_data:
                    cpu_tflops_values.append(float(cpu_data["compute_tflops"]))
                if "intensity_flop_per_byte" in cpu_data:
                    cpu_intensity_values.append(float(cpu_data["intensity_flop_per_byte"]))
            
            # GPU results
            if "gpu_benchmarks" in data and data["gpu_benchmarks"] and "error" not in data["gpu_benchmarks"]:
                gpu_data = data["gpu_benchmarks"]
                aggregated["gpu_results"]["runs"].append(gpu_data)
                
                if "memory_bandwidth_gbs" in gpu_data:
                    gpu_bw_values.append(float(gpu_data["memory_bandwidth_gbs"]))
                if "compute_tflops" in gpu_data:
                    gpu_tflops_values.append(float(gpu_data["compute_tflops"]))
                if "intensity_flop_per_byte" in gpu_data:
                    gpu_intensity_values.append(float(gpu_data["intensity_flop_per_byte"]))
                    
        except Exception as e:
            print(f"Warning: Failed to process {result_file}: {e}", file=sys.stderr)
    
    # Aggregate CPU statistics with enhanced metadata
    if cpu_bw_values:
        aggregated["cpu_results"]["aggregated"] = {
            "memory_bandwidth_gbs": {
                "values": cpu_bw_values,
                "mean": round(statistics.mean(cpu_bw_values), 3),
                "median": round(statistics.median(cpu_bw_values), 3),
                "stdev": round(statistics.stdev(cpu_bw_values), 3) if len(cpu_bw_values) > 1 else 0.0,
                "ci95": round(ci95(cpu_bw_values), 3),
                "min": round(min(cpu_bw_values), 3),
                "max": round(max(cpu_bw_values), 3),
                "best": round(max(cpu_bw_values), 3),
                "coefficient_of_variation": round(100 * statistics.stdev(cpu_bw_values) / statistics.mean(cpu_bw_values), 2) if len(cpu_bw_values) > 1 and statistics.mean(cpu_bw_values) > 0 else 0.0,
                "units": "GB/s"
            },
            "compute_gflops": {
                "values": [v * 1000 for v in cpu_tflops_values],  # Convert TFLOP/s to GFLOP/s
                "mean": round(statistics.mean(cpu_tflops_values) * 1000, 3),
                "median": round(statistics.median(cpu_tflops_values) * 1000, 3),
                "stdev": round(statistics.stdev(cpu_tflops_values) * 1000, 3) if len(cpu_tflops_values) > 1 else 0.0,
                "ci95": round(ci95(cpu_tflops_values) * 1000, 3),
                "min": round(min(cpu_tflops_values) * 1000, 3),
                "max": round(max(cpu_tflops_values) * 1000, 3),
                "best": round(max(cpu_tflops_values) * 1000, 3),
                "coefficient_of_variation": round(100 * statistics.stdev(cpu_tflops_values) / statistics.mean(cpu_tflops_values), 2) if len(cpu_tflops_values) > 1 and statistics.mean(cpu_tflops_values) > 0 else 0.0,
                "units": "GFLOP/s"
            },
            "intensity_flop_per_byte": {
                "values": cpu_intensity_values,
                "mean": round(statistics.mean(cpu_intensity_values), 3),
                "median": round(statistics.median(cpu_intensity_values), 3),
                "stdev": round(statistics.stdev(cpu_intensity_values), 3) if len(cpu_intensity_values) > 1 else 0.0,
                "ci95": round(ci95(cpu_intensity_values), 3),
                "min": round(min(cpu_intensity_values), 3),
                "max": round(max(cpu_intensity_values), 3),
                "best": round(max(cpu_intensity_values), 3),
                "coefficient_of_variation": round(100 * statistics.stdev(cpu_intensity_values) / statistics.mean(cpu_intensity_values), 2) if len(cpu_intensity_values) > 1 and statistics.mean(cpu_intensity_values) > 0 else 0.0,
                "units": "FLOP/byte"
            }
        }
    
    # Aggregate GPU statistics with enhanced metadata
    if gpu_bw_values:
        aggregated["gpu_results"]["aggregated"] = {
            "memory_bandwidth_gbs": {
                "values": gpu_bw_values,
                "mean": round(statistics.mean(gpu_bw_values), 3),
                "median": round(statistics.median(gpu_bw_values), 3),
                "stdev": round(statistics.stdev(gpu_bw_values), 3) if len(gpu_bw_values) > 1 else 0.0,
                "ci95": round(ci95(gpu_bw_values), 3),
                "min": round(min(gpu_bw_values), 3),
                "max": round(max(gpu_bw_values), 3),
                "best": round(max(gpu_bw_values), 3),
                "coefficient_of_variation": round(100 * statistics.stdev(gpu_bw_values) / statistics.mean(gpu_bw_values), 2) if len(gpu_bw_values) > 1 and statistics.mean(gpu_bw_values) > 0 else 0.0,
                "units": "GB/s"
            },
            "compute_tflops": {
                "values": gpu_tflops_values,
                "mean": round(statistics.mean(gpu_tflops_values), 3),
                "median": round(statistics.median(gpu_tflops_values), 3),
                "stdev": round(statistics.stdev(gpu_tflops_values), 3) if len(gpu_tflops_values) > 1 else 0.0,
                "ci95": round(ci95(gpu_tflops_values), 3),
                "min": round(min(gpu_tflops_values), 3),
                "max": round(max(gpu_tflops_values), 3),
                "best": round(max(gpu_tflops_values), 3),
                "coefficient_of_variation": round(100 * statistics.stdev(gpu_tflops_values) / statistics.mean(gpu_tflops_values), 2) if len(gpu_tflops_values) > 1 and statistics.mean(gpu_tflops_values) > 0 else 0.0,
                "units": "TFLOP/s"
            },
            "intensity_flop_per_byte": {
                "values": gpu_intensity_values,
                "mean": round(statistics.mean(gpu_intensity_values), 3),
                "median": round(statistics.median(gpu_intensity_values), 3),
                "stdev": round(statistics.stdev(gpu_intensity_values), 3) if len(gpu_intensity_values) > 1 else 0.0,
                "ci95": round(ci95(gpu_intensity_values), 3),
                "min": round(min(gpu_intensity_values), 3),
                "max": round(max(gpu_intensity_values), 3),
                "best": round(max(gpu_intensity_values), 3),
                "coefficient_of_variation": round(100 * statistics.stdev(gpu_intensity_values) / statistics.mean(gpu_intensity_values), 2) if len(gpu_intensity_values) > 1 and statistics.mean(gpu_intensity_values) > 0 else 0.0,
                "units": "FLOP/byte"
            }
        }
    
    return aggregated

if __name__ == "__main__":
    result_files = sys.argv[1:]
    if not result_files:
        print("Error: No result files provided", file=sys.stderr)
        sys.exit(1)
    
    aggregated = aggregate_results(result_files)
    print(json.dumps(aggregated, indent=2))
PY

# Create final aggregated results
if [ ${#ALL_RESULTS[@]} -gt 0 ]; then
  log "Creating final aggregated results JSON…"
  python3 "$SETUP_DIR/aggregate_results.py" "${ALL_RESULTS[@]}" > "$ROOT/$OUTPUT_FILE"
  
  # Display summary
  log "=== PERFORMANCE BENCHMARK SUMMARY ==="
  log "Requested runs: $RUN_COUNT"
  log "Successful runs: $SUCCESS_COUNT"
  log "Failed runs: $FAILED_RUNS"
  log "Output file: $OUTPUT_FILE"
  
  if [ $SUCCESS_COUNT -gt 0 ]; then
    log ""
    log "📊 Results Summary:"
    
    # Extract and display key metrics
    python3 -c "
import json
try:
    with open('$ROOT/$OUTPUT_FILE', 'r') as f:
        data = json.load(f)
    
    print('CPU Performance:')
    if 'cpu_results' in data and 'aggregated' in data['cpu_results'] and data['cpu_results']['aggregated']:
        cpu = data['cpu_results']['aggregated']
        if 'memory_bandwidth_gbs' in cpu:
            bw = cpu['memory_bandwidth_gbs']
            print(f'   Memory BW: {bw[\"best\"]:.2f} GB/s (best) | {bw[\"mean\"]:.2f} ± {bw[\"ci95\"]:.2f} (95% CI)')
        if 'compute_gflops' in cpu:
            comp = cpu['compute_gflops']
            print(f'   Compute:   {comp[\"best\"]:.3f} GFLOP/s (best) | {comp[\"mean\"]:.3f} ± {comp[\"ci95\"]:.3f} (95% CI)')
        if 'intensity_flop_per_byte' in cpu:
            intens = cpu['intensity_flop_per_byte']
            print(f'   Intensity: {intens[\"best\"]:.2f} FLOP/byte (FP32-exact) | {intens[\"mean\"]:.2f} ± {intens[\"ci95\"]:.2f} (95% CI)')
    else:
        print('   No CPU results available')
    
    print()
    print('GPU Performance:')
    if 'gpu_results' in data and 'aggregated' in data['gpu_results'] and data['gpu_results']['aggregated']:
        gpu = data['gpu_results']['aggregated']
        if 'memory_bandwidth_gbs' in gpu:
            bw = gpu['memory_bandwidth_gbs']
            print(f'   Memory BW: {bw[\"best\"]:.2f} GB/s (best) | {bw[\"mean\"]:.2f} ± {bw[\"ci95\"]:.2f} (95% CI)')
        if 'compute_tflops' in gpu:
            comp = gpu['compute_tflops']
            print(f'   Compute:   {comp[\"best\"]:.3f} TFLOP/s (FP32-exact) | {comp[\"mean\"]:.3f} ± {comp[\"ci95\"]:.3f} (95% CI)')
        if 'intensity_flop_per_byte' in gpu:
            intens = gpu['intensity_flop_per_byte']
            print(f'   Intensity: {intens[\"best\"]:.2f} FLOP/byte (FP32-exact) | {intens[\"mean\"]:.2f} ± {intens[\"ci95\"]:.2f} (95% CI)')
    else:
        print('   No GPU results available')
        
except Exception as e:
    print(f'Error reading results: {e}')
"
    
    log ""
    log "Performance benchmark complete!"
    log "Detailed results saved to: $OUTPUT_FILE"
    
  else
    warn "❌ All benchmark runs failed!"
    exit 1
  fi
else
  warn "❌ No successful benchmark results to aggregate!"
  exit 1
fi

# Cleanup temporary files
log "Cleaning up temporary files…"
rm -rf "$SETUP_DIR"

log "Performance benchmark completed successfully!"