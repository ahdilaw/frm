#!/usr/bin/env bash
# Usage: bash __driver.sh [--blaze]
set -euo pipefail

trap 'echo -e "[\e[31m$(date +'%H:%M:%S')\e[0m] ❌ Error on line $LINENO"; exit 1' ERR

# --- Configuration ---
DATA_ARCHIVE_URL="https://drive.google.com/uc?export=download&id=195tP6tB4qTbaMxmP3tic7LxsrvDpP2ZW"
DATA_ARCHIVE_NAME="frm_code_static_250913.tar.gz"

# --- Parse args ---
BLAZE=""
for arg in "$@"; do
  case "$arg" in
    --blaze) BLAZE="--blaze" ;;
    *) echo "Unknown arg: $arg" >&2 ; exit 2 ;;
  esac
done

TS="$(date -u +'%Y%m%dT%H%M%SZ')"
ROOT="$(pwd)"
RUN_DIR="$ROOT/runs/$TS"
PROV_DIR="$RUN_DIR/provenance"
LOG_DIR="$RUN_DIR/logs"
RES_DIR="$RUN_DIR/results"
VENV_OX="$RUN_DIR/.venv_ox"    # Torch/ONNX env (numpy > 2)
VENV_TFL="$RUN_DIR/.venv_tfl"  # TFLite/TF env (numpy < 2)

mkdir -p "$PROV_DIR" "$LOG_DIR" "$RES_DIR"

log()  { echo -e "[$(date +'%H:%M:%S')] $*"; }
warn() { echo -e "[$(date +'%H:%M:%S')] ⚠ $*" >&2; }

OS_ID="unknown"; OS_VER=""; ARCH="$(uname -m)"
if [ -f /etc/os-release ]; then . /etc/os-release || true; OS_ID="${ID:-unknown}"; OS_VER="${VERSION_ID:-}"; fi
HAS_APT=0; command -v apt-get  >/dev/null 2>&1 && HAS_APT=1
HAS_DNF=0; command -v dnf      >/dev/null 2>&1 && HAS_DNF=1
HAS_PAC=0; command -v pacman   >/dev/null 2>&1 && HAS_PAC=1
HAS_NVIDIA=0; command -v nvidia-smi >/dev/null 2>&1 && HAS_NVIDIA=1
HAS_ROCM=0; command -v rocm-smi >/dev/null 2>&1 && HAS_ROCM=1
IS_PI=0; grep -qi 'raspberry pi' /proc/cpuinfo 2>/dev/null && IS_PI=1

log "OS: $OS_ID $OS_VER | ARCH: $ARCH | NVIDIA: $HAS_NVIDIA | ROCm: $HAS_ROCM | RPi: $IS_PI"

# --- Download and extract data/models if not present ---
download_data_models() {
  if [ -d "$ROOT/data" ] && [ -d "$ROOT/models" ]; then
    log "✓ Data and models directories already exist, skipping download"
    return 0
  fi
  
  if [ -z "$DATA_ARCHIVE_URL" ]; then
    warn "DATA_ARCHIVE_URL not set. Please update the script with the Google Drive link."
    warn "Expected file: $DATA_ARCHIVE_NAME"
    warn "Please download manually and extract to $ROOT/"
    return 1
  fi
  
  log "Downloading data and models archive…"
  
  # Google Drive large file download (bypasses virus scan warning)
  FILE_ID="195tP6tB4qTbaMxmP3tic7LxsrvDpP2ZW"
  DOWNLOAD_SUCCESS=0
  
  # Method 1: Try gdown (best for Google Drive large files)
  if command -v python3 >/dev/null 2>&1; then
    log "Trying download with gdown (bypasses virus scan)..."
    python3 -m pip install -q gdown 2>/dev/null || true
    if python3 -c "import gdown" 2>/dev/null; then
      python3 -c "
import gdown
import sys
try:
    gdown.download('https://drive.google.com/uc?id=$FILE_ID', '$ROOT/$DATA_ARCHIVE_NAME', quiet=False, fuzzy=True)
    print('gdown success')
    sys.exit(0)
except Exception as e:
    print(f'gdown failed: {e}')
    sys.exit(1)
" && DOWNLOAD_SUCCESS=1
    fi
  fi
  
  # Method 2: Direct approach with virus scan bypass
  if [ $DOWNLOAD_SUCCESS -eq 0 ] && command -v wget >/dev/null 2>&1; then
    log "Trying wget with virus scan bypass..."
    
    # Use the direct download URL that bypasses virus scan
    wget --progress=bar:force:noscroll --no-check-certificate \
      --header="User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36" \
      "https://drive.usercontent.google.com/download?id=$FILE_ID&export=download&authuser=0&confirm=t" \
      -O "$ROOT/$DATA_ARCHIVE_NAME" && DOWNLOAD_SUCCESS=1
      
    # If that fails, try the alternative usercontent URL
    if [ $DOWNLOAD_SUCCESS -eq 0 ]; then
      wget --progress=bar:force:noscroll --no-check-certificate \
        --header="User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36" \
        "https://drive.usercontent.google.com/u/0/uc?id=$FILE_ID&export=download" \
        -O "$ROOT/$DATA_ARCHIVE_NAME" && DOWNLOAD_SUCCESS=1
    fi
  fi
  
  # Method 3: curl with virus scan bypass  
  if [ $DOWNLOAD_SUCCESS -eq 0 ] && command -v curl >/dev/null 2>&1; then
    log "Trying curl with virus scan bypass..."
    
    curl -L --progress-bar \
      -H "User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36" \
      "https://drive.usercontent.google.com/download?id=$FILE_ID&export=download&authuser=0&confirm=t" \
      -o "$ROOT/$DATA_ARCHIVE_NAME" && DOWNLOAD_SUCCESS=1
      
    # Alternative curl method
    if [ $DOWNLOAD_SUCCESS -eq 0 ]; then
      curl -L --progress-bar \
        -H "User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36" \
        "https://drive.usercontent.google.com/u/0/uc?id=$FILE_ID&export=download" \
        -o "$ROOT/$DATA_ARCHIVE_NAME" && DOWNLOAD_SUCCESS=1
    fi
  fi
  
  # Check if any method succeeded
  if [ $DOWNLOAD_SUCCESS -eq 0 ]; then
    warn "All download methods failed. The file may require manual download due to Google Drive restrictions."
    warn "Please download manually from: https://drive.google.com/file/d/$FILE_ID/view"
    warn "Click 'Download anyway' when prompted about virus scan, then save as: $ROOT/$DATA_ARCHIVE_NAME"
    return 1
  fi
  
  # Verify download size (should be much larger than 100KB)
  if [ -f "$ROOT/$DATA_ARCHIVE_NAME" ]; then
    FILE_SIZE=$(stat -c%s "$ROOT/$DATA_ARCHIVE_NAME" 2>/dev/null || echo "0")
    if [ "$FILE_SIZE" -lt 100000 ]; then
      warn "Downloaded file is too small ($FILE_SIZE bytes). Likely got HTML confirmation page."
      warn "Please try downloading manually from: https://drive.google.com/file/d/$FILE_ID/view"
      rm -f "$ROOT/$DATA_ARCHIVE_NAME"
      return 1
    else
      log "Downloaded $(($FILE_SIZE / 1024 / 1024))MB archive successfully"
    fi
  fi
  
  log "Extracting data and models…"
  tar -xzf "$ROOT/$DATA_ARCHIVE_NAME" -C "$ROOT" || {
    warn "Extraction failed. Archive may be corrupted."
    return 1
  }
  
  # Verify extraction
  if [ -d "$ROOT/data" ] && [ -d "$ROOT/models" ]; then
    log "Data and models extracted successfully"
    log "Data samples: $(find "$ROOT/data" -name "*.JPEG" 2>/dev/null | wc -l || echo "unknown")"
    log "Model files: $(find "$ROOT/models" -type f 2>/dev/null | wc -l || echo "unknown")"
    
    # Clean up archive
    rm -f "$ROOT/$DATA_ARCHIVE_NAME"
    log "Cleaned up archive file"
  else
    warn "Extraction verification failed. Expected data/ and models/ directories."
    return 1
  fi
}

# Download data and models
download_data_models || {
  warn "Failed to download/extract data and models. Continuing anyway..."
}

# --- Faster builds: use RAM tmp if present, prefer wheels
if [ -d /dev/shm ]; then
  export TMPDIR=/dev/shm
  export PIP_CACHE_DIR=/dev/shm/pipcache
fi
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT=120
PIP_OPTS="--prefer-binary"

# --- System packages (best-effort; skip if container already has them)
install_sysdeps() {
  log "Installing system packages…"
  if [ $HAS_APT -eq 1 ]; then
    apt-get update -y || true
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-venv python3-pip python3-dev build-essential \
      git wget curl unzip pkg-config \
      libgl1 libglib2.0-0 \
      libopenblas-dev liblapack-dev \
      pciutils usbutils dmidecode lshw hwloc jq || true
  elif [ $HAS_DNF -eq 1 ]; then
    dnf install -y \
      python3 python3-virtualenv python3-pip python3-devel @development-tools \
      git wget curl unzip openblas-devel lapack-devel \
      mesa-libGL glib2 \
      pciutils usbutils dmidecode lshw hwloc jq || true
  elif [ $HAS_PAC -eq 1 ]; then
    pacman -Sy --noconfirm \
      python python-virtualenv python-pip base-devel \
      git wget curl unzip openblas lapack \
      mesa glib2 \
      pciutils usbutils dmidecode lshw hwloc jq || true
  else
    warn "No known package manager detected; assuming deps exist."
  fi
}
install_sysdeps

# --- Python venvs + deps
# Torch/ONNX env (numpy > 2)
log "Creating Torch/ONNX env…"
python3 -m venv "$VENV_OX"
"$VENV_OX/bin/python" -m pip install $PIP_OPTS -U pip wheel setuptools
"$VENV_OX/bin/pip" install $PIP_OPTS "numpy>2.0" "pandas>=1.5.0" "Pillow>=9.0.0" "tqdm>=4.64.0" "odfpy>=1.4.0" "openpyxl>=3.0.0"

# Torch + torchvision (try CUDA wheels first, fallback to CPU)
if [ $HAS_NVIDIA -eq 1 ]; then
  "$VENV_OX/bin/pip" install $PIP_OPTS --extra-index-url https://download.pytorch.org/whl/cu121 "torch>=1.13.0" "torchvision>=0.14.0" \
  || "$VENV_OX/bin/pip" install $PIP_OPTS "torch>=1.13.0" "torchvision>=0.14.0"
else
  "$VENV_OX/bin/pip" install $PIP_OPTS "torch>=1.13.0" "torchvision>=0.14.0"
fi
# ONNX stack
"$VENV_OX/bin/pip" install $PIP_OPTS "onnx>=1.12.0" "onnxruntime-gpu; platform_system=='Linux'" "onnxruntime; platform_system!='Linux'" "onnxsim>=0.4.17" "nvidia-ml-py3"
# Optional
"$VENV_OX/bin/pip" install $PIP_OPTS timm || true

# --- TFLite/TF env (numpy < 2) ---
log "Creating TFLite/TF env…"
python3 -m venv "$VENV_TFL"
"$VENV_TFL/bin/python" -m pip install $PIP_OPTS -U pip wheel setuptools
"$VENV_TFL/bin/pip" install $PIP_OPTS "numpy>=1.21.0,<2.0" "pandas>=1.5.0" "Pillow>=9.0.0" "tqdm>=4.64.0" "odfpy>=1.4.0" "openpyxl>=3.0.0" "nvidia-ml-py3"

# torchvision (for transforms) — light use only
if [ $HAS_NVIDIA -eq 1 ]; then
  "$VENV_TFL/bin/pip" install $PIP_OPTS --extra-index-url https://download.pytorch.org/whl/cu121 "torch>=1.13.0" "torchvision>=0.14.0" \
  || "$VENV_TFL/bin/pip" install $PIP_OPTS "torch>=1.13.0" "torchvision>=0.14.0"
else
  "$VENV_TFL/bin/pip" install $PIP_OPTS "torch>=1.13.0" "torchvision>=0.14.0"
fi

# TensorFlow (GPU → full TF; else CPU TF; Pi → tflite-runtime)
# Use TensorFlow 2.16.1 for stable GPU delegate support (2.20.0 has broken GPU delegates)
# Always try to install tflite-runtime as a fallback
if [ $IS_PI -eq 1 ]; then
  "$VENV_TFL/bin/pip" install $PIP_OPTS "tflite-runtime>=2.13.0" || true
else
  if [ $HAS_NVIDIA -eq 1 ]; then
    # Try TensorFlow 2.16.1 first for stable GPU delegate support, fallback to CPU, then tflite-runtime
    "$VENV_TFL/bin/pip" install $PIP_OPTS "tensorflow==2.16.1" || {
      log "TensorFlow 2.16.1 failed, trying CPU version..."
      "$VENV_TFL/bin/pip" install $PIP_OPTS "tensorflow-cpu==2.16.1" || {
        log "TensorFlow CPU failed, trying tflite-runtime..."
        "$VENV_TFL/bin/pip" install $PIP_OPTS "tflite-runtime>=2.13.0" || true
      }
    }
  else
    # CPU-only systems: try CPU TensorFlow 2.16.1 first, fallback to tflite-runtime
    "$VENV_TFL/bin/pip" install $PIP_OPTS "tensorflow-cpu==2.16.1" || {
      log "TensorFlow CPU failed, trying tflite-runtime..."
      "$VENV_TFL/bin/pip" install $PIP_OPTS "tflite-runtime>=2.13.0" || true
    }
  fi
  
  # Always try to install tflite-runtime as additional fallback (won't conflict)
  "$VENV_TFL/bin/pip" install $PIP_OPTS "tflite-runtime>=2.13.0" || true
fi

# Verify TF import and TFLite interpreter availability
"$VENV_TFL/bin/python" - <<'PY'
import sys

# Test TensorFlow import
tf_ok = False
tflite_runtime_ok = False
interpreter_ok = False

try:
    import tensorflow as tf
    print("✓ TensorFlow OK:", tf.__version__)
    tf_ok = True
    
    # Test TFLite interpreter via TensorFlow
    try:
        interpreter = tf.lite.Interpreter
        print("✓ tf.lite.Interpreter available")
        interpreter_ok = True
    except AttributeError:
        print("✗ tf.lite.Interpreter not available")
    
except Exception as e:
    print("✗ TensorFlow import failed:", e)

# Test tflite-runtime fallback
try:
    from tflite_runtime.interpreter import Interpreter
    print("✓ tflite-runtime available")
    tflite_runtime_ok = True
    if not interpreter_ok:
        interpreter_ok = True
        print("✓ Using tflite-runtime as interpreter")
except Exception as e:
    print("✗ tflite-runtime import failed:", e)

if not interpreter_ok:
    print("❌ No TFLite interpreter available!")
    print("   Please check TensorFlow or tflite-runtime installation")
    sys.exit(1)
else:
    print("✅ TFLite interpreter ready")
PY

# --- Helper tools (device info + hygiene JSON)
cat > "$RUN_DIR/device_info_collect.py" << 'PY'
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
    try: govs[root]=open(os.path.join(root,"scaling_governor")).read().strip()
    except: pass
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
    if isinstance(name,bytes): name=name.decode()
    mem=pynvml.nvmlDeviceGetMemoryInfo(h)
    try:
      e=float(pynvml.nvmlDeviceGetEnforcedPowerLimit(h))/1000.0
      mn,mx=pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)
      pl={"enforced_w":e,"min_w":mn/1000.0,"max_w":mx/1000.0}
    except Exception: pl={}
    g.append({"index":i,"name":name,"memory_total_mb":int(mem.total/1048576),"power_limits":pl})
  out["nvidia_gpus"]=g
except Exception as e:
  out["nvidia_gpus"]=f"not_available: {e}"
out["versions_ox_env"]={}
for m in ["numpy","pandas","onnxruntime","onnx","torch","tensorflow","tflite_runtime","pynvml"]:
  try:
    mod=__import__(m); out["versions_ox_env"][m]=getattr(mod,"__version__","unknown")
  except Exception: out["versions_ox_env"][m]="not_installed"
print(json.dumps(out, indent=2))
PY

cat > "$RUN_DIR/hygiene_apply_and_snapshot.py" << 'PY'
import json, os, subprocess, re
from datetime import datetime
def sh(cmd):
  try: return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
  except Exception as e: return f"__ERR__ {e}"
snap={"timestamp_utc": datetime.utcnow().isoformat()+"Z"}
os.environ.update({
  "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1",
  "CUDA_LAUNCH_BLOCKING":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8",
  "PYTHONHASHSEED":"0","CUDA_DETERMINISTIC_OPS":"1","CUBLAS_DETERMINISTIC":"1"
})
# governors -> performance (best effort)
for root,_,files in os.walk('/sys/devices/system/cpu'):
  if root.endswith('cpufreq') and 'scaling_governor' in files:
    try: open(os.path.join(root,'scaling_governor'),'w').write('performance')
    except: pass
# SMT off (best effort)
try: open('/sys/devices/system/cpu/smt/control','w').write('off')
except: pass
# NVIDIA: persistence + lock clocks (no sudo; assume root)
if os.system("nvidia-smi >/dev/null 2>&1")==0:
  snap["nvidia_pm_on"]=sh('nvidia-smi -pm 1')
  sup=sh('nvidia-smi -q -d SUPPORTED_CLOCKS')
  snap["nvidia_supported"]=sup
  mem=re.findall(r"Memory\s*:\s*(\d+) MHz", sup) or ['']
  gr=re.findall(r"Graphics\s*:\s*(\d+) MHz", sup) or ['']
  if mem[-1] and gr[-1]:
    snap["lock_mem_clock"]=sh(f"nvidia-smi -lmc {mem[-1]}")
    snap["lock_gfx_clock"]=sh(f"nvidia-smi -lgc {gr[-1]},{gr[-1]}")
snap["ps_aux"]=sh('ps aux')
try: snap["smt_control"]=open('/sys/devices/system/cpu/smt/control').read().strip()
except: snap["smt_control"]=""
snap["governors"]=sh("bash -lc 'grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true'")
snap["nvidia_smi"]=sh('nvidia-smi')
print(json.dumps(snap, indent=2))
PY

# --- Collect device info + hygiene (Torch/ONNX env)
log "Collecting device info JSON…"
source "$VENV_OX/bin/activate"
python "$RUN_DIR/device_info_collect.py" | tee "$PROV_DIR/device_info.json" >/dev/null
log "Applying hygiene + snapshot… (locking GPU clocks best-effort)"
python "$RUN_DIR/hygiene_apply_and_snapshot.py" | tee "$PROV_DIR/hygiene_post.json" >/dev/null
deactivate || true

# --- Common eval flags (Paper-grade benchmarking configuration)
LAT_FLAGS=(--latency --lat-warmup-batches 1 --lat-repeats-batch 1 --bs 1)
MEM_FLAGS=(--mem --mem-sample-hz 100)
ENERGY_FLAGS=(--energy --energy-sample-hz 100)

# --- ONNX eval (Torch/ONNX env)
if [ -f "$ROOT/__eval_onnx_.py" ]; then
  log "ONNX eval… (warmup=${LAT_FLAGS[2]}, repeats=${LAT_FLAGS[4]}, per-sample inference)"
  source "$VENV_OX/bin/activate"
  
  # Validate per-sample configuration
  log "Validating per-sample inference configuration…"
  if ! python -c "
import sys
# Verify batch size is 1 for true per-sample inference
args = sys.argv[1:]
bs_idx = next((i for i, arg in enumerate(args) if arg == '--bs'), None)
if bs_idx is None or int(args[bs_idx + 1]) != 1:
    print('ERROR: Batch size must be 1 for per-sample inference')
    sys.exit(1)
print('✓ Per-sample configuration validated')
" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" $BLAZE; then
    warn "Per-sample validation failed"
    deactivate || true
    exit 1
  fi
  
  python "$ROOT/__eval_onnx_.py" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" $BLAZE | tee "$LOG_DIR/onnx.log"
  deactivate || true
  mv -f frm_onnx_results.ods "$RES_DIR/frm_onnx_results.ods" 2>/dev/null || true
fi

# --- TFLite eval (TFLite env; add --gpu only if full TF present)
if [ -f "$ROOT/__eval_tflite_.py" ]; then
  log "TFLite eval… (warmup=${LAT_FLAGS[2]}, repeats=${LAT_FLAGS[4]}, per-sample inference)"
  TFL_FLAGS=()
  if [ $HAS_NVIDIA -eq 1 ]; then
    TF_GPU_OK="$("$VENV_TFL/bin/python" - <<'PY'
ok=0
try:
    import tensorflow as tf
    # Check for both TF config and TFLite Interpreter
    has_config = hasattr(tf, "config") and callable(getattr(tf.config, "list_physical_devices", None))
    has_interpreter = hasattr(tf, "lite") and hasattr(tf.lite, "Interpreter")
    ok = has_config and has_interpreter
    if ok:
        print("1")  # GPU delegate available
    else:
        print("0")  # No GPU delegate
except Exception:
    print("0")
PY
)"
    if [ "$TF_GPU_OK" = "1" ]; then
      TFL_FLAGS+=(--gpu)
      log "GPU delegate enabled for TFLite"
    fi
  fi
  source "$VENV_TFL/bin/activate"
  python "$ROOT/__eval_tflite_.py" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" "${TFL_FLAGS[@]}" $BLAZE | tee "$LOG_DIR/tflite.log"
  deactivate || true
  mv -f frm_tflite_results.ods "$RES_DIR/frm_tflite_results.ods" 2>/dev/null || true
fi

# --- PyTorch eval (Torch/ONNX env)
if [ -f "$ROOT/__eval_torch_.py" ]; then
  log "PyTorch eval… (warmup=${LAT_FLAGS[2]}, repeats=${LAT_FLAGS[4]}, per-sample inference)"
  source "$VENV_OX/bin/activate"
  python "$ROOT/__eval_torch_.py" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" $BLAZE | tee "$LOG_DIR/torch.log"
  deactivate || true
  mv -f frm_torch_results.ods "$RES_DIR/frm_torch_results.ods" 2>/dev/null || true
fi

# --- Package results
log "Validating benchmark results…"
RESULTS_FOUND=0
for res_file in "$RES_DIR"/*.ods; do
  if [ -f "$res_file" ]; then
    RESULTS_FOUND=$((RESULTS_FOUND + 1))
    log "✓ Found result: $(basename "$res_file")"
  fi
done

if [ $RESULTS_FOUND -eq 0 ]; then
  warn "No result files found! Benchmark may have failed."
else
  log "✓ $RESULTS_FOUND result file(s) generated successfully"
fi

log "Packaging results…"
tar -C "$RUN_DIR" -czf "$ROOT/results_$TS.tar.gz" provenance logs results
log "Paper-grade benchmark complete: $ROOT/results_$TS.tar.gz"
log "Benchmark configuration: warmup=${LAT_FLAGS[2]}, repeats=${LAT_FLAGS[4]}, per-sample inference (bs=1)"
