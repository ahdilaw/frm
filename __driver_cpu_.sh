#!/usr/bin/env bash
# Usage: bash __driver.sh [--blaze] [--run N] [--tflite] [--onnx] [--torch]
set -euo pipefail

show_error_context() {
  local error_line=$1
  local script_file="${BASH_SOURCE[0]}"
  echo -e "[\e[31m$(date +'%H:%M:%S')\e[0m] ❌ Error on line $error_line"
  echo -e "[\e[33m$(date +'%H:%M:%S')\e[0m] Context (lines $((error_line-2)) to $((error_line+2))):"
  
  # Show 5 lines of context (2 before, error line, 2 after)
  local start_line=$((error_line - 2))
  local end_line=$((error_line + 2))
  
  # Ensure we don't go below line 1
  if [ $start_line -lt 1 ]; then
    start_line=1
  fi
  
  # Extract and display the context lines
  sed -n "${start_line},${end_line}p" "$script_file" | nl -ba -v$start_line | while IFS= read -r line; do
    local line_num=$(echo "$line" | awk '{print $1}')
    local line_content=$(echo "$line" | cut -f2-)
    
    if [ "$line_num" -eq "$error_line" ]; then
      echo -e "[\e[31m$(date +'%H:%M:%S')\e[0m] >>> $line_num: $line_content"
    else
      echo -e "[\e[37m$(date +'%H:%M:%S')\e[0m]     $line_num: $line_content"
    fi
  done
}

trap 'show_error_context $LINENO; exit 1' ERR

# --- Configuration ---
DATA_ARCHIVE_URL="https://drive.google.com/uc?export=download&id=195tP6tB4qTbaMxmP3tic7LxsrvDpP2ZW"
DATA_ARCHIVE_NAME="frm_code_static_250913.tar.gz"

# --- Parse args ---
BLAZE=""
RUN_COUNT=1
USE_TFLITE=0
USE_ONNX=0
USE_TORCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --blaze) 
      BLAZE="--blaze" 
      shift ;;
    --run)
      if [[ $# -lt 2 ]] || ! [[ "$2" =~ ^[0-9]+$ ]]; then
        echo "Error: --run requires a positive integer" >&2
        exit 2
      fi
      RUN_COUNT="$2"
      shift 2 ;;
    --tflite)
      USE_TFLITE=1
      shift ;;
    --onnx)
      USE_ONNX=1
      shift ;;
    --torch)
      USE_TORCH=1
      shift ;;
    *) 
      echo "Unknown arg: $1" >&2
      echo "Usage: bash __driver.sh [--blaze] [--run N] [--tflite] [--onnx] [--torch]" >&2
      exit 2 ;;
  esac
done

# If no framework specified, enable all
if [[ $USE_TFLITE -eq 0 && $USE_ONNX -eq 0 && $USE_TORCH -eq 0 ]]; then
  USE_TFLITE=1
  USE_ONNX=1
  USE_TORCH=1
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
HAS_ROCM=0; command -v rocm-smi >/dev/null 2>&1 && HAS_ROCM=1
IS_PI=0; grep -qi 'raspberry pi' /proc/cpuinfo 2>/dev/null && IS_PI=1

log "Multiple runs requested: $RUN_COUNT"
log "Frameworks: TFLite=$USE_TFLITE ONNX=$USE_ONNX Torch=$USE_TORCH BLAZE=$BLAZE"
log "OS: $OS_ID $OS_VER | ARCH: $ARCH | ROCm: $HAS_ROCM | RPi: $IS_PI"

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

# --- Setup paths and environments (once) ---
BASE_TS="$(date -u +'%Y%m%dT%H%M%SZ')"
ROOT="$(pwd)"
SETUP_DIR="$ROOT/setup_$BASE_TS"
VENV_OX="$SETUP_DIR/.venv_ox"    # Torch/ONNX env (numpy > 2)
VENV_TFL="$SETUP_DIR/.venv_tfl"  # TFLite/TF env (numpy < 2)

mkdir -p "$SETUP_DIR"

# --- Main execution function ---
run_benchmark() {
  local run_num=$1
  local base_ts=$2
  
  TS="${base_ts}_run${run_num}"
  RUN_DIR="$ROOT/runs/$TS"
  PROV_DIR="$RUN_DIR/provenance"
  LOG_DIR="$RUN_DIR/logs"
  RES_DIR="$RUN_DIR/results"
  
  mkdir -p "$PROV_DIR" "$LOG_DIR" "$RES_DIR"
  
  log "=== Starting benchmark run $run_num/$RUN_COUNT ==="

# --- Helper tools (device info + hygiene JSON)
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
    try: govs[root]=open(os.path.join(root,"scaling_governor")).read().strip()
    except: pass
out["cpu_governors"]=govs
out["meminfo"]=sh("cat /proc/meminfo")
out["lspci_vga"]=sh("lspci | grep -i -E 'vga|3d|display'")
out["nvidia_gpus"]="cpu_only_mode"
out["versions_ox_env"]={}
for m in ["numpy","pandas","onnxruntime","onnx","torch","tensorflow","tflite_runtime"]:
  try:
    mod=__import__(m); out["versions_ox_env"][m]=getattr(mod,"__version__","unknown")
  except Exception: out["versions_ox_env"][m]="not_installed"
print(json.dumps(out, indent=2))
PY

cat > "$SETUP_DIR/hygiene_apply_and_snapshot.py" << 'PY'
import json, os, subprocess, re
from datetime import datetime
def sh(cmd):
  try: return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True)
  except Exception as e: return f"__ERR__ {e}"
snap={"timestamp_utc": datetime.utcnow().isoformat()+"Z"}
os.environ.update({
  # Core threading controls
  "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1",
  "BLIS_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1","TBB_NUM_THREADS":"1",
  # PyTorch specific
  "TORCH_NUM_THREADS":"1","TORCH_NUM_INTEROP_THREADS":"1",
  # TensorFlow specific  
  "TF_NUM_INTRAOP_THREADS":"1","TF_NUM_INTEROPTHREADS":"1",
  # ONNX Runtime specific
  "ORT_NUM_THREADS":"1",
  # Deterministic execution
  "PYTHONHASHSEED":"0","CUDA_DETERMINISTIC_OPS":"1","CUBLAS_DETERMINISTIC":"1",
  "CUDA_LAUNCH_BLOCKING":"1","CUBLAS_WORKSPACE_CONFIG":":4096:8",
  # CPU affinity (pin to CPU 0)
  "GOMP_CPU_AFFINITY":"0","KMP_AFFINITY":"granularity=fine,compact,1,0",
  "OMP_PROC_BIND":"true","OMP_PLACES":"cores"
})
# governors -> performance (best effort)
for root,_,files in os.walk('/sys/devices/system/cpu'):
  if root.endswith('cpufreq') and 'scaling_governor' in files:
    try: open(os.path.join(root,'scaling_governor'),'w').write('performance')
    except: pass
# SMT off (best effort)
try: open('/sys/devices/system/cpu/smt/control','w').write('off')
except: pass
snap["ps_aux"]=sh('ps aux')
try: snap["smt_control"]=open('/sys/devices/system/cpu/smt/control').read().strip()
except: snap["smt_control"]=""
snap["governors"]=sh("bash -lc 'grep . /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor || true'")
snap["nvidia_smi"]="cpu_only_mode"
print(json.dumps(snap, indent=2))
PY

  # --- Helper scripts for this run
  cp "$SETUP_DIR/device_info_collect.py" "$RUN_DIR/"
  cp "$SETUP_DIR/hygiene_apply_and_snapshot.py" "$RUN_DIR/"
  
  # --- Collect device info + hygiene (appropriate env based on frameworks)
  log "Collecting device info JSON…"
  if [[ $USE_ONNX -eq 1 || $USE_TORCH -eq 1 ]]; then
    source "$VENV_OX/bin/activate"
    python "$RUN_DIR/device_info_collect.py" | tee "$PROV_DIR/device_info.json" >/dev/null
    log "Applying hygiene + snapshot… (locking GPU clocks best-effort)"
    python "$RUN_DIR/hygiene_apply_and_snapshot.py" | tee "$PROV_DIR/hygiene_post.json" >/dev/null
    deactivate || true
  fi

  # --- Common eval flags
  LAT_FLAGS=(--latency --lat-warmup-batches 2 --lat-repeats-batch 1 --bs 1)
  MEM_FLAGS=(--mem --mem-sample-hz 500)
  ENERGY_FLAGS=(--energy --energy-sample-hz 300)

  # --- Check taskset availability and setup CPU affinity
  TASKSET_CMD=""
  if command -v taskset >/dev/null 2>&1; then
    TASKSET_CMD="taskset -c 0"
    log "✓ Using CPU affinity: taskset -c 0 (pinning to CPU core 0)"
  else
    warn "taskset not available - running without CPU affinity pinning"
    TASKSET_CMD=""
  fi
  
  log "Threading configuration: OMP_NUM_THREADS=1, TORCH_NUM_THREADS=1, TF_NUM_INTRAOP_THREADS=1"
  log "SMT status: $(cat /sys/devices/system/cpu/smt/control 2>/dev/null || echo 'unknown')"

  # --- Alternating ONNX/Torch execution ---
  if [[ $USE_ONNX -eq 1 && $USE_TORCH -eq 1 ]]; then
    log "Running ONNX and Torch in alternating pattern..."
    
    # ONNX first
    if [ -f "$ROOT/__eval_onnx_.py" ]; then
      log "ONNX eval (run $run_num)…"
      source "$VENV_OX/bin/activate"
      $TASKSET_CMD python "$ROOT/__eval_onnx_.py" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" $BLAZE | tee "$LOG_DIR/onnx.log"
      deactivate || true
      mv -f frm_onnx_results.ods "$RES_DIR/frm_onnx_results.ods" 2>/dev/null || true
    fi

    # Torch second
    if [ -f "$ROOT/__eval_torch_.py" ]; then
      log "PyTorch eval (run $run_num)…"
      source "$VENV_OX/bin/activate"
      $TASKSET_CMD python "$ROOT/__eval_torch_.py" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" $BLAZE | tee "$LOG_DIR/torch.log"
      deactivate || true
      mv -f frm_torch_results.ods "$RES_DIR/frm_torch_results.ods" 2>/dev/null || true
    fi
    
  elif [[ $USE_ONNX -eq 1 ]]; then
    # ONNX only
    if [ -f "$ROOT/__eval_onnx_.py" ]; then
      log "ONNX eval (run $run_num)…"
      source "$VENV_OX/bin/activate"
      $TASKSET_CMD python "$ROOT/__eval_onnx_.py" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" $BLAZE | tee "$LOG_DIR/onnx.log"
      deactivate || true
      mv -f frm_onnx_results.ods "$RES_DIR/frm_onnx_results.ods" 2>/dev/null || true
    fi
    
  elif [[ $USE_TORCH -eq 1 ]]; then
    # Torch only
    if [ -f "$ROOT/__eval_torch_.py" ]; then
      log "PyTorch eval (run $run_num)…"
      source "$VENV_OX/bin/activate"
      $TASKSET_CMD python "$ROOT/__eval_torch_.py" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" $BLAZE | tee "$LOG_DIR/torch.log"
      deactivate || true
      mv -f frm_torch_results.ods "$RES_DIR/frm_torch_results.ods" 2>/dev/null || true
    fi
  fi

  # --- TFLite eval (separate env for CPU/edge devices) ---
  if [[ $USE_TFLITE -eq 1 ]]; then
    if [ -f "$ROOT/__eval_tflite_.py" ]; then
      log "TensorFlow Lite eval (run $run_num)…"
      source "$VENV_TFL/bin/activate"
      $TASKSET_CMD python "$ROOT/__eval_tflite_.py" "${LAT_FLAGS[@]}" "${MEM_FLAGS[@]}" "${ENERGY_FLAGS[@]}" $BLAZE | tee "$LOG_DIR/tflite.log"
      deactivate || true
      mv -f frm_tflite_results.ods "$RES_DIR/frm_tflite_results.ods" 2>/dev/null || true
    fi
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
    return 1
  else
    log "✓ $RESULTS_FOUND result file(s) generated successfully"
  fi

  log "Packaging results…"
  tar -C "$RUN_DIR" -czf "$ROOT/results_$TS.tar.gz" provenance logs results
  log "=== Finished benchmark run $run_num/$RUN_COUNT ==="
  return 0
}

# --- Main execution: setup once, run multiple times ---
log "Setting up environments (once)…"

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
      git wget curl unzip pkg-config util-linux \
      libgl1 libglib2.0-0 \
      libopenblas-dev liblapack-dev \
      pciutils usbutils dmidecode lshw hwloc jq || true
  elif [ $HAS_DNF -eq 1 ]; then
    dnf install -y \
      python3 python3-virtualenv python3-pip python3-devel @development-tools \
      git wget curl unzip util-linux openblas-devel lapack-devel \
      mesa-libGL glib2 \
      pciutils usbutils dmidecode lshw hwloc jq || true
  elif [ $HAS_PAC -eq 1 ]; then
    pacman -Sy --noconfirm \
      python python-virtualenv python-pip base-devel \
      git wget curl unzip util-linux openblas lapack \
      mesa glib2 \
      pciutils usbutils dmidecode lshw hwloc jq || true
  else
    warn "No known package manager detected; assuming deps exist."
  fi
}
install_sysdeps

# --- Python venvs + deps (setup once, only for requested frameworks)
if [[ $USE_ONNX -eq 1 || $USE_TORCH -eq 1 ]]; then
  log "Creating Torch/ONNX env…"
  python3 -m venv "$VENV_OX"
  "$VENV_OX/bin/python" -m pip install $PIP_OPTS -U pip wheel setuptools
  "$VENV_OX/bin/pip" install $PIP_OPTS "numpy>2.0" "pandas>=1.5.0" "Pillow>=9.0.0" "tqdm>=4.64.0" "odfpy>=1.4.0" "openpyxl>=3.0.0"

  if [[ $USE_TORCH -eq 1 ]]; then
    # Torch + torchvision (CPU-only)
    "$VENV_OX/bin/pip" install $PIP_OPTS --extra-index-url https://download.pytorch.org/whl/cpu "torch>=1.13.0" "torchvision>=0.14.0"
    "$VENV_OX/bin/pip" install $PIP_OPTS timm || true
  fi
  
  if [[ $USE_ONNX -eq 1 ]]; then
    # ONNX stack (CPU-only)
    "$VENV_OX/bin/pip" install $PIP_OPTS "onnx>=1.12.0" "onnxruntime>=1.12.0" "onnxsim>=0.4.17"
  fi
fi

if [[ $USE_TFLITE -eq 1 ]]; then
  log "Creating TensorFlow Lite env…"
  python3 -m venv "$VENV_TFL"
  "$VENV_TFL/bin/python" -m pip install $PIP_OPTS -U pip wheel setuptools
  "$VENV_TFL/bin/pip" install $PIP_OPTS "numpy<2.0" "pandas>=1.5.0" "Pillow>=9.0.0" "tqdm>=4.64.0" "odfpy>=1.4.0" "openpyxl>=3.0.0"
  "$VENV_TFL/bin/pip" install $PIP_OPTS "tensorflow>=2.10.0" "tflite-runtime>=2.10.0"
fi

# --- Execute benchmark runs ---
FAILED_RUNS=0
SUCCESS_COUNT=0

for ((i=1; i<=RUN_COUNT; i++)); do
  log "Starting benchmark run $i of $RUN_COUNT"
  if run_benchmark "$i" "$BASE_TS"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    log "✓ Run $i completed successfully"
  else
    FAILED_RUNS=$((FAILED_RUNS + 1))
    warn "✗ Run $i failed"
  fi
done

# --- Final summary ---
log "=== BENCHMARK SUMMARY ==="
log "Requested runs: $RUN_COUNT"
log "Successful runs: $SUCCESS_COUNT"
log "Failed runs: $FAILED_RUNS"

if [ $SUCCESS_COUNT -gt 0 ]; then
  log "Result files:"
  find "$ROOT" -name "results_${BASE_TS}_run*.tar.gz" -exec basename {} \; | sort
  log "✅ Multi-run benchmark complete!"
  
  if [ $RUN_COUNT -gt 1 ]; then
    log "For averaging analysis:"
    log "  - Extract all result files"
    log "  - Combine latency measurements from each run"
    log "  - Calculate mean ± std dev across runs"
    log "  - Report final averaged performance metrics"
  fi
else
  warn "❌ All benchmark runs failed!"
  exit 1
fi
