# FRM: Framework Benchmarking Suite

A comprehensive, paper-grade benchmarking suite for evaluating machine learning models across PyTorch, ONNX Runtime, and TensorFlow Lite frameworks.

## Features

**Per-Sample Inference Benchmarking**
- True per-sample latency measurements (batch size = 1)
- Statistical analysis across 1000 ImageNet validation samples
- Cross-framework performance comparison

**Performance Optimization**
- Automatic GPU detection and acceleration
- CPU governor optimization and SMT control
- GPU clock locking for consistent measurements
- Deterministic execution settings

**Comprehensive Metrics**
- Latency distribution (median, percentiles, variance)
- Memory usage tracking (host + device)
- Energy consumption monitoring (RAPL + NVML)
- Model accuracy validation (top-1, top-5)

**Scientific Rigor**
- Isolated virtual environments per framework
- Complete system provenance capture
- Reproducible benchmarking methodology
- Paper-grade statistical analysis

## Quick Start

### 1. Download and Setup

```bash
# Clone the repository
git clone https://github.com/ahdilaw/frm.git
cd frm

# Run the benchmarking suite (downloads data automatically)
bash __driver.sh
```

### 2. Data and Models

The script automatically downloads and extracts:
- **ImageNet validation data** (1000 samples)
- **Pre-trained models** for all frameworks
- **Configuration files** and metadata

**Note:** The data archive (~2GB) is hosted separately and downloaded on first run.

### 3. Results

Results are packaged in timestamped archives:
```
results_YYYYMMDDTHHMMSSZ.tar.gz
├── provenance/          # System configuration and environment
├── logs/               # Detailed execution logs
└── results/           # Benchmark results (ODS format)
    ├── frm_torch_results.ods
    ├── frm_onnx_results.ods
    └── frm_tflite_results.ods
```

## Usage Options

```bash
# Standard benchmarking (recommended)
bash __driver.sh

# Quick testing (subset of samples)
bash __driver.sh --blaze
```

## Supported Platforms

- **Operating Systems:** Linux (Ubuntu, CentOS, Arch), with partial Windows support
- **Hardware:** x86_64, ARM64, including Raspberry Pi
- **Accelerators:** NVIDIA GPUs (CUDA), AMD GPUs (ROCm), CPU-only

## Framework Support

| Framework | Models | GPU Support | Quantization |
|-----------|--------|-------------|--------------|
| PyTorch | ✅ | CUDA/ROCm | ✅ |
| ONNX Runtime | ✅ | CUDA/TensorRT | ✅ |
| TensorFlow Lite | ✅ | GPU Delegate | ✅ |

## Requirements

### System Dependencies
- Python 3.8+
- Git, wget/curl, tar
- Build tools (gcc, cmake)
- Hardware monitoring tools

### Python Dependencies
Automatically installed in isolated environments:
- PyTorch + torchvision
- ONNX + onnxruntime
- TensorFlow/TensorFlow Lite
- NumPy, Pandas, Pillow, tqdm

## Benchmark Configuration

### Default Settings (Optimized for 1000 samples)
```bash
Warmup batches: 1
Timing repeats: 1  
Batch size: 1 (per-sample)
Memory sampling: 100Hz
Energy sampling: 100Hz
```

This configuration provides statistically robust results through large sample size (N=1000) rather than multiple repeats, making it both efficient and scientifically valid.

## Architecture

```
frm/
├── __driver.sh              # Main orchestration script
├── __eval_torch_.py         # PyTorch evaluation
├── __eval_onnx_.py          # ONNX Runtime evaluation  
├── __eval_tflite_.py        # TensorFlow Lite evaluation
├── imagenet_class_index.json # ImageNet class mapping
└── README.md               # This file

# Downloaded automatically:
├── data/                   # ImageNet validation samples
│   ├── manifest.csv       # Sample metadata
│   └── imagenet/         # Image files
└── models/               # Pre-trained models
    ├── torch/           # PyTorch models (.pth)
    ├── onnx/           # ONNX models (.onnx)
    └── tflite/        # TensorFlow Lite models (.tflite)
```

## Scientific Methodology

### Statistical Validity
- **Large sample size (N=1000)** provides robust statistical power
- **Per-sample timing** captures full latency distribution
- **Median reporting** robust to outliers
- **Percentile analysis** characterizes tail behavior

### Reproducibility
- **Deterministic execution** (fixed seeds, CUDA settings)
- **Environment isolation** (separate venvs per framework)
- **Complete provenance** (system specs, library versions)
- **Standardized preprocessing** (ImageNet normalization)

### Performance Hygiene
- **CPU optimization** (performance governors, SMT control)
- **GPU optimization** (clock locking, persistence mode)
- **Memory management** (explicit cleanup, monitoring)
- **Thread control** (single-threaded BLAS operations)

## Contributing

We welcome contributions! Please see our contributing guidelines and submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this benchmarking suite in your research, please cite:

```bibtex
@software{frm_benchmark_suite,
  title={FRM: Framework Benchmarking Suite},
  author={Ahmed Wali, Murtaza Taj},
  year={2025},
  url={https://github.com/ahdilaw/frm}
}
```