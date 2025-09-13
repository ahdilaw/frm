#!/usr/bin/env python3

import sys
import math
import time
sys.path.append('.')

# Import from the ONNX script
from __eval_onnx_ import NVMLPower, EnergySampler

print("Testing NVML fix...")

# Test NVMLPower directly
nvml = NVMLPower()
print(f"NVML OK: {nvml.ok}")
print(f"NVML handles: {len(nvml.handles)}")

if nvml.ok:
    power = nvml.read_power_w()
    print(f"NVML power reading: {power} W")
else:
    print("NVML not available")
    sys.exit(1)

# Test EnergySampler with GPU power integration
print("\nTesting EnergySampler...")
es = EnergySampler(hz=100)
print(f"EnergySampler RAPL: {es.rapl.ok}, NVML: {es.nvml.ok}")

print("Starting 2-second energy sampling...")
es.start()
time.sleep(2.0)
result = es.stop()

print("\nEnergySampler results:")
for key, value in result.items():
    if isinstance(value, float):
        if math.isnan(value):
            print(f"  {key}: NaN (PROBLEM!)")
        else:
            print(f"  {key}: {value:.6f}")
    else:
        print(f"  {key}: {value}")

# Check if GPU energy is valid
gpu_j = result.get('gpu_j', math.nan)
if not math.isnan(gpu_j) and gpu_j > 0:
    print(f"\n✅ SUCCESS: GPU energy measurement working: {gpu_j:.3f}J")
else:
    print(f"\n❌ FAILED: GPU energy is still NaN or zero: {gpu_j}")