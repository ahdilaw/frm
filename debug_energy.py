#!/usr/bin/env python3

# Debug script to test EnergySampler in isolation
import sys
import time
import numpy as np
import math

# Import the relevant classes from the ONNX script
exec("""
try:
    import pynvml
    HAVE_NVML = True
except Exception:
    try:
        import nvidia_ml_py3 as pynvml
        HAVE_NVML = True
    except Exception:
        HAVE_NVML = False

class NVMLPower:
    def __init__(self):
        self.ok = False
        self.handles = []
        if not HAVE_NVML:
            return
        try:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                self.handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            self.ok = bool(self.handles)
        except Exception:
            pass

    def read_power_w(self):
        try:
            total = 0.0
            for h in self.handles:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
                total += power_mw / 1000.0
            return total
        except Exception:
            return None

class RAPLReader:
    def __init__(self):
        self.ok = False

    def read_energy_j(self):
        return None, None

class EnergySampler:
    def __init__(self, hz: int = 300):
        self.hz = max(200, min(int(hz), 500))
        self.dt = 1.0 / float(self.hz)
        self.rapl = RAPLReader()
        self.nvml = NVMLPower()
        print(f"EnergySampler init: RAPL={self.rapl.ok}, NVML={self.nvml.ok}, hz={self.hz}, dt={self.dt}")
        
    def _read_gpu_power(self):
        power = self.nvml.read_power_w()
        print(f"  GPU power reading: {power}")
        return power
        
    def test_sampling(self, duration=2.0):
        print(f"Starting {duration}s sampling test...")
        
        # Simulate the sampling
        import threading
        self.tstamps = []
        self.gpu_power_w = []
        self._stop = threading.Event()
        
        def _runner():
            count = 0
            while not self._stop.is_set() and count < 10:  # Limit to 10 samples for debug
                t0 = time.perf_counter()
                self.tstamps.append(t0)
                gpw = self._read_gpu_power()
                self.gpu_power_w.append(gpw if gpw is not None else math.nan)
                count += 1
                
                # sleep to respect sampling rate
                t1 = time.perf_counter()
                remain = self.dt - (t1 - t0)
                if remain > 0:
                    time.sleep(remain)
                    
        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        
        # Let it run
        time.sleep(duration)
        self._stop.set()
        thread.join(timeout=1.0)
        
        print(f"Collected {len(self.gpu_power_w)} power samples")
        print(f"Power values: {self.gpu_power_w[:5]}...")  # Show first 5
        
        # Calculate energy like the real code
        if len(self.tstamps) >= 2:
            dt_est = float(np.median(np.diff(self.tstamps)))
            print(f"Estimated dt: {dt_est}")
        else:
            dt_est = self.dt
            print(f"Using configured dt: {dt_est}")
            
        gp = np.array(self.gpu_power_w, dtype=float)
        gp = gp[np.isfinite(gp)]
        gpu_j = float(np.sum(gp) * dt_est) if gp.size > 0 else math.nan
        
        print(f"Valid power samples: {gp.size}")
        print(f"Sum of power: {np.sum(gp) if gp.size > 0 else 'N/A'}")
        print(f"Calculated GPU energy: {gpu_j} J")
        
        return gpu_j
""")

# Run the test
if __name__ == "__main__":
    sampler = EnergySampler(hz=100)  # 100 Hz for easier debugging
    energy = sampler.test_sampling(duration=2.0)
    print(f"Final result: {energy} J")