#!/usr/bin/env python3
"""
Test GPU delegate availability in TensorFlow 2.20.0
"""

def test_gpu_delegate():
    print("="*60)
    print("GPU Delegate API Test for TensorFlow 2.20.0")
    print("="*60)
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        
        # Test Method 1: tf.lite.experimental.GpuDelegate
        try:
            gpu_delegate = tf.lite.experimental.GpuDelegate()
            print("✓ Method 1: tf.lite.experimental.GpuDelegate - WORKS")
            del gpu_delegate
        except AttributeError as e:
            print(f"✗ Method 1: tf.lite.experimental.GpuDelegate - FAILED: {e}")
        except Exception as e:
            print(f"? Method 1: tf.lite.experimental.GpuDelegate - ERROR: {e}")
        
        # Test Method 2: Direct import from interpreter module
        try:
            from tensorflow.lite.python.interpreter import GpuDelegate
            gpu_delegate = GpuDelegate()
            print("✓ Method 2: tensorflow.lite.python.interpreter.GpuDelegate - WORKS")
            del gpu_delegate
        except ImportError as e:
            print(f"✗ Method 2: tensorflow.lite.python.interpreter.GpuDelegate - FAILED: {e}")
        except Exception as e:
            print(f"? Method 2: tensorflow.lite.python.interpreter.GpuDelegate - ERROR: {e}")
        
        # Test Method 3: Direct import from experimental
        try:
            from tensorflow.lite.experimental import GpuDelegate
            gpu_delegate = GpuDelegate()
            print("✓ Method 3: tensorflow.lite.experimental.GpuDelegate (direct import) - WORKS")
            del gpu_delegate
        except ImportError as e:
            print(f"✗ Method 3: tensorflow.lite.experimental.GpuDelegate (direct import) - FAILED: {e}")
        except Exception as e:
            print(f"? Method 3: tensorflow.lite.experimental.GpuDelegate (direct import) - ERROR: {e}")
        
        # Test Method 4: Check what's available in tf.lite.experimental
        print(f"\nAvailable in tf.lite.experimental:")
        if hasattr(tf.lite, 'experimental'):
            attrs = [attr for attr in dir(tf.lite.experimental) if not attr.startswith('_')]
            print(f"  {attrs}")
        else:
            print("  tf.lite.experimental not available")
        
        # Test Method 5: Check what's available in tf.lite
        print(f"\nAvailable in tf.lite:")
        if hasattr(tf, 'lite'):
            attrs = [attr for attr in dir(tf.lite) if not attr.startswith('_')]
            print(f"  {attrs}")
        else:
            print("  tf.lite not available")
            
        # Check GPU availability
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(f"\nGPU devices: {len(gpus)} found")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        except Exception as e:
            print(f"\nGPU detection failed: {e}")
            
    except ImportError as e:
        print(f"TensorFlow import failed: {e}")

if __name__ == "__main__":
    test_gpu_delegate()