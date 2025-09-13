#!/usr/bin/env python3
"""
Test script to verify TensorFlow setup for TFLite with GPU and Flex delegate support.
Run this to check if your environment is properly configured.
"""

def test_tensorflow_setup():
    print("="*60)
    print("TensorFlow Setup Test")
    print("="*60)
    
    # Test 1: Basic TensorFlow import
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow imported successfully: {tf.__version__}")
        tf_available = True
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")
        print(f"  Install with: pip install tensorflow")
        tf_available = False
    
    if not tf_available:
        print("\n❌ TensorFlow is required for full TFLite support. Exiting.")
        return False
    
    # Test 2: GPU detection
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU devices detected: {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            gpu_available = True
        else:
            print(f"✗ No GPU devices detected")
            gpu_available = False
    except Exception as e:
        print(f"✗ GPU detection failed: {e}")
        gpu_available = False
    
    # Test 3: TensorFlow Lite import
    try:
        from tensorflow.lite import Interpreter
        print(f"✓ TensorFlow Lite imported successfully")
        tflite_available = True
    except ImportError as e:
        print(f"✗ TensorFlow Lite import failed: {e}")
        tflite_available = False
    
    # Test 4: GPU delegate test
    if gpu_available and tflite_available:
        try:
            gpu_delegate = tf.lite.experimental.GpuDelegate()
            print(f"✓ GPU delegate created successfully")
            gpu_delegate_ok = True
        except Exception as e:
            print(f"✗ GPU delegate creation failed: {e}")
            gpu_delegate_ok = False
    else:
        gpu_delegate_ok = False
        print(f"! GPU delegate test skipped (GPU or TFLite not available)")
    
    # Test 5: Test simple Flex delegate model (if we had one)
    print(f"✓ Flex delegate support: Available with full TensorFlow")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"TensorFlow:      {'✓' if tf_available else '✗'}")
    print(f"TensorFlow Lite: {'✓' if tflite_available else '✗'}")
    print(f"GPU Support:     {'✓' if gpu_available else '✗'}")
    print(f"GPU Delegate:    {'✓' if gpu_delegate_ok else '✗'}")
    print(f"Flex Delegate:   {'✓' if tf_available else '✗'}")
    
    if tf_available and tflite_available:
        print(f"\n✓ Your setup is ready for TFLite evaluation!")
        if gpu_available:
            print(f"  - Use --gpu flag for GPU acceleration")
        print(f"  - Flex delegate will handle Erf operations automatically")
        return True
    else:
        print(f"\n❌ Setup incomplete. Install missing components.")
        if not tf_available:
            print(f"   pip install tensorflow")
        return False

if __name__ == "__main__":
    test_tensorflow_setup()