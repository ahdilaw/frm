#!/usr/bin/env python3
"""
Debug TensorFlow Lite import issues
"""

def debug_tensorflow_lite():
    print("="*60)
    print("TensorFlow Lite Import Debug")
    print("="*60)
    
    # Test 1: Basic TensorFlow import
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow imported: {tf.__version__}")
        
        # Show what's available in tf
        print(f"TensorFlow attributes containing 'lite': {[attr for attr in dir(tf) if 'lite' in attr.lower()]}")
        
        if hasattr(tf, 'lite'):
            print(f"tf.lite attributes: {[attr for attr in dir(tf.lite) if not attr.startswith('_')]}")
        
        # Test different import methods
        print("\nTesting different import methods:")
        
        # Method 1: tf.lite.Interpreter
        try:
            if hasattr(tf, 'lite') and hasattr(tf.lite, 'Interpreter'):
                interpreter = tf.lite.Interpreter
                print(f"✓ Method 1: tf.lite.Interpreter works - {interpreter}")
            else:
                print(f"✗ Method 1: tf.lite.Interpreter not available")
        except Exception as e:
            print(f"✗ Method 1: tf.lite.Interpreter failed - {e}")
        
        # Method 2: tf.compat.v1.lite.Interpreter
        try:
            if hasattr(tf, 'compat') and hasattr(tf.compat, 'v1') and hasattr(tf.compat.v1, 'lite'):
                interpreter = tf.compat.v1.lite.Interpreter
                print(f"✓ Method 2: tf.compat.v1.lite.Interpreter works - {interpreter}")
            else:
                print(f"✗ Method 2: tf.compat.v1.lite.Interpreter not available")
        except Exception as e:
            print(f"✗ Method 2: tf.compat.v1.lite.Interpreter failed - {e}")
        
        # Method 3: Direct import
        try:
            from tensorflow.lite import Interpreter
            print(f"✓ Method 3: Direct import works - {Interpreter}")
        except ImportError as e:
            print(f"✗ Method 3: Direct import failed - {e}")
        
        # Method 4: Alternative import
        try:
            import tensorflow.lite as tflite
            if hasattr(tflite, 'Interpreter'):
                print(f"✓ Method 4: tensorflow.lite module import works - {tflite.Interpreter}")
            else:
                print(f"✗ Method 4: tensorflow.lite.Interpreter not found")
        except ImportError as e:
            print(f"✗ Method 4: tensorflow.lite module import failed - {e}")
            
        # Check TensorFlow installation type
        import sys
        print(f"\nTensorFlow installation info:")
        print(f"  Python: {sys.version}")
        print(f"  TensorFlow path: {tf.__file__}")
        print(f"  TensorFlow version: {tf.__version__}")
        
        # Check for GPU
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            print(f"  GPUs detected: {len(gpus)} - {gpus}")
        except Exception as e:
            print(f"  GPU detection failed: {e}")
            
    except ImportError as e:
        print(f"✗ TensorFlow import failed: {e}")

if __name__ == "__main__":
    debug_tensorflow_lite()