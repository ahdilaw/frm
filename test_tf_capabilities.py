#!/usr/bin/env python3
"""
Test script to discover TensorFlow Lite capabilities in current installation
"""

import sys
import os

print("=" * 60)
print("TensorFlow Lite Capabilities Test")
print("=" * 60)

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check what's available in tf.lite
    print("\nAvailable in tf.lite:")
    tf_lite_attrs = [attr for attr in dir(tf.lite) if not attr.startswith('_')]
    for attr in sorted(tf_lite_attrs):
        print(f"  {attr}")
    
    # Check experimental module
    print("\nAvailable in tf.lite.experimental:")
    exp_attrs = [attr for attr in dir(tf.lite.experimental) if not attr.startswith('_')]
    for attr in sorted(exp_attrs):
        print(f"  {attr}")
    
    # Test load_delegate
    print("\nTesting load_delegate functionality:")
    try:
        # Try to create a dummy delegate to see what happens
        test_delegate = tf.lite.experimental.load_delegate('nonexistent_lib')
        print("  load_delegate call succeeded (unexpected)")
    except Exception as e:
        print(f"  load_delegate error (expected): {e}")
    
    # Check GPU detection
    print(f"\nGPU devices detected: {len(tf.config.list_physical_devices('GPU'))}")
    for i, gpu in enumerate(tf.config.list_physical_devices('GPU')):
        print(f"  GPU {i}: {gpu}")
    
    # Test basic interpreter
    print("\nTesting basic TF Lite Interpreter:")
    try:
        # Create a simple model to test with
        import numpy as np
        
        # Create minimal test model
        test_model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        test_model.compile(optimizer='adam', loss='mse')
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(test_model)
        tflite_model = converter.convert()
        
        # Test interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()
        print("  ✓ Basic TF Lite Interpreter works")
        
    except Exception as e:
        print(f"  ✗ Basic interpreter failed: {e}")
    
    # Test Flex delegate
    print("\nTesting Flex delegate:")
    try:
        import tensorflow as tf
        flex_delegate = tf.lite.experimental.load_delegate('libdelegate_flex.so')
        print("  ✓ Flex delegate libdelegate_flex.so available")
    except Exception as e:
        print(f"  ✗ Flex delegate libdelegate_flex.so failed: {e}")
        
        try:
            # Try without .so extension
            flex_delegate = tf.lite.experimental.load_delegate('libdelegate_flex')
            print("  ✓ Flex delegate libdelegate_flex available")
        except Exception as e2:
            print(f"  ✗ Flex delegate libdelegate_flex failed: {e2}")
            
            try:
                # Try different name
                flex_delegate = tf.lite.experimental.load_delegate('flex_delegate')
                print("  ✓ Flex delegate flex_delegate available")
            except Exception as e3:
                print(f"  ✗ Flex delegate flex_delegate failed: {e3}")
    
    # Check what files are in TensorFlow installation
    print("\nTensorFlow installation path:")
    tf_path = tf.__file__.replace('__init__.py', '')
    print(f"  {tf_path}")
    
    # Look for delegate files
    print("\nSearching for delegate libraries...")
    import glob
    
    search_paths = [
        os.path.join(tf_path, "**", "*delegate*"),
        os.path.join(tf_path, "**", "*gpu*"),
        os.path.join(tf_path, "**", "*flex*")
    ]
    
    found_files = []
    for pattern in search_paths:
        found_files.extend(glob.glob(pattern, recursive=True))
    
    if found_files:
        print("  Found delegate-related files:")
        for f in sorted(set(found_files)):
            print(f"    {f}")
    else:
        print("  No delegate files found in TensorFlow installation")

except ImportError as e:
    print(f"Failed to import TensorFlow: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")

print("\n" + "=" * 60)