#!/usr/bin/env python3
"""
Quick test to verify the project setup works correctly.
"""

import sys
import torch
import numpy as np
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported."""
    try:
        import torch
        import transformers
        import datasets
        import wandb
        import matplotlib.pyplot as plt
        import seaborn as sns
        import yaml
        import sklearn
        print("✅ All imports successful!")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_pytorch():
    """Test PyTorch functionality."""
    try:
        # Test tensor operations
        x = torch.randn(2, 3, 4)
        y = torch.matmul(x, x.transpose(-1, -2))
        print(f"✅ PyTorch working! Tensor shape: {y.shape}")
        
        # Test GPU if available
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print(f"✅ CUDA available! GPU: {torch.cuda.get_device_name()}")
        else:
            print("ℹ️  CUDA not available, using CPU")
        
        return True
    except Exception as e:
        print(f"❌ PyTorch error: {e}")
        return False

def test_project_structure():
    """Test that project structure is correct."""
    required_dirs = [
        "src", "src/models", "src/training", "src/data",
        "configs", "experiments", "notebooks", "tests", "results"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ Project structure correct!")
        return True

def main():
    print("🧪 Testing Adaptive Sparse Transformer Setup")
    print("=" * 50)
    
    tests = [
        ("Package imports", test_imports),
        ("PyTorch functionality", test_pytorch),
        ("Project structure", test_project_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        results.append(test_func())
    
    print("\n" + "=" * 50)
    if all(results):
        print("🎉 All tests passed! Setup is complete.")
        print("\n🚀 Next steps:")
        print("1. Create your model files in src/models/")
        print("2. Set up your first experiment")
        print("3. Start coding your adaptive attention mechanism!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()