#!/bin/bash

# Complete experiment runner script
set -e  # Exit on any error

echo "🚀 Starting Adaptive Sparse Transformer Experiment"
echo "=================================================="

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Please run: python -m venv venv"
    exit 1
fi

# Verify setup
echo "🧪 Running setup verification..."
python test_setup.py

# Run quick test
echo "🔍 Running quick pipeline test..."
python scripts/quick_test.py

# Run actual training
echo "🏋️ Starting model training..."

# Train adaptive model
python experiments/train_adaptive.py \
    --config configs/base_config.yaml \
    --experiment-name "adaptive_v1_$(date +%Y%m%d_%H%M%S)" \
    --model-type adaptive

echo "🎉 Training complete!"
echo "📊 Check results/ directory for outputs"
echo "📈 View training progress at https://wandb.ai"