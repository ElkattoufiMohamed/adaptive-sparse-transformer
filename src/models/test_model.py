"""
Simple test model to verify our setup works.
"""

import torch
import torch.nn as nn


class SimpleTestModel(nn.Module):
    """A simple model to test our setup."""

    def __init__(self, input_dim: int = 768, num_classes: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        # Use mean pooling for simplicity
        pooled = torch.mean(x, dim=1)  # (batch_size, input_dim)
        return self.classifier(pooled)


def test_model():
    """Test that our model works correctly."""
    batch_size, seq_len, input_dim = 4, 128, 768
    num_classes = 2

    # Create model and test data
    model = SimpleTestModel(input_dim, num_classes)
    test_input = torch.randn(batch_size, seq_len, input_dim)

    # Forward pass
    output = model(test_input)

    print(f"✅ Model test passed!")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Expected output shape: ({batch_size}, {num_classes})")

    assert output.shape == (
        batch_size,
        num_classes,
    ), f"Wrong output shape: {output.shape}"
    print("✅ Shape assertion passed!")


if __name__ == "__main__":
    test_model()
