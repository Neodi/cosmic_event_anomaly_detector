import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..models.losses import masked_mse

def test_masked_mse_basic():
    # batch=2, T=4, C=1 para simplificar
    pred   = torch.tensor([[[1.0],[2.0],[3.0],[4.0]],
                           [[0.0],[0.0],[0.0],[0.0]]])
    target = torch.zeros_like(pred)
    # máscara: sólo dos valores válidos por muestra
    mask = torch.tensor([[[1],[0],[1],[0]],
                         [[1],[1],[0],[0]]], dtype=torch.float32)

    # cálculo manual:
    # sample0: ((1-0)^2 + (3-0)^2)/2 = (1+9)/2 = 5.0
    # sample1: ((0-0)^2 + (0-0)^2)/2 = 0.0
    expected = (5.0 + 0.0) / 2  # mean over batch = 2.5

    loss = masked_mse(pred, target, mask)
    assert abs(loss.item() - expected) < 1e-6

if __name__ == "__main__":
    test_masked_mse_basic()
    print("✅ masked_mse basic test passed")