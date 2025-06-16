import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..models.losses import kl_divergence

def test_kl_divergence_zero():
    # si mu=0 y logvar=0 (sigma=1), KL debe ser cero
    batch_size, D = 4, 5
    mu = torch.zeros(batch_size, D)
    logvar = torch.zeros(batch_size, D)
    kl = kl_divergence(mu, logvar)
    assert abs(kl.item()) < 1e-6

def test_kl_divergence_nonzero():
    # para un caso sencillo mu=1, logvar=0 (sigma=1)
    #                -0.5 * Σ  (1 + log(var) - mu^2 - var)
    # sum over dims: -0.5 * sum(1 +    0     -   1  -  1 )  = -0.5 * (-1*D) = D/2
    batch_size, D = 2, 3
    mu = torch.ones(batch_size, D)
    logvar = torch.zeros(batch_size, D)
    kl = kl_divergence(mu, logvar)
    expected = (D/2)  # por muestra, pero luego .mean() deja el mismo
    assert abs(kl.item() - expected) < 1e-6

if __name__ == "__main__":
    test_kl_divergence_zero()
    test_kl_divergence_nonzero()
    print("✅ kl_divergence tests passed")
