import torch
import torch.nn.functional as F

def masked_mse(pred: torch.Tensor, 
               target: torch.Tensor, 
               mask: torch.Tensor) -> torch.Tensor: 
    """
    pred, target, mask: Tensor de forma (batch, T, C)

    Calcula el error cuadrático medio (MSE) entre las predicciones y los objetivos, aplicando una máscara.
    La mascara indica donde hay datos validos.

    Formula:
    MSE = (1/n) * Σ (pred - target)^2 * mask
    donde n es el número de elementos válidos (donde mask es 1).

    """

    diff2 = (pred - target).pow(2) * mask

    numerator = diff2.sum(dim=(1,2)) 
    denominator = mask.sum(dim=(1,2)).clamp(min=1)

    mse = numerator / denominator                    
    mse = mse.mean()                                
    return mse

def kl_divergence(mu: torch.Tensor, 
                  log_var: torch.Tensor) -> torch.Tensor:
    """
    mu, log_var: Tensor de forma (batch, D)

    Calcula la divergencia KL entre dos distribuciones gaussianas.

    Formula:
    KL(N(mu, var) || N(0, 1)) = -0.5 * Σ (1 + log(var) - mu^2 - var)

    Kingma, D. P., & Welling, M. (2014). Auto‐Encoding Variational Bayes. arXiv:1312.6114
    """
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1)
    return kl.mean()

def linear_beta_scheduler(epoch: int,
                          warmup_epochs: int = 0,
                          beta_start: float = 0.0,
                          beta_end: float = 1.0) -> float:
    """
    Calcula el valor beta para el scheduler linealcdurante las primeras epocas del entrenamiento.
    para que el modelo pueda aprener algo antes de se le comience a penalizar.
    """
    if epoch < warmup_epochs:
        return beta_start + (beta_end - beta_start) * (epoch / warmup_epochs)
    else:
        return beta_end
    

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
    
    