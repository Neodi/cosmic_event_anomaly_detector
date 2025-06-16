import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..models.encoder import EncoderCNN

def test_encoder_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo disponible: {device}")

    T_max = 512          # longitud de la curva (tras padding)
    batch = 10000
    latent_dim = 32

    enc = EncoderCNN(latent_dim).to(device)
    dummy = torch.randn(batch, T_max, 6).to(device)     

    mu, log_var = enc(dummy)

    # comprobamos shapes
    assert mu.shape == (batch, latent_dim)
    assert log_var.shape == (batch, latent_dim)
    # valores finitos
    assert torch.isfinite(mu).all()
    assert torch.isfinite(log_var).all()

if __name__ == "__main__":
    test_encoder_shapes()
    print("✅ EncoderCNN shape test passed")