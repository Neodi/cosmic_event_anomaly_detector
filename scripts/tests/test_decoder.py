import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.decoder import DecoderCNN


def test_decoder_shapes_and_values():
    """
    Comprueba que el decoder:
    - Acepta un tensor (batch, latent+meta)
    - Devuelve (batch, T, 6)
    - No produce valores inf o NaN
    """
    batch_size   = 4
    latent_dim   = 32
    meta_dim     = 4
    T_max        = 128            # longitud final de la curva
    T0           = T_max // 4     # debe coincidir con el used en LightningModule
    inp_dim      = latent_dim + meta_dim

    # Instanciar el decoder
    dec = DecoderCNN(latent_and_meta=inp_dim, T0=T0)

    # Entrada aleatoria
    z = torch.randn(batch_size, inp_dim)

    # Forward
    recon = dec(z)

    # 1) Forma correcta
    assert recon.shape == (batch_size, T0 * 4, 6), (
        f"Se esperaba (B, {T0*4}, 6), pero salió {recon.shape}"
    )

    # 2) Valores finitos
    assert torch.isfinite(recon).all(), "El decoder produjo infs o NaNs"

if __name__ == "__main__":
    test_decoder_shapes_and_values()
    print("✅ test_decoder_shapes_and_values passed")