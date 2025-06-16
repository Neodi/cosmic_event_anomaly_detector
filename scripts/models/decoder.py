import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderCNN(nn.Module):
    """
    Conv 1-D decoder para curvas de luz multibanda.
    Entrada: (batch, latent_dim + meta_dim)
    Salida: (batch, T, 6)  

    """

    def __init__(self, latent_and_meta: int, T0:int):
        # T0: longitud (nº de mediciones) intermedia despues del encoder (T/4)
        # latent_and_meta: suma de las dimensiones del espacio latente y metadatos.
        
        super().__init__()

        # Proyección densa de z_concat a T0 * 128
        self.fc = nn.Linear(latent_and_meta, T0 * 128)

        # Upsampling y convoluciones
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)

        self.conv_6_bands = nn.Conv1d(32, 6, kernel_size=3, padding=1)

        self.__init_weights()

        self.T0 = T0
    
    def __init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_concat: torch.Tensor) -> torch.Tensor:
        """
        z_concat: Tensor de forma (batch, latent_dim + meta_dim)
        Devuelve: Tensor de forma (batch, T, 6) con las curvas de luz multibanda.
        """

        B = z_concat.size(0)

        # Densa y reshape
        x = self.fc(z_concat)                   # (B, T0 * 128)
        x = x.view(B, 128, self.T0)             # (B, 128, T0)

        # Umsample, convoluciones, batch norm y ReLU
        x = self.up1(x)                         # (B, 128, T0 * 2)
        x = F.relu(self.bn1(self.conv1(x)))     # (B, 64, T0 * 2)

        x = self.up2(x)                         # (B, 64, T0 * 4)
        x = F.relu(self.bn2(self.conv2(x)))     # (B, 32, T0 * 4)

        # Convolución final para 6 bandas
        x = self.conv_6_bands(x)                # (B, 6, T0 * 4)
        # Transponer
        x = x.permute(0, 2, 1)                  # (B, T0 * 4, 6)

        return x  # (B, T, 6) 
