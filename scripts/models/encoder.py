
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    """
    Conv 1-D encoder para curvas de luz multibanda.
    Entrada: (batch, T, 6)          # batch es el tamaño del lote, T es el número de puntos de tiempo, 6 son las bandas.
    Salida : (mu, log_var)          # mu y log_var son los parámetros de la distribución gaussiana.
    """
    def __init__(self, latent_dim:int=32):
        super().__init__()

        # --- Bloque 1
        # Se mantiene la dimensión de entrada (T, 6) a través del padding.
        self.conv1 = nn.Conv1d(
            in_channels=6,      # 6 bandas
            out_channels=32,    # Número de filtros
            kernel_size=5,      # Tamaño del kernel
            padding=2           # Padding para mantener la dimensión
        )
        self.bn1 = nn.BatchNorm1d(32)

        # --- Bloque 2
        # Reduce la dimensión a la mitad (T/2, 32) con stride=2.
        self.conv2 = nn.Conv1d(
            in_channels=32,     # 32 filtros de la capa anterior
            out_channels=64,
            kernel_size=5,
            padding=2,
            stride=2            # Reducción de la dimensión a la mitad
        )
        self.bn2 = nn.BatchNorm1d(64)

        # --- Bloque 3
        # Reduce la dimensión a la mitad (T/4, 64) con stride=2.
        self.conv3 = nn.Conv1d(
            in_channels=64,     # 64 filtros de la capa anterior
            out_channels=128,
            kernel_size=3,      # Tamaño del kernel actualizado
            padding=1,          # Padding ajustado para mantener la dimensión
            stride=2            # Reducción de la dimensión a la mitad
        )
        self.bn3 = nn.BatchNorm1d(128)
        # Dimensión de salida: (batch, 128, T/4)

        # Dropout
        self.dropout = nn.Dropout(p=0.3)

        # Pooling para reducir la salida a (batch, 128, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Flatten
        # 1/2 neuronas para la media y otro 1/2 para la varianza logarítmica.
        self.fc = nn.Linear(128, latent_dim * 2) 

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: Tensor de entrada con forma (batch, T, 6)
        Devuelve: (mu, log_var) donde mu y log_var son tensores de salida.
        """
        # Input shape: (batch_size, in_channels, sequence_length)
        # Cambiar de (batch, T, 6) a (batch, 6, T)
        x = x.permute(0, 2, 1)

        # Conv1d + BatchNorm + ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Aplicar Dropout
        x = self.dropout(x)

        # Adaptative Average Pooling | (batch, 128, 1) -> (batch, 128)
        x = self.pool(x).squeeze(-1)

        # Flatten 
        x = self.fc(x)

        # Dividir en mu y log_var
        mu, log_var = torch.chunk(x, 2, dim=-1)
        return mu, log_var


