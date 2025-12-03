import torch
import torch.nn as nn


class DummyECGEncoder(nn.Module):
    """
    Placeholder ECG encoder.

    Input:  x of shape [B, 1, T]
    Output: features of shape [B, out_dim]
    """

    def __init__(self, out_dim: int = 256) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # [B, 32, 1]
        )
        self.fc = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        h = self.feature(x)          # [B, 32, 1]
        h = h.squeeze(-1)            # [B, 32]
        return self.fc(h)            # [B, out_dim]


def build_encoder(device: str = "cpu") -> nn.Module:
    encoder = DummyECGEncoder()
    encoder.to(device)
    encoder.eval()  # frozen
    return encoder
