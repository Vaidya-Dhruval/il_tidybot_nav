import torch
import torch.nn as nn

class BCPolicy(nn.Module):
    """
    State-only BC policy:
      state (B,7) -> action (B,3) in [-1,1]

    We keep the same class/file name so the rest of the pipeline remains simple.
    """
    def __init__(self, state_dim: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, state):
        return torch.tanh(self.net(state))