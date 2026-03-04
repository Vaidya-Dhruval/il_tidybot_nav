import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 3, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 128, 128)
            h = self.conv(dummy)
            flat = h.view(1, -1).shape[1]
        self.fc = nn.Sequential(nn.Linear(flat, out_dim), nn.ReLU())

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class BCPolicy(nn.Module):
    """
    image (B,3,128,128) + state (B,7) -> action (B,3) in [-1,1]
    """
    def __init__(self, state_dim: int = 7, img_feat: int = 256, state_feat: int = 128):
        super().__init__()
        self.cnn = SmallCNN(3, img_feat)
        self.mlp = MLP(state_dim, state_feat)
        self.head = nn.Sequential(
            nn.Linear(img_feat + state_feat, 256), nn.ReLU(),
            nn.Linear(256, 3),
        )

    def forward(self, image, state):
        fi = self.cnn(image)
        fs = self.mlp(state)
        z = torch.cat([fi, fs], dim=1)
        return torch.tanh(self.head(z))