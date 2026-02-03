import torch
import torch.nn as nn
import numpy as np


class EmbeddingAdapter(nn.Module):
    def __init__(self, dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim)
        )

    def forward(self, x):
        x = x + self.net(x)              # residual
        x = torch.nn.functional.normalize(x, dim=1)
        return x


class FaceEmbedder:
    def __init__(self, adapter_path, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.adapter = EmbeddingAdapter()
        self.adapter.load_state_dict(
            torch.load(adapter_path, map_location=self.device)
        )
        self.adapter.to(self.device)
        self.adapter.eval()

    @torch.no_grad()
    def adapt(self, emb_np: np.ndarray) -> np.ndarray:
        """
        emb_np: (512,) numpy array
        returns: (512,) adapted + normalized numpy array
        """
        emb = torch.from_numpy(emb_np).float().unsqueeze(0).to(self.device)
        emb = self.adapter(emb)
        return emb.cpu().numpy()[0]
