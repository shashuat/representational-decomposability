"""TopK sparse autoencoder on cached activations, optionally with the OrtSAE penalty (g > 0).

Inputs are rescaled to E||x||^2 = 1 and decoder rows kept unit-norm (the regime the penalty and its gamma assume).
AuxK lets dead latents reconstruct the residual so they get revived."""
import torch
import torch.nn.functional as F
from torch import nn

from model import ort


class SAE(nn.Module):
    def __init__(self, d, m, K):
        super().__init__()
        self.K = K
        W = F.normalize(torch.randn(m, d), dim=-1)
        self.Wd, self.We, self.b = nn.Parameter(W.clone()), nn.Parameter(W.clone()), nn.Parameter(torch.zeros(d))
        self.register_buffer('sf', torch.ones(()))  # input scale
        self.register_buffer('ds', torch.zeros(m))  # tokens since each latent last fired

    def forward(self, x):
        """Raw activations (..., d) -> raw reconstruction, sparse codes (..., m), pre-TopK activations."""
        pre = F.relu((x * self.sf - self.b) @ self.We.T)
        v, i = pre.topk(self.K, -1)
        c = torch.zeros_like(pre).scatter(-1, i, v)
        return (c @ self.Wd + self.b) / self.sf, c, pre


@torch.enable_grad()
def train_sae(X, m=1024, K=16, g=0., steps=3000, bs=4096, lr=5e-4, aux=1 / 32, dead_tok=5e5, seed=0):
    """Train on activations X (N, d). g: OrtSAE penalty weight. Returns the SAE in eval mode."""
    torch.manual_seed(seed)
    X = X.float()
    sae = SAE(X.shape[-1], m, K).to(X.device)
    sae.sf.fill_(X.pow(2).sum(-1).mean().rsqrt())
    sae.b.data = X.mean(0) * sae.sf
    opt = torch.optim.Adam(sae.parameters(), lr)
    for _ in range(steps):
        x = X[torch.randint(len(X), (bs,), device=X.device)]
        x_hat, c, pre = sae(x)
        err = (x - x_hat) * sae.sf
        loss = err.pow(2).sum(-1).mean()
        sae.ds += bs
        sae.ds.masked_fill_((c > 0).any(0), 0)
        dead = sae.ds > dead_tok
        if dead.any():  # AuxK: the top-2K dead latents must reconstruct the residual
            v, i = (pre * dead).topk(min(2 * K, int(dead.sum())), -1)
            aux_hat = torch.zeros_like(pre).scatter(-1, i, v) @ sae.Wd
            loss = loss + aux * (err.detach() - aux_hat).pow(2).sum(-1).mean()
        if g:
            loss = loss + g * ort(sae.Wd)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.Wd.data = F.normalize(sae.Wd.data, dim=-1)
    return sae.eval()
