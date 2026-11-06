"""Decoder-only LM (default 2 layers, 2 heads) with three switchable decomposability mechanisms:

  k > 0   Backpack readout: logits = sum_l sum_j alpha_lij * sense_l(x_j) @ E^T            (Hewitt et al., 2023)
  C > 0   CoCoMix concept head after block 0 (CE on attribution labels); mix=1 also interleaves
          the compressed concept vector into the sequence before the remaining blocks      (Tack et al., 2025)
  og > 0  OrtSAE-style orthogonality penalty (max-cos^2) on sense vectors / concept directions (Korznikov et al., 2025)
"""
import torch
import torch.nn.functional as F
from torch import nn


def ort(W, K=1):
    """OrtSAE penalty. W: (..., m, d). Mean over rows of (max cosine to another row of the same random chunk)^2.
    The diagonal is zeroed (as in the released code), so the max is hinged at 0."""
    W = F.normalize(W[..., torch.randperm(W.shape[-2], device=W.device), :], dim=-1)
    return sum(((c @ c.mT) * (1 - torch.eye(c.shape[-2], device=W.device))).max(-1).values.pow(2).mean() for c in W.chunk(K, -2)) / K


class Block(nn.Module):
    """Pre-LN transformer block with causal SDPA attention."""

    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_head, -1).permute(2, 0, 3, 1, 4)
        x = x + self.proj(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


class LM(nn.Module):
    def __init__(self, V=4096, d=128, L=2, H=2, T=256, k=0, C=0, mix=0, Kc=4, sln=0, **_):
        """V vocab, d width, L layers, H heads, T context; k Backpack senses (0 = off); C concept dictionary size (0 = off);
        mix: interleave compressed concepts; Kc: TopK used for the compression; sln: LayerNorm on the sense-net input (the released
        Backpack code and project/backpacks/13_*.md §2 have it; 0 reproduces the first grid). Extra config keys are ignored."""
        super().__init__()
        self.k, self.C, self.mix, self.Kc = k, C, mix, Kc
        self.E, self.P = nn.Embedding(V, d), nn.Embedding(T, d)  # token embedding (tied with the readout), learned positions
        self.blocks, self.ln_f = nn.ModuleList(Block(d, H) for _ in range(L)), nn.LayerNorm(d)
        if k:  # shared feed-forward sense network + query/key projections for the per-sense attention weights
            self.sense = nn.Sequential(*([nn.LayerNorm(d)] if sln else []), nn.Linear(d, d), nn.GELU(), nn.Linear(d, k * d))
            self.Wq, self.Wk = nn.Linear(d, d), nn.Linear(d, d)
        if C:  # concept head M(h) and compression W
            self.M, self.W = nn.Linear(d, C), nn.Linear(C, d)
        self.apply(lambda m: nn.init.normal_(m.weight, std=.02) if isinstance(m, (nn.Linear, nn.Embedding)) else None)

    def mid(self, x):
        """Residual stream after block 0: the concept insertion point and the site analysed by every SAE."""
        return self.blocks[0](self.E(x) + self.P(torch.arange(x.shape[1], device=x.device)))

    def forward(self, x, y=None, lab=None, sc=None, zadd=None, h=None):
        """x, y: (B, T) token ids; lab: (B, T, Ka) concept labels; sc: (B, T, k) per-(token, sense) scales (intervention);
        zadd: (B, T, C) shift of the concept logits (steering); h: replacement for mid(x) (SAE splicing, attribution).
        Returns logits (B, T, V) and a dict with 'ce' (if y), 'lc' (if lab), 'z' concept logits (if C)."""
        B, T = x.shape
        out = {}
        h = self.mid(x) if h is None else h
        if self.C:
            out['z'] = z = self.M(h) + (0 if zadd is None else zadd)
            if lab is not None:
                out['lc'] = -F.log_softmax(z.float(), -1).gather(-1, lab).mean()
            if self.mix:  # TopK(z) -> linear compression -> interleave (h1, c1, h2, c2, ...)
                v, i = z.topk(self.Kc, -1)
                c = self.W(torch.zeros_like(z).scatter(-1, i, v))
                h = torch.stack([h, c.to(h.dtype)], 2).view(B, 2 * T, -1)
        for block in self.blocks[1:]:
            h = block(h)
        h = self.ln_f(h)
        h_q, h_k = (h[:, 1::2], h[:, 0::2]) if self.mix else (h, h)  # with mixing the LM reads from the concept slots
        if self.k:  # k attention heads whose values are token-only sense vectors; no value/out projection, residual or LN after the sum
            S = self.sense(self.E(x)).view(B, T, self.k, -1).transpose(1, 2)  # (B, k, T, d)
            if sc is not None:
                S = S * sc.transpose(1, 2)[..., None]
            q, k = (f(u).view(B, T, self.k, -1).transpose(1, 2) for f, u in ((self.Wq, h_q), (self.Wk, h_k)))
            h_q = F.scaled_dot_product_attention(q, k, S, is_causal=True).sum(1)  # (B, T, d)
        logits = h_q @ self.E.weight.T
        if y is not None:
            out['ce'] = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        return logits, out

    def pen(self, n=256):
        """Orthogonality penalty: among the k senses of a random token set, and among concept-compression directions."""
        p = 0.
        if self.k:
            tok = torch.randint(self.E.num_embeddings, (n,), device=self.E.weight.device)
            p = p + ort(self.sense(self.E(tok)).view(n, self.k, -1))
        if self.C and self.mix:
            p = p + ort(self.W.weight.T)
        return p
