"""CoCoMix concept teacher.

Trains an SAE (TopK or OrtSAE) on a trained baseline's block-0 residual stream (split 'sae'), then caches per-token concept labels for the
train/dev/test windows: the top-Ka latents by attribution (dL/dh . W_dec) * code, as in the released CoCoMix code
(signed values; --sign -1 keeps the loss-reducing concepts instead)."""
import argparse
import os

import torch

from data import windows
from model import LM
from sae import train_sae


def attrib(lm, sae, x, y, Ka=4, sign=1):
    """x, y: (B, T) -> (B, T, Ka) int16 latent indices. Chain rule: dL/dcode = W_dec . dL/dh for h = code @ W_dec."""
    with torch.no_grad():
        h = lm.mid(x)
    grad, = torch.autograd.grad(lm(x, y, h=h.requires_grad_())[1]['ce'], h)
    with torch.no_grad():
        return (sign * grad @ sae.Wd.T * sae(h)[1]).topk(Ka, -1).indices.short()


def main(teacher, out, g=0., m=512, K=16, Ka=4, steps=3000, sign=1):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck = torch.load(teacher, map_location=dev)
    cfg = ck['cfg']
    lm = LM(**cfg).to(dev).eval()
    lm.load_state_dict(ck['sd'])
    with torch.no_grad():
        A = torch.cat([lm.mid(x.to(dev)) for x in windows('sae', cfg['T'])[0].split(256)]).flatten(0, 1)
    sae = train_sae(A, m, K, g, steps)
    os.makedirs(out, exist_ok=True)
    torch.save(sae.state_dict(), f'{out}/sae.pt')
    for split in ('train', 'dev', 'test'):
        X, Y = windows(split, cfg['T'])
        labels = [attrib(lm, sae, x.to(dev), y.to(dev), Ka, sign).cpu() for x, y in zip(X.split(128), Y.split(128))]
        torch.save(torch.cat(labels), f'{out}/{split}.pt')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--teacher')
    p.add_argument('--out')
    p.add_argument('--g', type=float, default=0.)
    p.add_argument('--m', type=int, default=512)
    p.add_argument('--K', type=int, default=16)
    p.add_argument('--Ka', type=int, default=4)
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--sign', type=int, default=1)
    main(**vars(p.parse_args()))
