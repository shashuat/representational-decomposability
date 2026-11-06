"""Step 2 of 2: re-load runs/<name>/ckpt.pt in a fresh process and compute every metric on a held-out split.

Writes eval_<split>.json and sae_decoders.pt next to the checkpoint. Metric groups:
  LM        ce / ppl / next-token accuracy
  concept   precision@Ka of the concept head vs attribution labels; steering KL (+delta on one concept logit)
  backpack  sense geometry, top-token projections of each sense, gender sense-zeroing (effect and collateral KL)
  SAE       a TopK SAE (g=0) and an OrtSAE (g>0) trained on the block-0 residual stream (split 'sae', never the eval split):
            EV, L0, dead, CE-recovered, KL-score, decoder max-cos, Meta-SAE EV, first-letter probe F1 and absorption proxy
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from data import D, windows
from model import LM, ort
from sae import train_sae

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
PROFS = 'doctor teacher nurse farmer baker artist pilot chef driver singer writer builder scientist dancer officer cook'.split()
SENSE_WORDS = 'he she boy girl dog happy ball'.split()


def timed(times, name, f, *a, **k):
    """Run f, record its wall time (CUDA-synchronised) in times[name], return its result."""
    if dev == 'cuda':
        torch.cuda.synchronize()
    t = time.time()
    r = f(*a, **k)
    if dev == 'cuda':
        torch.cuda.synchronize()
    times[name] = time.time() - t
    return r


def kl(logp, logq):
    """KL(p || q) averaged over positions, from log-probabilities (..., V)."""
    return (logp.exp() * (logp - logq)).sum(-1).mean().item()


def first_letters(tk):
    """Index 0-25 of the first letter of every vocabulary token (ASCII letters only), -1 otherwise."""
    out = []
    for i in range(tk.get_vocab_size()):
        t = (tk.id_to_token(i) or '').lstrip('Ġ')
        out.append(ord(t[0].lower()) - 97 if t[:1].isascii() and t[:1].isalpha() else -1)
    return torch.tensor(out, device=dev)


@torch.no_grad()
def lm_metrics(model, X, Y, L, bs=128):
    """Test CE / perplexity / accuracy and, with labels L (n, T, Ka), concept CE and precision@Ka."""
    tot = torch.zeros(4, dtype=torch.float64, device=dev)  # ce, acc, lc, hit: accumulated on the GPU, one host transfer at the end
    n = 0
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(dev), Y[i:i + bs].to(dev)
        lab = None if L is None else L[i:i + bs].long().to(dev)
        logits, o = model(x, lab=lab)
        tot[0] += F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), reduction='sum')
        tot[1] += (logits.argmax(-1) == y).sum()
        n += y.numel()
        if lab is not None:
            top = o['z'].topk(lab.shape[-1], -1).indices
            tot[2] += o['lc'] * y.numel()
            tot[3] += (top[..., :, None] == lab[..., None, :]).any(-1).float().mean(-1).sum()
    tot = (tot / n).tolist()
    r = {'ce': tot[0], 'acc': tot[1]}
    r['ppl'] = math.exp(r['ce'])
    if L is not None:
        r.update(concept_ce=tot[2], concept_hit=tot[3])
    return r


@torch.no_grad()
def steer(model, X, L, n=32, delta=5.):
    """Mean KL(base || steered) of the next-token distribution when adding +delta to one concept logit (n most frequent concepts)."""
    x = X[:64].to(dev)
    base = F.log_softmax(model(x)[0].float(), -1)
    out = []
    for c in L.flatten().long().bincount(minlength=model.C).topk(min(n, model.C)).indices:
        z = torch.zeros(*x.shape, model.C, device=dev)
        z[..., c] = delta
        out.append(kl(base, F.log_softmax(model(x, zadd=z)[0].float(), -1)))
    return sum(out) / len(out)


@torch.no_grad()
def sense_stats(model, tk):
    """Geometry of the k sense vectors of every token plus the top-8 vocabulary projection of each sense for a few words."""
    V, k = model.E.num_embeddings, model.k
    S = model.sense(model.E.weight).view(V, k, -1)
    Sn = F.normalize(S, dim=-1)
    G = Sn @ Sn.mT
    top = {}
    for w in SENSE_WORDS:
        if (i := tk.token_to_id('Ġ' + w)) is not None:
            top[w] = [[tk.id_to_token(j) for j in (S[i, ell] @ model.E.weight.T).topk(8).indices.tolist()] for ell in range(k)]
    return {'sense_cos': ((G.sum((-1, -2)) - k) / (k * (k - 1))).mean().item(), 'sense_maxcos2': ort(S).item(), 'sense_top8': top}


@torch.no_grad()
def gender(model, tk):
    """Zero one sense of the profession token ('The <prof> said that'); measure |log p(he)/p(she)| and the KL cost over the whole next-token distribution."""
    he, she = tk.token_to_id('Ġhe'), tk.token_to_id('Ġshe')
    if he is None or she is None:
        return {}
    k, prof = model.k, {}
    for w in PROFS:
        if len(tk.encode(' ' + w).ids) != 1:
            continue
        ids = tk.encode(f'The {w} said that').ids
        x, pos = torch.tensor([ids], device=dev), ids.index(tk.token_to_id('Ġ' + w))
        lp = F.log_softmax(model(x)[0][0, -1].float(), -1)
        rec = {'base': (lp[he] - lp[she]).abs().item(), 'zero': [], 'kl': []}
        for ell in range(k):
            sc = torch.ones(1, len(ids), k, device=dev)
            sc[0, pos, ell] = 0
            lq = F.log_softmax(model(x, sc=sc)[0][0, -1].float(), -1)
            rec['zero'].append((lq[he] - lq[she]).abs().item())
            rec['kl'].append(kl(lp, lq))
        prof[w] = rec
    mean = lambda vs: sum(vs) / max(len(vs), 1)
    zero = [mean([p['zero'][ell] for p in prof.values()]) for ell in range(k)]
    base = mean([p['base'] for p in prof.values()])
    return {'gender_bias': base, 'gender_bias_zero_sense': zero, 'gender_best_reduction': 1 - min(zero) / max(base, 1e-9),
            'gender_kl_zero_sense': [mean([p['kl'][ell] for p in prof.values()]) for ell in range(k)], 'gender_prof': prof}


@torch.no_grad()
def probe(A, C, lab):
    """First-letter task. Dense reference: ridge probe on raw activations A. SAE side: 1-sparse probe = the latent whose mean activation
    differs most between letter-c tokens and the rest. Absorption proxy = fraction of tokens the dense probe gets right on which that
    latent is silent. Returns means over letters and a per-letter breakdown."""
    keep = lab >= 0
    A, C, lab = A[keep], C[keep], lab[keep]
    A1 = torch.cat([A, torch.ones_like(A[:, :1])], 1)
    ok = (A1 @ torch.linalg.lstsq(A1, F.one_hot(lab, 26).float()).solution).argmax(1) == lab
    per = {}
    for c in range(26):
        pos = lab == c
        if pos.sum() < 20:
            continue
        on = C[:, (C[pos].mean(0) - C[~pos].mean(0)).argmax()] > 0
        tp, fp, fn = (on & pos).sum(), (on & ~pos).sum(), (~on & pos).sum()
        f1 = (2 * tp / (2 * tp + fp + fn).clamp(min=1)).item()
        absorb = (((~on) & pos & ok).sum() / (pos & ok).sum().clamp(min=1)).item()
        per[chr(97 + c)] = [f1, absorb]
    return {'probe_dense_acc': ok.float().mean().item(), 'probe_f1': sum(v[0] for v in per.values()) / len(per),
            'absorption': sum(v[1] for v in per.values()) / len(per), 'per_letter': per}


@torch.no_grad()
def sae_eval(model, first, A, Xt, Yt, g, sae_kw):
    """Train an SAE (OrtSAE if g > 0) on A, then score it on test windows. Returns metrics and the decoder matrix."""
    times = {}
    sae = timed(times, 'train_sae_s', train_sae, A, g=g, **sae_kw)
    t_score = time.time()
    acts = lambda X: torch.cat([model.mid(x.to(dev)) for x in X.split(256)]).flatten(0, 1)
    T = acts(Xt[:1024])
    parts = [sae(t) for t in T.split(32768)]  # chunked: the GPU may be shared with training runs
    x_hat, C = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    del parts
    W = F.normalize(sae.Wd, dim=-1)
    mu = T.mean(0, keepdim=True)
    x, y = Xt[:512].to(dev), Yt[:512].to(dev)
    bs = 64

    def spliced(hf):  # CE and KL(clean || patched) with the block-0 stream replaced by hf(x chunk); chunk means over equal-sized chunks = overall means
        ces, kls = [], []
        for i in range(0, len(x), bs):
            xc, yc = x[i:i + bs], y[i:i + bs]
            lg0, _ = model(xc, yc)
            lg, o = model(xc, yc, h=None if hf is None else hf(xc))
            ces.append(o['ce'].item())
            kls.append(kl(F.log_softmax(lg0.float(), -1), F.log_softmax(lg.float(), -1)))
        return sum(ces) / len(ces), sum(kls) / len(kls)

    ce0 = spliced(None)[0]
    ce_sae, kl_sae = spliced(lambda xc: torch.cat([sae(t)[0] for t in model.mid(xc).flatten(0, 1).split(32768)]).view(*xc.shape, -1))
    ce_abl, kl_abl = spliced(lambda xc: mu.expand(*xc.shape, -1))  # mean-ablation reference
    W0 = sae.Wd.detach()
    meta = timed(times, 'meta_sae_s', train_sae, W0, 256, 4, 0., 1000, 1024)  # Meta-SAE: how well do decoder rows decompose into other rows?
    r = {'ev': 1 - ((x_hat - T) ** 2).sum().item() / ((T - mu) ** 2).sum().item(),
         'l0': (C > 0).float().sum(-1).mean().item(), 'dead': 1 - (C > 0).any(0).float().mean().item(),
         'ce_recovered': (ce_abl - ce_sae) / max(ce_abl - ce0, 1e-9), 'kl_score': 1 - kl_sae / max(kl_abl, 1e-9),
         'mean_max_cos': (W @ W.T).fill_diagonal_(-1).max(1).values.mean().item(),
         'meta_ev': 1 - ((meta(W0)[0] - W0) ** 2).sum().item() / ((W0 - W0.mean(0)) ** 2).sum().item()}
    r.update(timed(times, 'probe_s', probe, T, C, first[Xt[:1024].flatten().to(dev)]))
    times['score_s'] = time.time() - t_score - times['meta_sae_s'] - times['probe_s']
    return {**r, 'time': times}, W0.cpu()


def main(ckpt, split='test', gs=(0., .25), **sae_kw):
    t0, times = time.time(), {}
    ck = torch.load(ckpt, map_location=dev)
    cfg = ck['cfg']
    model = LM(**cfg).to(dev).eval()
    model.load_state_dict(ck['sd'])
    tk = Tokenizer.from_file(f'{D}/tok.json')
    X, Y = windows(split, cfg['T'])
    L = torch.load(f"{cfg['lab']}/{split}.pt") if cfg['lab'] else None
    times['load_s'] = time.time() - t0
    r = {'name': cfg['name'], 'split': split, 'n_params': sum(p.numel() for p in model.parameters()), **timed(times, 'lm_metrics_s', lm_metrics, model, X, Y, L)}
    if L is not None and model.C:
        r['steer_kl'] = timed(times, 'steer_s', steer, model, X, L)
    if model.k:
        r.update(timed(times, 'sense_s', sense_stats, model, tk), **timed(times, 'gender_s', gender, model, tk))
    with torch.no_grad():
        A = timed(times, 'acts_s', lambda: torch.cat([model.mid(x.to(dev)) for x in windows('sae', cfg['T'])[0].split(256)]).flatten(0, 1))
    first, decoders = first_letters(tk), {}
    for g in gs:
        r[f'sae_g{g}'], decoders[g] = sae_eval(model, first, A, X, Y, g, sae_kw)
    times['total_s'] = time.time() - t0
    r['time'] = times
    out_dir = os.path.dirname(ckpt)
    torch.save(decoders, f'{out_dir}/sae_decoders.pt')
    with open(f'{out_dir}/eval_{split}.json', 'w') as f:
        json.dump(r, f, indent=1)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, (dict, list))}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt')
    p.add_argument('--split', default='test')
    p.add_argument('--gs', type=float, nargs='+', default=[0., .25])
    p.add_argument('--sae', type=json.loads, default={}, help='train_sae overrides as JSON, e.g. {"m": 64, "K": 4, "steps": 20}')
    a = p.parse_args()
    main(a.ckpt, a.split, a.gs, **a.sae)
