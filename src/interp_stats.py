"""Measurement error for every headline metric of the 30k, seed-0 runs (inference only, no training).

For each run: error of a mean = std / sqrt(n) over the metric's natural samples
  LM CE / accuracy / concept hit@Ka   n = test tokens          (paired contrasts: per-token differences against the baseline)
  steering KL                         n = concepts steered
  gender reduction                    n = profession words
  SAE EV, CE recovered, Meta-SAE EV   ratio of sums, linearised over tokens / windows / decoder rows
  SAE L0, max-cos                     n = tokens / latents;  probe F1, absorption   n = letters
Also saves the two post-hoc SAEs of every run (same recipe and seed as evaluate.py) for the qualitative figures.
  python interp_stats.py [--runs RUNS_30K] [--out RESULTS_DIR]  ->  errbars.json, errbars.md, saes/<run>.pt, ce_tokens.pt"""
import argparse
import glob
import json
import math
import os

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import evaluate as ev
import paths
from data import D, windows
from model import LM
from sae import train_sae

dev = ev.dev
BASE = 'k0_cm0_-_og0_s0'


def sem(x):
    x = torch.as_tensor(x, dtype=torch.float64)
    return (x.std() / math.sqrt(len(x))).item() if len(x) > 1 else float('nan')


def ratio_sem(num, den):
    """Mean(num)/mean(den) and its standard error by linearisation (per-sample residuals num - r * den)."""
    num, den = num.double().flatten(), den.double().flatten()
    r = num.sum() / den.sum()
    return r.item(), ((num - r * den).pow(2).sum().sqrt() / den.sum().abs()).item()


def load(runs, name):
    ck = torch.load(f'{runs}/{name}/ckpt.pt', map_location=dev)
    m = LM(**ck['cfg']).to(dev).eval()
    m.load_state_dict(ck['sd'])
    return m, ck['cfg']


@torch.no_grad()
def lm_tokens(model, X, Y, L, bs=256):
    """Per-token CE, correctness and (with labels) precision@Ka as flat float32 GPU tensors."""
    ce, ok, hit = [], [], []
    for i in range(0, len(X), bs):
        x, y = X[i:i + bs].to(dev), Y[i:i + bs].to(dev)
        logits, o = model(x, lab=None if L is None else L[i:i + bs].long().to(dev))
        ce.append(F.cross_entropy(logits.flatten(0, 1).float(), y.flatten(), reduction='none'))
        ok.append((logits.argmax(-1) == y).flatten().float())
        if L is not None:
            lab = L[i:i + bs].long().to(dev)
            top = o['z'].topk(lab.shape[-1], -1).indices
            hit.append((top[..., :, None] == lab[..., None, :]).any(-1).float().mean(-1).flatten())
    cat = lambda v: torch.cat(v) if v else None
    return cat(ce), cat(ok), cat(hit)


@torch.no_grad()
def steer_per_concept(model, X, L, n=32, delta=5.):
    """KL(base || steered) per steered concept (the n most frequent), averaged over positions."""
    x = X[:64].to(dev)
    base = F.log_softmax(model(x)[0].float(), -1)
    out = []
    for c in L.flatten().long().bincount(minlength=model.C).topk(min(n, model.C)).indices:
        z = torch.zeros(*x.shape, model.C, device=dev)
        z[..., c] = delta
        out.append(ev.kl(base, F.log_softmax(model(x, zadd=z)[0].float(), -1)))
    return torch.tensor(out)


@torch.no_grad()
def sae_stats(model, first, A, X, Y, g):
    """Same recipe as evaluate.sae_eval, but returns standard errors, the SAE and the decoder."""
    sae = train_sae(A, g=g)
    T = torch.cat([model.mid(x.to(dev)) for x in X[:1024].split(256)]).flatten(0, 1)
    parts = [sae(t) for t in T.split(32768)]  # chunked: the GPU is shared with training runs
    x_hat, C = torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])
    del parts
    mu = T.mean(0, keepdim=True)
    err, var = ((x_hat - T) ** 2).sum(-1), ((T - mu) ** 2).sum(-1)
    ev_, ev_se = ratio_sem(var - err, var)
    l0 = (C > 0).float().sum(-1)
    W = F.normalize(sae.Wd, dim=-1)
    mc = (W @ W.T).fill_diagonal_(-1).max(1).values
    x, y = X[:512].to(dev), Y[:512].to(dev)

    def win_ce(h, bs=64):  # per-window CE with the block-0 stream replaced by h
        out = []
        for i in range(0, len(x), bs):
            lg = model(x[i:i + bs], h=h[i:i + bs])[0].float()
            out.append(F.cross_entropy(lg.flatten(0, 1), y[i:i + bs].flatten(), reduction='none').view(len(lg), -1).mean(1))
        return torch.cat(out)

    ce0 = win_ce(model.mid(x))
    ce_sae, ce_abl = win_ce(torch.cat([sae(t)[0] for t in model.mid(x).flatten(0, 1).split(32768)]).view(len(x), x.shape[1], -1)), win_ce(mu.expand(*x.shape, -1))
    rec, rec_se = ratio_sem(ce_abl - ce_sae, ce_abl - ce0)
    W0 = sae.Wd.detach()
    meta = train_sae(W0, 256, 4, 0., 1000, 1024)
    e_row, v_row = ((meta(W0)[0] - W0) ** 2).sum(-1), ((W0 - W0.mean(0)) ** 2).sum(-1)
    mev, mev_se = ratio_sem(v_row - e_row, v_row)
    pr = ev.probe(T, C, first[X[:1024].flatten().to(dev)])
    f1, ab = [v[0] for v in pr['per_letter'].values()], [v[1] for v in pr['per_letter'].values()]
    r = {'ev': [ev_, ev_se], 'l0': [l0.mean().item(), sem(l0)], 'max_cos': [mc.mean().item(), sem(mc)], 'ce_recovered': [rec, rec_se],
         'meta_ev': [mev, mev_se], 'probe_f1': [sum(f1) / len(f1), sem(f1)], 'absorption': [sum(ab) / len(ab), sem(ab)],
         'per_letter': pr['per_letter'], 'max_cos_values': mc.cpu().tolist()}
    return r, sae.cpu().state_dict()


MAIN = ('k0_cm0_-_og0', 'k0_cm1_topk_og0', 'k0_cm2_topk_og0', 'k0_cm2_topk_og0.1', 'k4_cm0_-_og0', 'k4_cm0_-_og0.1', 'k4_cm2_topk_og0', 'k4_cm2_topk_og0.1')


def main(runs, out, seed=0):
    """seed 0: all runs found (the 23 models); other seeds: the eight main models only. Outputs get the suffix _s{seed} when seed > 0."""
    os.makedirs(f'{out}/saes', exist_ok=True)
    suf = f'_s{seed}' if seed else ''
    tk = Tokenizer.from_file(f'{D}/tok.json')
    first = ev.first_letters(tk)
    names = sorted(os.path.basename(p) for p in glob.glob(f'{runs}/*_s{seed}') if os.path.exists(f'{p}/ckpt.pt') and (seed == 0 or os.path.basename(p).rsplit('_s', 1)[0] in MAIN))
    X, Y = windows('test')
    res, ce_all = {}, {}
    for name in names:
        model, cfg = load(runs, name)
        L = torch.load(f"{cfg['lab']}/test.pt") if cfg['lab'] else None
        ce, ok, hit = lm_tokens(model, X, Y, L)
        ce_all[name] = ce.cpu()
        r = {'n_tokens': len(ce), 'ce': [ce.mean().item(), sem(ce)], 'acc': [ok.mean().item(), sem(ok)],
             'ppl': [math.exp(ce.mean().item()), math.exp(ce.mean().item()) * sem(ce)]}
        if hit is not None:
            r['concept_hit'] = [hit.mean().item(), sem(hit)]
            s = steer_per_concept(model, X, L)
            r['steer_kl'] = [s.mean().item(), sem(s)]
        with open(f'{runs}/{name}/eval_test.json') as f:
            old = json.load(f)
        if model.k:  # per-profession best-sense reduction (best sense chosen on these same words, as in evaluate.py)
            prof = old['gender_prof']
            best = min(range(model.k), key=lambda s: sum(p['zero'][s] for p in prof.values()))
            red = [1 - p['zero'][best] / max(p['base'], 1e-9) for p in prof.values()]
            r['gender_best_reduction'] = [sum(red) / len(red), sem(red)]
            r['n_prof'] = len(red)
        with torch.no_grad():
            Asae = torch.cat([model.mid(x.to(dev)) for x in windows('sae', cfg['T'])[0].split(256)]).flatten(0, 1)
        saes = {}
        for g in (0., .25):
            r[f'sae_g{g}'], saes[g] = sae_stats(model, first, Asae, X, Y, g)
        torch.save(saes, f'{out}/saes/{name}.pt')
        res[name] = r
        print(name, f"ppl {r['ppl'][0]:.3f} +- {r['ppl'][1]:.3f}", flush=True)
        del model
        torch.cuda.empty_cache()
    base = ce_all[f'k0_cm0_-_og0_s{seed}']
    for name, ce in ce_all.items():  # paired per-token difference to the baseline
        d = (ce - base).double()
        res[name]['d_ce_vs_base'] = [d.mean().item(), sem(d)]
    torch.save({k: v.half() for k, v in ce_all.items()}, f'{out}/ce_tokens{suf}.pt')
    with open(f'{out}/errbars{suf}.json', 'w') as f:
        json.dump(res, f)
    with open(f'{out}/errbars{suf}.md', 'w') as f:
        f.write(f'# 30k seed {seed}: metrics with standard error (std / sqrt(n samples))\n\n| run | ppl | CE | dCE vs baseline | hit@4 | steer KL | gender red. |\n|---|---|---|---|---|---|---|\n')
        for n, r in res.items():
            g = lambda k, p=3: f'{r[k][0]:.{p}f} ± {r[k][1]:.{p}f}' if k in r else '-'
            f.write(f"| {n} | {g('ppl')} | {g('ce', 4)} | {g('d_ce_vs_base', 4)} | {g('concept_hit')} | {g('steer_kl')} | {g('gender_best_reduction')} |\n")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default=f'{paths.ROOT}/runs_30k')
    p.add_argument('--out', default=f'{paths.ROOT}/results_30k/interp')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    main(a.runs, a.out, a.seed)
