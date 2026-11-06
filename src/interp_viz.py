"""Interpretability analyses on the 30k-step, seed-0 checkpoints (inference only). Figures -> $RDP_ROOT/figures_30k/interp, LaTeX tables -> report/generated/ti_*.tex,
numbers -> $RDP_ROOT/results_30k/interp/*.json.   python interp_viz.py [analysis ...]    (no argument = all)

Common probes (every model):   units (readability of SAE latents / concept units), geometry (nearest-neighbour cosine of the exposed units), steer (intervention response)
Backpack:                      senses (vocabulary projection + sense roles), decomp (exact per-sense logit decomposition), gender
CoCoMix:                       concepts (concept dictionary + concept stream over a story), cgen (steered generations), wnorm (compression weights)
OrtSAE:                        sdecomp (Meta-SAE-style decomposition of a TopK feature), absorb (first-letter absorption per letter)
Error bars everywhere are std / sqrt(n) over the samples named in the caption (interp_stats.py documents the sample of each metric)."""
import glob
import json
import math
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import nnls
from scipy.stats import spearmanr
from tokenizers import Tokenizer

import evaluate as ev
import paths
from data import D, windows
from model import LM
from sae import SAE

dev = ev.dev
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = int(os.environ.get('SEED', '0'))  # seed 0 = the full set of 23 models and the report's qualitative figures; seeds 1-4 = the eight main models, numbers only (OUT/seed<k>/*.json)
SUF = f'_s{SEED}' if SEED else ''
RUNS, OUT, FIG = f'{paths.ROOT}/runs_30k', f'{paths.ROOT}/results_30k/interp', f'{paths.ROOT}/figures_30k/interp'
OUTS, FIG = OUT + (f'/seed{SEED}' if SEED else ''), FIG + (f'/seed{SEED}' if SEED else '')
GEN = f'{HERE}/report/generated'
CELLS = {'P': 'k0_cm0_-_og0', 'CC': 'k0_cm1_topk_og0', 'CM': 'k0_cm2_topk_og0', 'CMo': 'k0_cm2_topk_og0.1', 'BP': 'k4_cm0_-_og0', 'BPo': 'k4_cm0_-_og0.1', 'BC': 'k4_cm2_topk_og0',
         'BCo': 'k4_cm2_topk_og0.1'}  # the eight main models
N = {k: f'{c}_s{SEED}' for k, c in CELLS.items()}
BLUE, ORANGE, AQUA, VIOLET, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#0b0b0b', '#52514e', '#e6e5e1'
SENSE_COL = [BLUE, ORANGE, AQUA, VIOLET]
plt.switch_backend('Agg')
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'legend.frameon': False, 'pdf.fonttype': 42})
TK = Tokenizer.from_file(f'{D}/tok.json')
_models = {}


# ---------------------------------------------------------------- helpers

def model(key):
    n = N.get(key, key)
    if n not in _models:
        ck = torch.load(f'{RUNS}/{n}/ckpt.pt', map_location=dev)
        _models[n] = LM(**ck['cfg']).to(dev).eval()
        _models[n].load_state_dict(ck['sd'])
    return _models[n]


def sae_of(key, g):
    """Post-hoc SAE (g = 0 TopK, .25 OrtSAE) saved by interp_stats.py."""
    s = SAE(128, 1024, 16)
    s.load_state_dict(torch.load(f'{OUT}/saes/{N.get(key, key)}.pt')[g])
    return s.to(dev).eval()


def codes_of(sae, A, chunk=32768):
    """SAE codes of many activations, computed in chunks (the GPU is shared)."""
    with torch.no_grad():
        return torch.cat([sae(a)[1] for a in A.split(chunk)])


def tok(i):
    """Display string of a vocabulary id: word-initial pieces plain, continuation pieces with a leading '-'; non-ASCII byte pieces shown as ?."""
    t = TK.id_to_token(int(i))
    if t == '<eos>':
        return t
    init = t.startswith('Ġ')
    t = ''.join(ch if ord(ch) < 128 else ('⏎' if ch == 'Ċ' else '?') for ch in (t[1:] if init else t))
    return t if init else ('-' + t if t[:1].isalpha() else t)


def tex(s):
    for a, b in (('\\', r'\textbackslash{}'), ('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'), ('_', r'\_'), ('{', r'\{'), ('}', r'\}'), ('~', r'\~{}'), ('^', r'\^{}'),
                 ('⏎', r'$\hookleftarrow$')):
        s = s.replace(a, b)
    return s


def ids(text):
    return TK.encode(text).ids


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f'{FIG}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{FIG}/{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def load_eval(name):
    with open(f'{RUNS}/{name}/eval_test.json') as f:
        return json.load(f)


def dump(name, obj):
    os.makedirs(OUTS, exist_ok=True)
    with open(f'{OUTS}/{name}.json', 'w') as f:
        json.dump(obj, f, indent=1)


def write_tex(name, body):
    if SEED:  # tables come from seed 0 (or from interp_agg.py)
        return
    os.makedirs(GEN, exist_ok=True)
    with open(f'{GEN}/{name}.tex', 'w') as f:
        f.write(body)


def sem(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float('nan')


_cache = {}


def data():
    """Test windows, block-0 activations of the baseline, teacher labels and bigram / co-occurrence statistics shared by several analyses."""
    if 'X' not in _cache:
        X, Y = windows('test')
        _cache.update(X=X, Y=Y, L=torch.load(f'{RUNS}/teacher/topk/test.pt'))
    return _cache['X'], _cache['Y'], _cache['L']


@torch.no_grad()
def mid_acts(m, n=1024):
    X = data()[0][:n]
    return torch.cat([m.mid(x.to(dev)) for x in X.split(256)]).flatten(0, 1)


def concept_stats(top=8, minc=15):
    """Per teacher concept: frequency in the labels, and the tokens it predicts (largest log-lift of P(next | concept) over P(next), count >= minc)."""
    if 'conc' not in _cache:
        X, Y, L = data()
        V, C = 4096, 512
        y = Y.flatten().to(dev)
        lab = L.reshape(-1, L.shape[-1]).long().to(dev)
        cnt = sum(torch.bincount(lab[:, s] * V + y, minlength=C * V) for s in range(lab.shape[1])).view(C, V).float()
        freq = cnt.sum(1) / len(y)
        uni = torch.bincount(y, minlength=V).float() / len(y)
        lift = torch.log(cnt / cnt.sum(1, keepdim=True).clamp(min=1) / uni.clamp(min=1e-9)).masked_fill(cnt < minc, -1e9)
        _cache['conc'] = {'freq': freq.cpu(), 'lift': lift.topk(top, -1).indices.cpu(), 'liftv': lift.topk(top, -1).values.cpu(), 'cnt': cnt.sum(1).cpu()}
    return _cache['conc']


def signature(codes, toks, minc=8, top=4):
    """Unit signature from activations: tokens with the highest mean activation (>= minc occurrences); purity = share of the unit's activation mass on its top token."""
    V = 4096
    S = torch.zeros(V, codes.shape[1], device=dev).index_add_(0, toks, codes.float())
    n = torch.bincount(toks, minlength=V).float()
    mean = (S / n.clamp(min=1)[:, None]).masked_fill((n < minc)[:, None], -1)
    return mean.T.topk(top, -1).indices.cpu(), (S.max(0).values / S.sum(0).clamp(min=1e-9)).cpu(), S.sum(0).cpu()


def teacher_sae():
    s = SAE(128, 512, 16)
    s.load_state_dict(torch.load(f'{RUNS}/teacher/topk/sae.pt'))
    return s.to(dev).eval()


@torch.no_grad()
def teacher_codes(n=1024):
    """Teacher SAE codes. The teacher SAE was trained on the 10k-step seed-0 baseline (the model that produced the labels), so it is applied to that model's block-0 stream."""
    ck = torch.load(f'{paths.RUNS}/k0_cm0_-_og0_s0/ckpt.pt', map_location=dev)
    m = LM(**ck['cfg']).to(dev).eval()
    m.load_state_dict(ck['sd'])
    return teacher_sae()(mid_acts(m, n))[1]


def nn_cos(W):
    W = F.normalize(W.float(), dim=-1)
    return (W @ W.T).fill_diagonal_(-1).max(1).values.cpu().numpy()


# ---------------------------------------------------------------- common probe 1: readability of units

@torch.no_grad()
def units():
    """How token-specific are the units each model exposes? Top-token share of activation mass per unit (higher = more word-like)."""
    X = data()[0][:1024].flatten().to(dev)
    res, curves = {}, {}
    for key in ('P', 'CM', 'BP'):
        m = model(key)
        A = mid_acts(m)
        for g, nm in ((0., 'TopK'), (.25, 'OrtSAE')):
            C = codes_of(sae_of(key, g), A)
            _, pur, mass = signature(C, X)
            live = mass > 0
            curves[(key, nm)] = pur[live].numpy()
            res[f'{key}_{nm}'] = {'purity_mean': float(pur[live].mean()), 'purity_sem': sem(pur[live]), 'n_units': int(live.sum())}
    for key in ('CC', 'CM', 'BC'):
        m = model(key)
        z = m.M(mid_acts(m))
        v, i = z.topk(4, -1)
        C = torch.zeros_like(z).scatter(-1, i, v.clamp(min=0))
        _, pur, mass = signature(C, X)
        live = mass > 0
        curves[(key, 'concept')] = pur[live].numpy()
        res[f'{key}_concept'] = {'purity_mean': float(pur[live].mean()), 'purity_sem': sem(pur[live]), 'n_units': int(live.sum())}
    _, pur, mass = signature(teacher_codes(), X)
    curves[('teacher', 'concept')] = pur[mass > 0].numpy()
    res['teacher'] = {'purity_mean': float(pur[mass > 0].mean()), 'purity_sem': sem(pur[mass > 0]), 'n_units': int((mass > 0).sum())}
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    col = {'P': BLUE, 'CM': ORANGE, 'BP': AQUA, 'CC': VIOLET, 'BC': AQUA, 'teacher': MUTED}
    lab = {'P': 'plain LM', 'CM': 'CoCoMix LM', 'BP': 'Backpack LM', 'CC': 'concept CE only', 'BC': 'Backpack + CoCoMix', 'teacher': 'teacher SAE latents'}
    for (key, nm), v in curves.items():
        a = ax[0] if nm in ('TopK', 'OrtSAE') else ax[1]
        a.plot(np.sort(v), np.arange(len(v)) / len(v), color=col[key], ls='--' if nm == 'OrtSAE' else '-', lw=1.3,
               label=f"{lab[key]}{' ' + nm if nm in ('TopK', 'OrtSAE') else ''}")
    for a, t in zip(ax, ('post-hoc SAE latents (1024)', 'concept units (512)')):
        a.set_xlabel('top-token share of activation mass')
        a.set_title(t, fontsize=8)
        a.legend(fontsize=6.2, loc='lower right')
    ax[0].set_ylabel('fraction of units $\\leq$ x')
    save(fig, 'ri_units')
    dump('units', res)
    return res


# ---------------------------------------------------------------- common probe 2: geometry of the exposed units

@torch.no_grad()
def geometry():
    """Nearest-neighbour cosine of the exposed units: SAE decoder rows, per-word sense vectors, concept compression directions."""
    out, panels = {}, []
    dm = lambda key: {g: F.normalize(sae_of(key, g).Wd.detach(), dim=-1) for g in (0., .25)}
    d = dm('P')
    panels.append(('SAE decoder rows', {'TopK SAE': nn_cos(d[0.]), 'OrtSAE': nn_cos(d[.25])}, 1024, 128))
    sv = {}
    for key, nm in (('BP', 'no penalty'), ('BPo', 'penalty')):
        m = model(key)
        S = m.sense(m.E.weight).view(4096, 4, -1)
        sv[nm] = np.concatenate([nn_cos(s) for s in S])  # nearest other sense of the same word
    panels.append(('senses of one word', sv, 4, 128))
    cv = {nm: nn_cos(model(key).W.weight.T) for key, nm in (('CM', 'no penalty'), ('CMo', 'penalty'))}
    panels.append(('concept directions', cv, 512, 128))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    cols = [BLUE, ORANGE]
    for a, (title, series, n, dd) in zip(ax, panels):
        rnd = nn_cos(torch.randn(n, dd)) if n > 4 else np.concatenate([nn_cos(torch.randn(4, dd)) for _ in range(2000)])
        for (nm, v), c in zip(series.items(), cols):
            a.hist(v, bins=40, range=(-.2, 1), density=True, color=c, alpha=.55, label=f'{nm}: {v.mean():.3f} $\\pm$ {sem(v):.3f}', lw=0)
            out[f'{title}|{nm}'] = [float(v.mean()), sem(v), len(v)]
        a.axvline(rnd.mean(), color=MUTED, ls=':', lw=1, label=f'random: {rnd.mean():.3f}')
        out[f'{title}|random'] = [float(rnd.mean()), sem(rnd), len(rnd)]
        a.set_title(title, fontsize=7.5)
        a.set_xlabel('max cosine to another unit')
        a.legend(fontsize=6.3, loc='upper right')
        a.set_yticks([])
    ax[0].set_ylabel('density')
    save(fig, 'ri_geometry')
    dump('geometry', out)
    return out


# ---------------------------------------------------------------- common probe 3: intervention response

@torch.no_grad()
def logmass(logits, S):
    """log of the next-token probability mass on the token set S (index tensor), per position."""
    return torch.logsumexp(F.log_softmax(logits.float(), -1)[..., S], -1)


@torch.no_grad()
def steer_concepts(key, units, tokens, strengths, nwin=24):
    """Concept steering (+delta on one concept logit, all positions). Returns (len(strengths), n_units, n_units): mean change in log-mass of every unit's target set when unit u is steered."""
    m = model(key)
    x = data()[0][:nwin].to(dev)
    base = [logmass(m(x)[0], S) for S in tokens]
    R = torch.zeros(len(strengths), len(units), len(units))
    for a, d in enumerate(strengths):
        for u, c in enumerate(units):
            z = torch.zeros(*x.shape, m.C, device=dev)
            z[..., c] = d
            lg = m(x, zadd=z)[0]
            for v, S in enumerate(tokens):
                R[a, u, v] = (logmass(lg, S) - base[v]).mean().item()
    return R


@torch.no_grad()
def steer_senses(key, words, tokens, strengths, nwin=100):
    """Sense scaling: multiply sense l of every occurrence of word w by s. R[a, (w,l), (w,l')] = change in log-mass of the target set of (w,l') at the positions of w."""
    m = model(key)
    X = data()[0]
    units = [(w, l) for w in words for l in range(m.k)]
    R = torch.zeros(len(strengths), len(units), len(units))
    for wi, w in enumerate(words):
        win = (w == X).any(1).nonzero().flatten()[:nwin]
        x = X[win].to(dev)
        pos = x == w
        logits0 = m(x)[0]
        for a, s in enumerate(strengths):
            for l in range(m.k):
                sc = torch.ones(*x.shape, m.k, device=dev)
                sc[..., l] = 1 + (s - 1) * pos.float()
                lg = m(x, sc=sc)[0]
                for l2 in range(m.k):
                    S = tokens[wi * m.k + l2]
                    R[a, wi * m.k + l, wi * m.k + l2] = (logmass(lg, S) - logmass(logits0, S))[pos].mean().item()
    return R, units


def sense_targets(m, words, top=8):
    S = m.sense(m.E.weight).view(4096, m.k, -1)
    return [(S[w, l] @ m.E.weight.T).topk(top).indices for w in words for l in range(m.k)]


@torch.no_grad()
def steer_latents(key, g, units, strengths, S_sets, codes_ref, nwin=24):
    """SAE-latent steering on the block-0 stream of an LM: add a * mean-active-code * decoder row (in the SAE's input scale)."""
    m, sae = model(key), sae_of(key, g)
    x = data()[0][:nwin].to(dev)
    h0 = m.mid(x)
    lg0 = m(x)[0]
    base = [logmass(lg0, S) for S in S_sets]
    R = torch.zeros(len(strengths), len(units), len(units))
    for a, s in enumerate(strengths):
        for u, j in enumerate(units):
            h = h0 + s * codes_ref[j] * sae.Wd[j] / sae.sf
            lg = m(x, h=h)[0]
            for v, S in enumerate(S_sets):
                R[a, u, v] = (logmass(lg, S) - base[v]).mean().item()
    return R


def present_targets(P, y, minc=15, top=8):
    """Target token set per unit from a presence matrix P (N, units) bool: tokens with the highest log-lift of P(next | unit on) over P(next)."""
    V = 4096
    cnt = torch.zeros(P.shape[1], V, device=dev)
    for u in range(P.shape[1]):
        cnt[u] = torch.bincount(y[P[:, u]], minlength=V).float()
    uni = torch.bincount(y, minlength=V).float() / len(y)
    lift = torch.log(cnt / cnt.sum(1, keepdim=True).clamp(min=1) / uni.clamp(min=1e-9)).masked_fill(cnt < minc, -1e9)
    ok = (lift > -1e8).sum(1) >= top
    return lift.topk(top, -1).indices, ok


def on_off(R):
    """R (n_strengths, n_units, n_units): on-target (diagonal) and off-target (mean of the others) response per strength, with SEM over units."""
    n = R.shape[1]
    eye = torch.eye(n, dtype=bool)
    on = R[:, eye]
    off = torch.stack([R[a][~eye].view(n, n - 1).mean(1) for a in range(len(R))])
    return on.numpy(), off.numpy()


@torch.no_grad()
def steer(n_units=24):
    """Intervention response in the three decomposable interfaces: concept logit (CoCoMix), sense scale (Backpack), SAE latent (any LM)."""
    X, Y, L = data()
    y = Y.flatten().to(dev)
    out, fig_data = {}, {}
    # (b) concepts
    cs = concept_stats()
    top_c = [c for c in cs['freq'].topk(80).indices.tolist() if cs['liftv'][c, -1] > -1e8][:n_units]
    sets = [cs['lift'][c].to(dev) for c in top_c]
    st_c = [-5., 0., 2.5, 5., 10.]
    for key in ('CM', 'BC', 'CC'):
        fig_data[('concept', key)] = (st_c, *on_off(steer_concepts(key, top_c, sets, st_c)))
    # (a) senses
    words = [i for i in torch.bincount(X.flatten(), minlength=4096).topk(120).indices.tolist() if TK.id_to_token(i).startswith('Ġ') and TK.id_to_token(i)[1:].isalpha()][:16]
    st_s = [0., .5, 1., 1.5, 2., 3.]
    for key in ('BP', 'BPo', 'BC'):
        R, units_ = steer_senses(key, words, sense_targets(model(key), words), st_s)
        fig_data[('sense', key)] = (st_s, *on_off(R))
    # (c) SAE latents of the plain LM and of the CoCoMix LM
    st_l = [0., 1., 2., 4., 8.]
    for key in ('P', 'CM'):
        A = mid_acts(model(key))
        for g, nm in ((0., 'TopK'), (.25, 'OrtSAE')):
            sae = sae_of(key, g)
            C = codes_of(sae, A)
            P = C > 0
            freq = P.float().mean(0)
            cand = freq.topk(300).indices.tolist()
            tg, ok = present_targets(P[:, cand], y[:len(P)])
            sel = [cand[i] for i in ok.nonzero().flatten().tolist()][:n_units]
            sets_l = [tg[cand.index(j)] for j in sel]
            mean_act = (C.sum(0) / (C > 0).sum(0).clamp(min=1))
            fig_data[('latent', f'{key}_{nm}')] = (st_l, *on_off(steer_latents(key, g, sel, st_l, sets_l, mean_act)))
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
    styles = {'concept': (ax[1], {'CM': (ORANGE, 'CoCoMix'), 'BC': (AQUA, 'Backpack + CoCoMix'), 'CC': (VIOLET, 'concept CE only')}, 'concept logit shift'),
              'sense': (ax[0], {'BP': (AQUA, 'Backpack'), 'BPo': (BLUE, 'Backpack + penalty'), 'BC': (ORANGE, 'Backpack + CoCoMix')}, 'sense scale'),
              'latent': (ax[2], {'P_TopK': (BLUE, 'plain, TopK SAE'), 'P_OrtSAE': (VIOLET, 'plain, OrtSAE'), 'CM_TopK': (ORANGE, 'CoCoMix, TopK SAE'), 'CM_OrtSAE': (AQUA, 'CoCoMix, OrtSAE')},
                         'latent strength (x mean active code)')}
    for (kind, key), (xs, on, off) in fig_data.items():
        a, cm, xl = styles[kind]
        c, lb = cm[key]
        mu, se = on.mean(1), on.std(1, ddof=1) / math.sqrt(on.shape[1])
        a.errorbar(xs, mu, se, color=c, marker='o', ms=3, lw=1.2, capsize=2, label=lb)
        a.plot(xs, off.mean(1), color=c, ls=':', lw=1)
        a.set_xlabel(xl)
        out[f'{kind}|{key}'] = {'x': xs, 'on': mu.tolist(), 'on_sem': se.tolist(), 'off': off.mean(1).tolist(), 'n_units': on.shape[1]}
    ax[0].set_ylabel('change in log target mass')
    ax[1].plot([], [], color=MUTED, ls=':', lw=1, label='dotted: off-target sets')
    for a, t in zip(ax, ('Backpack senses', 'CoCoMix concepts', 'post-hoc SAE latents')):
        a.set_title(t, fontsize=8)
        a.legend(fontsize=6.2, loc='upper left')
    save(fig, 'ri_steer')
    dump('steer', out)
    return out


# ---------------------------------------------------------------- Backpack: sense vocabulary projections and sense roles

def proj(m, w):
    """(k, V) projection of the k sense vectors of token w on the output embedding (exactly how each sense moves the logits), mean-centred."""
    with torch.no_grad():
        p = m.sense(m.E.weight[w][None]).view(m.k, -1) @ m.E.weight.T
    return p - p.mean(-1, keepdim=True)


def wid(w):
    return TK.token_to_id('Ġ' + w)


def lm_stats(n=20_000_000, win=4):
    """Bigram log P(next | w) and positive PMI of co-occurrence within +-win tokens, from the first n training tokens."""
    if 'bigram' not in _cache:
        a = torch.from_numpy(np.fromfile(f'{D}/train.bin', dtype=np.uint16, count=n).astype(np.int64)).to(dev)
        V = 4096
        big = torch.bincount(a[:-1] * V + a[1:], minlength=V * V).view(V, V).float()
        co = sum(torch.bincount(a[:-s] * V + a[s:], minlength=V * V).view(V, V).float() for s in range(1, win + 1))
        co = co + co.T
        pw = co.sum(1)
        pmi = torch.log(co * co.sum() / (pw[:, None] * pw[None] + 1e-9) + 1e-9).clamp(min=0)
        _cache['bigram'] = (torch.log((big + .5) / (big + .5).sum(1, keepdim=True)).cpu(), pmi.cpu(), torch.bincount(a, minlength=V).cpu())
    return _cache['bigram']


def corr(a, b):
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-9)


def senses(words=('dog', 'happy', 'ball', 'girl', 'play', 'mom'), top=6, n_words=300):
    """Top vocabulary tokens of every sense of a few words (Backpack paper's sense tables) and the role of each sense index over the 300 most frequent words."""
    out = {}
    rows = {}
    for key in ('BP', 'BPo'):
        m = model(key)
        for w in words:
            p = proj(m, wid(w))
            rows[(key, w)] = [[tok(i) for i in p[l].topk(top).indices.tolist()] for l in range(m.k)]
            out[f'{key}|{w}'] = rows[(key, w)]
    for fname, ws in (('ti_senses', words[:2]), ('ti_senses_full', words)):
        b = [r'\begin{tabular}{@{}llll@{}}', r'\toprule', r'word & sense & Backpack $k{=}4$ & + penalty ($\gamma_o{=}0.1$) \\', r'\midrule']
        for w in ws:
            for l in range(4):
                b.append(f"{tex(w) if l == 0 else ''} & {l} & {tex(' '.join(rows[('BP', w)][l]))} & {tex(' '.join(rows[('BPo', w)][l]))} \\\\")
            b.append(r'\addlinespace')
        write_tex(fname, '\n'.join(b[:-1] + [r'\bottomrule', r'\end{tabular}']) + '\n')
    big, pmi, cnt = lm_stats()
    ws = [i for i in cnt.topk(600).indices.tolist() if TK.id_to_token(i).startswith('Ġ') and TK.id_to_token(i)[1:].isalpha()][:n_words]
    roles = {}
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    cm = plt.matplotlib.colors.LinearSegmentedColormap.from_list('d', [BLUE, '#f2f2f0', ORANGE])
    for a, (key, ttl) in zip(ax, (('BP', 'Backpack'), ('BPo', 'Backpack + penalty'), ('BC', 'Backpack + CoCoMix'))):
        m = model(key)
        P = torch.stack([proj(m, w) for w in ws]).to('cpu')  # (W, k, V)
        z = (P - P.mean(-1, keepdim=True)) / P.std(-1, keepdim=True)
        self_ = torch.stack([z[i, :, w] for i, w in enumerate(ws)])  # (W, k)
        nxt = corr(P, big[ws][:, None])
        rel = corr(P, pmi[ws][:, None])
        R = torch.stack([self_ / 10, nxt, rel], -1)  # (W, k, 3); self-score divided by 10 to share the colour scale
        mu, se = R.mean(0).numpy(), R.std(0).numpy() / math.sqrt(len(ws))
        a.imshow(mu, cmap=cm, vmin=-.5, vmax=.5, aspect='auto')
        for i in range(m.k):
            for j in range(3):
                f_ = 10 if j == 0 else 1
                a.text(j, i, f'{mu[i, j] * f_:.2f}\n$\\pm${se[i, j] * f_:.2f}', ha='center', va='center', fontsize=6.5)
        a.set_xticks(range(3), ['own token\n(z-score)', 'next token\n(bigram corr)', 'related words\n(PMI corr)'], fontsize=6.5)
        a.set_yticks(range(m.k), [f'sense {l}' for l in range(m.k)] if key == 'BP' else [])
        a.set_title(ttl, fontsize=8)
        a.grid(False)
        roles[key] = {'mean': (R.mean(0) * torch.tensor([10., 1, 1])).tolist(), 'sem': (R.std(0) * torch.tensor([10., 1, 1]) / math.sqrt(len(ws))).tolist(), 'n_words': len(ws)}
    save(fig, 'ri_roles')
    out['roles'] = roles
    dump('senses', out)
    return out


# ---------------------------------------------------------------- Backpack: exact decomposition of one prediction

@torch.no_grad()
def bp_final(m, x):
    """Query and key streams (B, T, d) that set the sense weights of a Backpack LM (with interleaved concepts the queries come from the concept slots)."""
    h = m.mid(x)
    B, T = x.shape
    if m.C and m.mix:
        z = m.M(h)
        v, i = z.topk(m.Kc, -1)
        h = torch.stack([h, m.W(torch.zeros_like(z).scatter(-1, i, v))], 2).view(B, 2 * T, -1)
    for b in m.blocks[1:]:
        h = b(h)
    h = m.ln_f(h)
    return (h[:, 1::2], h[:, 0::2]) if m.mix else (h, h)


@torch.no_grad()
def bp_parts(m, x):
    """Per-sense attention weights att (k, T, T) and per-(sense, source token) vocabulary projections proj (k, T, V) of a Backpack LM (batch of one)."""
    h_q, h_k = bp_final(m, x)
    T = x.shape[1]
    S = m.sense(m.E(x)).view(1, T, m.k, -1).transpose(1, 2)[0]
    q, k = (f(u).view(1, T, m.k, -1).transpose(1, 2)[0] for f, u in ((m.Wq, h_q), (m.Wk, h_k)))
    att = (q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).masked_fill(~torch.ones(T, T, dtype=bool, device=x.device).tril(), -1e9).softmax(-1)
    return att, S @ m.E.weight.T


@torch.no_grad()
def decomp(prompt='One day, a boy and his dog went to the park. The dog', n_top=6):
    """Which source words, through which sense, make the Backpack predict its top token; contributions sum exactly to the logit."""
    m = model('BP')
    x = torch.tensor([ids(prompt)], device=dev)
    att, P = bp_parts(m, x)
    i = x.shape[1] - 1
    C = att[:, i, :, None] * P  # (k, T, V): contribution of (sense, source token j) to the logits at position i
    logits = m(x)[0][0, i].float()
    err = (C.sum((0, 1)) - logits).abs().max().item()
    top = logits.topk(n_top).indices
    fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.8), gridspec_kw={'width_ratios': [1.15, 1.15, 1.5]})
    words = [tok(t) for t in x[0].tolist()]
    cm = plt.matplotlib.colors.LinearSegmentedColormap.from_list('d', [BLUE, '#f2f2f0', ORANGE])
    A = att[:, i].T.cpu().numpy()
    ax[0].imshow(A, cmap='Greys', vmin=0, vmax=1, aspect='auto')
    ax[0].set_title('(a) sense attention $\\alpha_{\\ell,i,j}$', fontsize=7.5)
    G = C[:, :, top[0]].T.cpu().numpy()
    v = np.abs(G).max()
    ax[1].imshow(G, cmap=cm, vmin=-.6 * v, vmax=.6 * v, aspect='auto')
    ax[1].set_title(f'(b) contribution to logit of "{tok(top[0])}"', fontsize=7.5)
    for a in ax[:2]:
        a.set_yticks(range(len(words)), words, fontsize=6.5)
        a.set_xticks(range(m.k), [f'{l}' for l in range(m.k)])
        a.set_xlabel('sense')
        a.grid(False)
    Cs = C[:, :, top].sum(1).cpu().numpy()  # (k, n_top)
    pos, neg = np.zeros(n_top), np.zeros(n_top)
    for l in range(m.k):  # stacked: the bars add up to the logit
        v = Cs[l]
        bot = np.where(v >= 0, pos, neg)
        ax[2].bar(np.arange(n_top), v, .6, bottom=bot, color=SENSE_COL[l], label=f'sense {l}')
        pos, neg = pos + np.maximum(v, 0), neg + np.minimum(v, 0)
    ax[2].plot(np.arange(n_top), logits[top].cpu().numpy(), 'k_', ms=11, mew=1.6, label='logit')
    ax[2].set_xticks(range(n_top), [tok(t) for t in top], rotation=35, fontsize=7)
    ax[2].set_title('(c) the senses add up to the logit', fontsize=7.5)
    ax[2].legend(fontsize=6, ncol=3, loc='upper right', columnspacing=.8, handlelength=1)
    ax[2].set_ylim(top=ax[2].get_ylim()[1] * 1.25)
    save(fig, 'ri_decomp')
    res = {'prompt': prompt, 'top': [tok(t) for t in top], 'max_abs_sum_error': err, 'logit_spread': float(logits.std())}
    dump('decomp', res)
    return res


# ---------------------------------------------------------------- Backpack: gender sense zeroing

def gender():
    """Zero one sense of a profession word and read the he/she log-odds gap (7 single-token professions); collateral KL on the whole next-token distribution."""
    out, fig, ax = {}, *plt.subplots(1, 3, figsize=(7.6, 2.5), gridspec_kw={'wspace': .45})
    cfg = [('BP', 'Backpack', AQUA), ('BPo', '+ penalty', BLUE), ('BC', '+ CoCoMix', ORANGE), ('BCo', '+ CoCoMix + penalty', VIOLET)]
    w = .2
    for mi, (key, ttl, c) in enumerate(cfg):
        prof = load_eval(N[key])['gender_prof']
        base = np.array([p['base'] for p in prof.values()])
        zero = np.array([p['zero'] for p in prof.values()])  # (prof, k)
        red = (base[:, None] - zero).sum(0) / base.sum()  # fraction of the summed gap removed; linearised standard error over professions
        se = np.sqrt(((base[:, None] - zero - red * base[:, None]) ** 2).sum(0)) / base.sum()
        kl = np.array([p['kl'] for p in prof.values()])
        ax[0].bar(np.arange(4) + (mi - 1.5) * w, red, w, yerr=se, color=c, label=ttl, error_kw={'lw': .8, 'capsize': 1.5})
        ax[1].bar(np.arange(4) + (mi - 1.5) * w, kl.mean(0), w, yerr=kl.std(0, ddof=1) / math.sqrt(len(kl)), color=c, error_kw={'lw': .8, 'capsize': 1.5})
        out[key] = {'reduction_mean': red.tolist(), 'reduction_sem': se.tolist(), 'kl_mean': kl.mean(0).tolist(), 'n_prof': len(base), 'professions': list(prof)}
        if key == 'BP':
            b = int(red.argmax())
            for name in prof:
                ax[2].plot([0, 1], [prof[name]['base'], prof[name]['zero'][b]], color=AQUA, marker='o', ms=3, lw=1)
                ax[2].text(1.03, prof[name]['zero'][b], name, fontsize=6, va='center')
            out['BP']['best_sense'] = b
    for a in ax[:2]:
        a.set_xticks(range(4), range(4))
        a.set_xlabel('sense zeroed')
    ax[0].set_ylabel('fraction of he/she gap removed')
    ax[1].set_ylabel('KL(clean || zeroed), nats')
    ax[2].set_xticks([0, 1], ['clean', 'best sense\nzeroed'])
    ax[2].set_ylabel('$|\\log p(\\mathrm{he}) / p(\\mathrm{she})|$')
    ax[2].set_xlim(-.1, 1.5)
    ax[0].legend(fontsize=6, loc='upper left')
    for a, t in zip(ax, ('(a) effect of zeroing one sense', '(b) collateral change', '(c) Backpack, per profession')):
        a.set_title(t, fontsize=7.5)
    save(fig, 'ri_gender')
    m = model('BP')
    rows = []
    for wd in ('doctor', 'nurse', 'teacher'):
        p = proj(m, wid(wd))[out['BP']['best_sense']]
        rows.append(f"{tex(wd)} & {tex(' '.join(tok(i) for i in p.topk(5).indices.tolist()))} & {tex(' '.join(tok(i) for i in (-p).topk(5).indices.tolist()))} \\\\")
    he, she = wid('he'), wid('she')
    write_tex('ti_gender_sense', '\n'.join([r'\begin{tabular}{@{}lll@{}}', r'\toprule', r'profession & most up-weighted by the sense & most down-weighted \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    out['pronoun_projection'] = {wd: [float(proj(m, wid(wd))[out['BP']['best_sense']][t]) for t in (he, she)] for wd in ('doctor', 'nurse', 'teacher')}
    dump('gender', out)
    return out


# ---------------------------------------------------------------- CoCoMix: concept dictionary, concept stream, steered generation, compression weights

@torch.no_grad()
def concept_recall(key, n=4096, bs=256):
    """Per teacher concept: P(concept in the head's top-4 | concept is a label) and the number of labelled positions."""
    m = model(key)
    X, _, L = data()
    hit, tot = torch.zeros(512, device=dev), torch.zeros(512, device=dev)
    for i in range(0, n, bs):
        x, lab = X[i:i + bs].to(dev), L[i:i + bs].long().to(dev)
        top = m.M(m.mid(x)).topk(4, -1).indices
        pred = F.one_hot(top, 512).sum(-2).clamp(max=1).flatten(0, 1)
        lm = F.one_hot(lab, 512).sum(-2).clamp(max=1).flatten(0, 1)
        hit += (pred * lm).sum(0)
        tot += lm.sum(0)
    return (hit / tot.clamp(min=1)).cpu().numpy(), tot.cpu().numpy()


def concepts(n_rows=12):
    """Concept dictionary of the CoCoMix LM (teacher index space): what a concept fires on, what it predicts, how well the head recovers it."""
    cs = concept_stats()
    X = data()[0][:1024].flatten().to(dev)
    sig_in, pur, _ = signature(teacher_codes(), X)
    rec = {k: concept_recall(k) for k in ('CM', 'BC', 'CC')}
    pool = [c for c in cs['freq'].topk(150).indices.tolist() if cs['liftv'][c, 7] > -1e8]
    frequent = pool[:5]  # the five most frequent concepts, then the seven of the 150 most frequent with the largest mean lift of their predicted tokens
    sharp = sorted(pool[5:], key=lambda c: -float(cs['liftv'][c].mean()))[:7]
    top = frequent + sharp

    def row(c):
        r, n = rec['CM'][0][c], rec['CM'][1][c]
        return (f"{c} & {100 * cs['freq'][c]:.1f} & {tex(' '.join(tok(i) for i in sig_in[c]))} & {tex(' '.join(tok(i) for i in cs['lift'][c][:4].tolist()))} & "
                f"{r:.2f} $\\pm$ {math.sqrt(r * (1 - r) / max(n, 1)):.3f} \\\\")

    rows = [row(c) for c in frequent[:2]] + [r'\midrule'] + [row(c) for c in sharp]
    write_tex('ti_concepts', '\n'.join([r'\begin{tabular}{@{}rrllr@{}}', r'\toprule', r'concept & positions (\%) & fires on & predicts next & recall (CoCoMix) \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'width_ratios': [1, 1.1], 'wspace': 1.0})
    sel = cs['freq'].topk(100).indices.numpy()
    out = {}
    for key, c, lb in (('CC', VIOLET, 'concept CE only'), ('CM', ORANGE, 'CoCoMix'), ('BC', AQUA, 'Backpack + CoCoMix')):
        r, n = rec[key]
        f_ = cs['freq'][sel].numpy()
        edges = np.quantile(f_, np.linspace(0, 1, 6))
        bi = np.clip(np.searchsorted(edges, f_, side='right') - 1, 0, 4)
        mu_ = np.array([r[sel][bi == i].mean() for i in range(5)])
        se_ = np.array([sem(r[sel][bi == i]) for i in range(5)])
        ax[0].errorbar(np.arange(5) + {'CC': -.08, 'CM': 0, 'BC': .08}[key], mu_, se_, fmt='o-', ms=3, lw=1, color=c, capsize=1.5, label=f'{lb}: {r[sel].mean():.2f} $\\pm$ {sem(r[sel]):.2f}')
        ax[0].set_xticks(range(5), [f'{100 * np.mean(f_[bi == i]):.1f}%' for i in range(5)], fontsize=6.5)
        out[key] = {'mean_recall_top100': float(r[sel].mean()), 'sem': sem(r[sel]), 'n_concepts': 100}
    ax[0].set_xlabel('concept label frequency (binned)')
    ax[0].set_ylabel('recall of the concept by the head')
    ax[0].legend(fontsize=6.3, loc='upper center', bbox_to_anchor=(.5, -.2))
    ax[0].set_title('(a) 100 most frequent concepts', fontsize=7.5)
    # (b) stream over one story excerpt: the window with most activity in the readable concepts
    m = model('CM')
    X0, _, L0 = data()
    nw, T0 = 80, 26
    xs = X0[:nw, :T0].to(dev)
    with torch.no_grad():
        preds = m.M(m.mid(xs)).topk(4, -1).indices.cpu()
    readable = set(top[5:])
    score = [sum(int(c) in readable for c in torch.cat([L0[i, :T0].flatten(), preds[i].flatten()]).tolist()) for i in range(nw)]
    w = int(np.argmax(score))
    x, pred, lab = xs[w:w + 1], preds[w], L0[w, :T0].long()
    cons = [c for c in top[5:] if any(c in lab[t].tolist() or c in pred[t].tolist() for t in range(T0))]
    M = np.zeros((len(cons), T0))
    for t in range(T0):
        for r_, c in enumerate(cons):
            M[r_, t] = (c in lab[t].tolist()) + 2 * (c in pred[t].tolist())
    cmap = plt.matplotlib.colors.ListedColormap(['white', '#c3c2b7', '#f5b9a0', ORANGE])
    ax[1].imshow(M, cmap=cmap, vmin=0, vmax=3, aspect='auto')
    ax[1].set_xticks(range(T0), [tok(t) for t in x[0].tolist()], rotation=90, fontsize=6)
    ax[1].set_yticks(range(len(cons)), [f"c{c}: {' '.join(tok(i) for i in cs['lift'][c][:3].tolist())}" for c in cons], fontsize=6)
    ax[1].grid(False)
    ax[1].set_title('(b) one story excerpt, CoCoMix head', fontsize=7.5)
    ax[1].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in ('#c3c2b7', '#f5b9a0', ORANGE)], labels=['teacher label only', 'predicted only', 'both'], fontsize=6, loc='upper center', bbox_to_anchor=(.5, -.35), ncol=3)
    save(fig, 'ri_concepts')
    out['rows'] = top
    dump('concepts', out)
    return out


@torch.no_grad()
def generate(m, x, c=None, delta=8., n=20):
    """Greedy continuation of n tokens; with c, +delta on concept logit c at every position (prompt and generated)."""
    for _ in range(n):
        z = None
        if c is not None:
            z = torch.zeros(*x.shape, m.C, device=dev)
            z[..., c] = delta
        x = torch.cat([x, m(x, zadd=z)[0][:, -1].argmax(-1, keepdim=True)], 1)
    return x


def cgen(show=(109, 414, 393), n_prompts=32, delta=6.):
    """Steered generation: continuation with a concept boosted at every position vs unsteered; share of generated tokens in the concept's predicted-token set (mean over prompts, SEM)."""
    cs = concept_stats()
    X = data()[0]
    P = X[100:100 + n_prompts, :12].to(dev)
    out, rows = {}, []
    story = torch.tensor([ids('One day, a little girl named Lily went to the')], device=dev)
    for key in ('CM', 'BC'):
        m = model(key)
        for c in show:
            S = set(cs['lift'][c].tolist())
            rate = {}
            for d, cc in ((0., None), (delta, c)):
                g = generate(m, P, cc, d)[:, 12:]
                r = np.array([np.mean([t in S for t in row]) for row in g.tolist()])
                rate[d] = (float(r.mean()), sem(r))
            txt = [TK.decode(generate(m, story, cc, d)[0, story.shape[1]:].tolist()) for d, cc in ((0., None), (delta, c))]
            out[f'{key}|{c}'] = {'set': [tok(i) for i in cs['lift'][c].tolist()], 'rate_base': rate[0.], 'rate_steered': rate[delta], 'text': txt}
            rows.append((key, c, rate, txt))
    b = [r'\begin{tabular}{@{}lp{2.9cm}p{7.2cm}@{}}', r'\toprule', r'model & concept (predicts next) & continuation \\', r'\midrule']
    for key in ('CM',):  # the table shows the CoCoMix LM; the Backpack + CoCoMix texts are in cgen.json
        name = 'CoCoMix'
        mine = [r for r in rows if r[0] == key]
        b.append(f"{name} & none (unsteered) & {tex(mine[0][3][0].strip())} \\\\")
        for _, c, _, txt in mine:
            b.append(f" & {c}: {tex(' '.join(tok(i) for i in cs['lift'][c][:3].tolist()))} & {tex(txt[1].strip())} \\\\")
        b.append(r'\addlinespace')
    write_tex('ti_cgen', '\n'.join(b[:-1] + [r'\bottomrule', r'\end{tabular}']) + '\n')
    dump('cgen', out)
    return out


def wnorm():
    """Norm of every concept's compression column W[:, c] against how often the concept is a label (CoCoMix paper's weight-norm analysis)."""
    cs = concept_stats()
    out, fig, ax = {}, *plt.subplots(1, 2, figsize=(6.2, 2.4))
    for key, c, lb in (('CM', ORANGE, 'CoCoMix'), ('BC', AQUA, 'Backpack + CoCoMix')):
        nrm = model(key).W.weight.norm(dim=0).detach().cpu().numpy()
        f = cs['freq'].numpy()
        lab = f > 0
        ax[0].scatter(f[lab] + 1e-7, nrm[lab], s=5, color=c, alpha=.6, label=f'{lb} (rho = {spearmanr(f[lab], nrm[lab])[0]:.2f})')
        ax[1].hist(nrm, bins=30, color=c, alpha=.5, label=lb, lw=0)
        out[key] = {'rho_freq_norm': float(spearmanr(f[lab], nrm[lab])[0]), 'norm_labelled': [float(nrm[lab].mean()), sem(nrm[lab])], 'norm_unlabelled': [float(nrm[~lab].mean()) if (~lab).any() else None, sem(nrm[~lab]) if (~lab).sum() > 1 else None],
                    'n_labelled': int(lab.sum()), 'n_unlabelled': int((~lab).sum())}
    ax[0].set_xscale('log')
    ax[0].set_xlabel('label frequency')
    ax[0].set_ylabel('$\\|W_{:,c}\\|_2$')
    ax[1].set_xlabel('$\\|W_{:,c}\\|_2$')
    ax[1].set_ylabel('concepts')
    for a in ax:
        a.legend(fontsize=6.2)
    save(fig, 'ri_wnorm')
    dump('wnorm', out)
    return out


# ---------------------------------------------------------------- OrtSAE: decomposition of a TopK feature, absorption

def nnls_decomp(w, D, max_comp=3, min_coef=.25):
    """Greedy non-negative decomposition of w over the rows of D (unit norm): returns (indices, coefficients, cosine of the approximation)."""
    idx, r = [], w.copy()
    for _ in range(max_comp):
        s = D @ r
        s[idx] = -1
        j = int(s.argmax())
        if s[j] <= 0:
            break
        idx.append(j)
        coef, _ = nnls(D[idx].T, w)
        r = w - coef @ D[idx]
        if (coef < min_coef).any():
            idx = [i for i, c in zip(idx, coef) if c >= min_coef]
            if not idx:
                break
            coef, _ = nnls(D[idx].T, w)
            r = w - coef @ D[idx]
    if not idx:
        return [], [], 0.
    a = coef @ D[idx]
    return idx, coef.tolist(), float(a @ w / (np.linalg.norm(a) * np.linalg.norm(w) + 1e-9))


@torch.no_grad()
def sdecomp(key='P', thr=.9, n_show=4):
    """Cross-dictionary decomposition (OrtSAE paper, Queen example): can a TopK decoder row be written as <= 3 non-negative OrtSAE rows with cosine > thr, and vice versa?"""
    m = model(key)
    A, X = mid_acts(m), data()[0][:1024].flatten().to(dev)
    sa = {g: sae_of(key, g) for g in (0., .25)}
    W = {g: F.normalize(sa[g].Wd.detach(), dim=-1).cpu().numpy() for g in sa}
    codes = {g: codes_of(sa[g], A) for g in sa}
    sig = {g: signature(codes[g], X) for g in sa}
    out = {}
    for src, dst, nm in ((0., .25, 'TopK -> OrtSAE'), (.25, 0., 'OrtSAE -> TopK')):
        res = [nnls_decomp(W[src][j], W[dst]) for j in range(1024)]
        ok = np.array([r[2] > thr and len(r[0]) >= 2 for r in res])
        out[nm] = {'rate': float(ok.mean()), 'sem': float(math.sqrt(ok.mean() * (1 - ok.mean()) / len(ok))), 'n_latents': len(ok), 'thr': thr}
        if src == 0.:
            alive = sig[0.][2].numpy() > 0
            pick = [j for j in np.argsort(-sig[0.][2].numpy()) if ok[j] and alive[j]][:n_show]
            rows = []
            for j in pick[:3]:
                idx, coef, cs_ = res[j]
                comp = ' + '.join(f"{c:.2f}$\\times$[{tex(' '.join(tok(i) for i in sig[.25][0][u][:3].tolist()))}]" for u, c in zip(idx, coef))
                rows.append(f"{{}}[{tex(' '.join(tok(i) for i in sig[0.][0][j][:3].tolist()))}] & {comp} & {cs_:.2f} \\\\")
            write_tex('ti_sdecomp', '\n'.join([r'\begin{tabular}{@{}lll@{}}', r'\toprule', r'TopK latent (top tokens) & OrtSAE components (coefficient $\times$ top tokens) & cosine \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
            out['examples'] = [{'latent': int(j), 'tokens': [tok(i) for i in sig[0.][0][j].tolist()], 'components': [(int(u), c, [tok(i) for i in sig[.25][0][u].tolist()]) for u, c in zip(res[j][0], res[j][1])]} for j in pick]
    dump('sdecomp', out)
    return out


@torch.no_grad()
def absorb(keys=('P', 'CM', 'BP')):
    """First-letter probe: per-letter feature F1 and absorption (dense probe right, 1-sparse SAE probe silent), TopK vs OrtSAE, with binomial standard errors per letter."""
    X = data()[0][:1024].flatten().to(dev)
    first = ev.first_letters(TK)[X]
    out = {}
    fig, ax = plt.subplots(1, len(keys), figsize=(2.5 * len(keys) + .6, 2.6), sharey=True, sharex=True)
    for a, key in zip(ax, keys):
        m = model(key)
        A = mid_acts(m)
        keep = first >= 0
        A1 = torch.cat([A, torch.ones_like(A[:, :1])], 1)[keep]
        lab = first[keep]
        ok = (A1 @ torch.linalg.lstsq(A1, F.one_hot(lab, 26).float()).solution).argmax(1) == lab
        per = {}
        for g in (0., .25):
            C = codes_of(sae_of(key, g), A)[keep]
            for c in range(26):
                pos = lab == c
                if pos.sum() < 20:
                    continue
                on = C[:, (C[pos].mean(0) - C[~pos].mean(0)).argmax()] > 0
                n_ok = int((pos & ok).sum())
                p = ((~on) & pos & ok).sum().item() / max(n_ok, 1)
                per.setdefault(chr(97 + c), {})[g] = (p, math.sqrt(p * (1 - p) / max(n_ok, 1)))
        xs = np.array([v[0.][0] for v in per.values()])
        ys = np.array([v[.25][0] for v in per.values()])
        ex = np.array([v[0.][1] for v in per.values()])
        ey = np.array([v[.25][1] for v in per.values()])
        a.errorbar(xs, ys, ey, ex, fmt='o', ms=3, color=BLUE, elinewidth=.5, alpha=.8)
        a.plot([0, 1], [0, 1], color=MUTED, ls=':', lw=.8)
        a.set_title(f"{ {'P': 'plain', 'CM': 'CoCoMix', 'BP': 'Backpack'}[key] } LM: {xs.mean():.3f} vs {ys.mean():.3f}", fontsize=7.5)
        a.set_xlabel('absorption, TopK SAE')
        d = ys - xs
        out[key] = {'topk': [float(xs.mean()), sem(xs)], 'ort': [float(ys.mean()), sem(ys)], 'diff': [float(d.mean()), sem(d)], 'n_letters': len(xs), 'per_letter': {k: [v[0.][0], v[.25][0]] for k, v in per.items()}}
    ax[0].set_ylabel('absorption, OrtSAE')
    save(fig, 'ri_absorb')
    dump('absorb', out)
    return out


@torch.no_grad()
def reliance(n=512, bs=128):
    """Does the language model use its concept channel? Test cross-entropy when the concept logits of every position are replaced by those of the same position in another window (shuffled),
    against the intact model; s.e. over windows (paired). 0 by construction when the channel is not fed back (concept CE only)."""
    X, Y, _ = data()
    out = {}
    for key in ('CC', 'CM', 'CMo', 'BC', 'BCo'):
        m = model(key)
        d = []
        for i in range(0, n, bs):
            x, y = X[i:i + bs].to(dev), Y[i:i + bs].to(dev)
            z0 = m.M(m.mid(x))
            ce = lambda lg: F.cross_entropy(lg.float().flatten(0, 1), y.flatten(), reduction='none').view(len(x), -1).mean(1)
            d.append(ce(m(x, zadd=z0.roll(1, 0) - z0)[0]) - ce(m(x)[0]))
        d = torch.cat(d).cpu().numpy()
        out[key] = {'d_ce_shuffled': float(d.mean()), 'sem': sem(d), 'n_windows': len(d)}
    dump('reliance', out)
    return out


def faith(n=1536, chunk=48):
    """Faithfulness of explanations: for one random position per test window, the model's own top prediction, and one unit (a source-sense term of a Backpack, or an active post-hoc SAE latent), compare the
    effect the explanation PREDICTS for removing the unit (Backpack: the exact additive term; latent: first-order gradient x change) with the effect of ACTUALLY removing it (a second forward pass), on the logit."""
    X = data()[0][:n].to(dev)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    pos_all = torch.randint(16, 256, (n,), generator=gen).to(dev)
    out, pts = {}, {}
    for key in ('P', 'CM', 'BP', 'BC'):
        m, sae = model(key), sae_of(key, 0.)
        for p_ in m.parameters():
            p_.requires_grad_(False)
        res = {'latent': ([], []), 'sense': ([], [])}
        for i in range(0, n, chunk):
            x, pos = X[i:i + chunk], pos_all[i:i + chunk]
            B, T = x.shape
            ar = torch.arange(B, device=dev)
            with torch.no_grad():
                h0 = m.mid(x)
                C = sae(h0.flatten(0, 1))[1].view(B, T, -1)[ar, pos]
            j = torch.multinomial((C > 0).float() + 1e-9, 1).squeeze(1)
            delta = -(C[ar, j][:, None] * sae.Wd[j].detach() / sae.sf)
            h = h0.clone().requires_grad_(True)
            lg = m(x, h=h)[0][ar, pos]
            tgt = lg.argmax(-1)
            val = lg[ar, tgt]
            val.sum().backward()
            pred = (h.grad[ar, pos] * delta).sum(-1)
            with torch.no_grad():
                h2 = h0.clone()
                h2[ar, pos] += delta
                act = m(x, h=h2)[0][ar, pos, tgt] - val
            res['latent'][0].append(pred.detach().cpu())
            res['latent'][1].append(act.cpu())
            if m.k:
                with torch.no_grad():
                    hq_, hk_ = bp_final(m, x)
                    S = m.sense(m.E(x)).view(B, T, m.k, -1).transpose(1, 2)  # (B, k, T, d)
                    q, k_ = (f(u).view(B, T, m.k, -1).transpose(1, 2) for f, u in ((m.Wq, hq_), (m.Wk, hk_)))
                    att = (q @ k_.transpose(-1, -2) / math.sqrt(q.shape[-1])).masked_fill(~torch.ones(T, T, dtype=bool, device=dev).tril(), -1e9).softmax(-1)
                    tgt_s = m(x)[0][ar, pos].argmax(-1)
                    term = att[ar, :, pos] * (S @ m.E.weight[tgt_s][:, None, :, None]).squeeze(-1)  # (B, k, T): contribution of (sense, source) to the target logit at pos
                    term = term.masked_fill(torch.arange(T, device=dev)[None, None] > pos[:, None, None], 0)
                    flat = torch.multinomial(term.abs().flatten(1) + 1e-9, 1).squeeze(1)
                    l, jj = flat // T, flat % T
                    sc = torch.ones(B, T, m.k, device=dev)
                    sc[ar, jj, l] = 0
                    lg0 = m(x)[0][ar, pos, tgt_s]
                    actual = m(x, sc=sc)[0][ar, pos, tgt_s] - lg0
                res['sense'][0].append((-term[ar, l, jj]).cpu())
                res['sense'][1].append(actual.cpu())
        for kind, (p_, a_) in res.items():
            if not p_:
                continue
            p_, a_ = torch.cat(p_).numpy(), torch.cat(a_).numpy()
            slope = float((p_ * a_).sum() / (p_ ** 2).sum())
            out[f'{key}|{kind}'] = {'r': float(np.corrcoef(p_, a_)[0, 1]), 'slope': slope, 'rel_err': float(np.median(np.abs(a_ - p_) / (np.abs(a_) + 1e-3))), 'n': len(p_),
                                    'max_abs_err': float(np.abs(a_ - p_).max())}
            if SEED == 0:
                pts[f'{key}|{kind}'] = [p_[:400].tolist(), a_[:400].tolist()]
        for p_ in m.parameters():
            p_.requires_grad_(True)
    dump('faith', {'stats': out, 'points': pts})
    return out


# ---------------------------------------------------------------- common qualitative example: one prompt, one goal, every model

PROMPT = 'One day, a boy and his dog went to the park. The dog'
GOAL = ('barked', 'wagged', 'bark', 'ran', 'jumped', 'played', 'licked', 'chased', 'barking', 'chase')  # what a dog does: the goal of every edit below
BUDGET = .5  # collateral KL (nats, next-token distribution at the edited position) every edit may spend
GRID = {'sense': [1., 1.25, 1.5, 2., 2.5, 3., 4., 5., 6., 8., 12., 16.], 'concept': [0., 1., 2., 3., 4., 6., 8., 12., 16., 24.], 'latent': [0., .5, 1., 2., 3., 4., 6., 8., 16., 32.]}


@torch.no_grad()
def latent_table(key, n=256):
    """Post-hoc TopK SAE of a model's block-0 stream: next-token target tokens per latent (if enough evidence), frequency, mean active code, signature tokens."""
    A = mid_acts(model(key), n)
    sae = sae_of(key, 0.)
    C = torch.cat([sae(a)[1] for a in A.split(16384)])
    P = C > 0
    tg, ok = present_targets(P, data()[1][:n].flatten().to(dev))
    return {'sae': sae, 'tg': tg, 'ok': ok, 'freq': P.float().mean(0), 'mact': C.sum(0) / P.sum(0).clamp(min=1), 'sig': signature(C, data()[0][:n].flatten().to(dev))[0]}


def pick(desc, ok, prio, goal):
    """The unit whose description (list of token ids) shares most tokens with the goal; ties go to the higher priority; None if no overlap."""
    ov = [(len(set(d) & goal) if o else 0, p, i) for i, (d, o, p) in enumerate(zip(desc, ok, prio))]
    best = max(ov)
    return best[2] if best[0] > 0 else None


@torch.no_grad()
def edit_unit(key, x, kind, u, lt=None):
    """Last-position edit of one unit at every strength of GRID[kind]: returns the next-token log-probabilities, shape (n_strengths, V)."""
    m, ss = model(key), torch.tensor(GRID[kind], device=dev)
    B, T = len(ss), x.shape[1]
    xb = x.expand(B, -1)
    if kind == 'sense':
        sc = torch.ones(B, T, m.k, device=dev)
        sc[:, -1, u] = ss
        return F.log_softmax(m(xb, sc=sc)[0][:, -1].float(), -1)
    if kind == 'concept':
        z = torch.zeros(B, T, m.C, device=dev)
        z[:, -1, u] = ss
        return F.log_softmax(m(xb, zadd=z)[0][:, -1].float(), -1)
    h = m.mid(xb).clone()
    h[:, -1] += ss[:, None] * lt['mact'][u] * lt['sae'].Wd[u].detach() / lt['sae'].sf
    return F.log_softmax(m(xb, h=h)[0][:, -1].float(), -1)


def at_budget(lp0, lp, gl):
    """Goal mass of the strongest edit whose collateral KL stays within BUDGET: (goal mass, index of the strength, KL)."""
    kl = (lp0.exp() * (lp0 - lp)).sum(-1)
    pg = lp[:, gl].exp().sum(-1)
    i = int(pg.masked_fill(kl > BUDGET, -1).argmax())
    return float(pg[i]), i, float(kl[i])


@torch.no_grad()
def common():
    """One prompt, one goal (make the dog do something dogs do), every model, every lever it has. Each lever is used two ways: the unit found by reading descriptions (its top tokens overlap the goal words
    most) and the best unit of that interface (oracle over all units); edits are pushed as far as BUDGET nats of collateral KL allow. Also: the last word of the prompt read through each interface."""
    cs = concept_stats()
    goal = {wid(w) for w in GOAL}
    gl = torch.tensor(sorted(goal), device=dev)
    x = torch.tensor([ids(PROMPT)], device=dev)
    res, read = {}, {}
    for key in N:
        m = model(key)
        lp0 = F.log_softmax(m(x)[0][0, -1].float(), -1)
        res[key] = {'base_top': [(tok(i), round(float(lp0[i].exp()), 3)) for i in lp0.topk(5).indices.tolist()], 'p_goal_base': float(lp0[gl].exp().sum())}
        lt = latent_table(key)
        P = proj(m, int(x[0, -1])) if m.k else None
        levers = {}
        if m.k:
            levers['sense'] = (m.k, [P[l].topk(12).indices.tolist() for l in range(m.k)], [True] * m.k, [0.] * m.k, None, lambda u: [tok(i) for i in P[u].topk(8).indices.tolist()])
        if m.C:
            levers['concept'] = (m.C, [cs['lift'][c].tolist() for c in range(m.C)], (cs['liftv'][:, -1] > -1e8).tolist(), cs['freq'].tolist(), None, lambda u: [tok(i) for i in cs['lift'][u].tolist()])
        levers['latent'] = (1024, lt['tg'].tolist(), lt['ok'].tolist(), lt['freq'].tolist(), lt, lambda u: [tok(i) for i in lt['tg'][u].tolist()])
        out = {}
        for kind, (n_u, desc, ok, prio, lt_, show) in levers.items():
            r = {'n_units': n_u}
            u = pick(desc, ok, prio, goal)
            if u is not None:
                p, i, kl = at_budget(lp0, edit_unit(key, x, kind, u, lt_), gl)
                lp = edit_unit(key, x, kind, u, lt_)[i]
                r['read'] = {'unit': u, 'desc': show(u), 'p_goal': p, 'strength': GRID[kind][i], 'kl': kl, 'top_after': [tok(j) for j in lp.topk(3).indices.tolist()]}
            best = max(((*at_budget(lp0, edit_unit(key, x, kind, v, lt_), gl), v) for v in range(n_u) if kind != 'latent' or lt['ok'][v]), key=lambda t: t[0])
            r['best'] = {'unit': best[3], 'desc': show(best[3]), 'p_goal': best[0], 'strength': GRID[kind][best[1]], 'kl': best[2]}
            out[kind] = r
        res[key]['levers'] = out
        rd = {}
        if m.k:
            rd['sense'] = [[tok(i) for i in P[l].topk(5).indices.tolist()] for l in range(m.k)]
        if m.C:
            rd['concept'] = [[tok(i) for i in cs['lift'][c][:5].tolist()] for c in m.M(m.mid(x))[0, -1].topk(3).indices.tolist()]
        c = lt['sae'](m.mid(x))[1][0, -1]
        rd['latent'] = [[tok(i) for i in (lt['tg'][j][:5] if lt['ok'][j] else lt['sig'][j]).tolist()] for j in c.topk(3).indices.tolist()]
        read[key] = rd
    names = {'P': 'plain', 'CC': 'concept CE only', 'CM': 'CoCoMix', 'CMo': 'CoCoMix + pen.', 'BP': 'Backpack', 'BPo': 'Backpack + pen.', 'BC': 'Backpack + CoCoMix', 'BCo': 'BP + CM + pen.'}
    lv = {'sense': 'sense of \\emph{dog}', 'concept': 'concept', 'latent': 'SAE latent'}
    rows = []
    for key in ('P', 'CM', 'BP', 'BC'):
        for kind, r in res[key]['levers'].items():
            rd_ = r.get('read')
            rows.append(f"{names[key]} & {lv[kind]} & " + (f"{tex(' '.join(rd_['desc'][:4]))} & {rd_['p_goal']:.2f} & {tex(' '.join(rd_['top_after']))}" if rd_ else r'-- & -- & --') + f" & {r['best']['p_goal']:.2f} \\\\")
    write_tex('ti_common', '\n'.join([r'\begin{tabular}{@{}lllrlr@{}}', r'\toprule', r'model & lever & unit found by reading (top tokens) & goal mass & next word after & best unit \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    rows = []
    for key in ('P', 'CM', 'BP'):
        for kind, units in read[key].items():
            rows.append(f"{names[key]} & {lv[kind]} & " + ' ; '.join(f"[{tex(' '.join(u))}]" for u in units) + r' \\')
    write_tex('ti_common_read', '\n'.join([r'\begin{tabular}{@{}llp{9.2cm}@{}}', r'\toprule', r'model & interface & units that carry the last word (top tokens of each) \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    dump('common', {'prompt': PROMPT, 'goal': list(GOAL), 'budget': BUDGET, 'models': res, 'read': read})
    return {k: {'base': v['base_top'][:3], 'p0': round(v['p_goal_base'], 3), **{kk: (round(vv.get('read', {}).get('p_goal', -1), 3), round(vv['best']['p_goal'], 3)) for kk, vv in v['levers'].items()}} for k, v in res.items()}


# ---------------------------------------------------------------- quantitative tables and figures with standard errors (all 23 runs)

def errbars():
    with open(f'{OUT}/errbars{SUF}.json') as f:
        return json.load(f)


def fmt(v, p=3):
    return f'{v[0]:.{p}f} $\\pm$ {v[1]:.{p}f}'


def quality():
    """Perplexity with its standard error (std of per-token CE / sqrt(tokens)), paired CE difference to the baseline, interface metrics: figure ri_lm and table ti_quality."""
    import report_data as rd
    import report_figs as rf
    E = errbars()
    cells = sorted(E, key=lambda n: -E[n]['ppl'][0])
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 5.4), sharey=True, gridspec_kw={'wspace': .06})
    for i, n in enumerate(cells):
        cell = n[:-3]
        c, mk = rf.COL[rf.cls(cell)], rf.mk(cell)
        ax[0].errorbar(E[n]['ppl'][0], i, xerr=E[n]['ppl'][1], fmt=mk, color=c, ms=5, capsize=2, elinewidth=1)
        ax[1].errorbar(E[n]['d_ce_vs_base'][0], i, xerr=E[n]['d_ce_vs_base'][1], fmt=mk, color=c, ms=5, capsize=2, elinewidth=1)
    ax[0].set_yticks(range(len(cells)), [rf.plain_label(n[:-3]) for n in cells])
    ax[0].axvline(E[N['P']]['ppl'][0], color=MUTED, lw=.8, ls=':')
    ax[1].axvline(0, color=MUTED, lw=.8, ls=':')
    ax[0].set_xlabel('test perplexity $\\pm$ s.e. (lower is better)')
    ax[1].set_xlabel('paired $\\Delta$CE vs plain baseline $\\pm$ s.e. (nats)')
    for a in ax:
        a.grid(axis='y', visible=False)
    rf.class_legend(ax[0], loc='upper right')
    save(fig, 'ri_lm')
    rows = []
    for n in sorted(E, key=lambda n: (rd.plots.parse(n)['k'], rd.plots.parse(n)['cm'], n)):
        r = E[n]
        old = load_eval(n)
        g = lambda k, p=3: fmt(r[k], p) if k in r else '--'
        rows.append(f"{rd.tex_label(n[:-3])} & {old['n_params'] / 1e6:.2f} & {g('ppl')} & {g('d_ce_vs_base', 4)} & {g('concept_hit')} & {g('steer_kl', 2)} & {g('gender_best_reduction', 2)} \\\\")
    write_tex('ti_quality', '\n'.join([r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule', r'model & params (M) & test ppl & $\Delta$CE vs baseline & concept hit@4 & steering KL & gender gap removed \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    main = [f'{c}_s0' for c in ('k0_cm0_-_og0', 'k0_cm0_wm136_og0', 'k0_cm0_wm152_og0', 'k0_cm1_topk_og0', 'k0_cm2_topk_og0', 'k0_cm3_-_og0', 'k4_cm0_-_og0', 'k4_cm0_-_og0.1', 'k4_cm1_topk_og0', 'k4_cm2_topk_og0', 'k4_cm3_-_og0')]
    keep = [r for n, r in zip(sorted(E, key=lambda n: (rd.plots.parse(n)['k'], rd.plots.parse(n)['cm'], n)), rows) if n in main]
    write_tex('ti_quality_main', '\n'.join([r'\begin{tabular}{@{}lrrrrr@{}}', r'\toprule', r'model & params (M) & test ppl & $\Delta$CE vs baseline & concept hit@4 & steering KL \\', r'\midrule', *[k.rsplit(' & ', 1)[0] + r' \\' for k in keep], r'\bottomrule', r'\end{tabular}']) + '\n')
    return {n: {k: E[n][k] for k in ('ppl', 'd_ce_vs_base')} for n in cells}


def contrasts():
    """Matched-pair factor effects on test CE (A - B, mean over the cell pairs that differ in that factor only) from per-token paired differences; s.e. = std / sqrt(tokens)."""
    import report_data as rd
    ce = {k: v.float() for k, v in torch.load(f'{OUT}/ce_tokens.pt').items()}
    cells = sorted({n[:-3] for n in ce})
    pairs = {name: [(c, f(rd.plots.parse(c + '_s0'))) for c in cells if f(rd.plots.parse(c + '_s0')) in cells] for name, f in rd.PAIRS.items()}
    pairs['Backpack $k{=}4$ vs plain $d{=}136$ (width-matched)'] = [('k4_cm0_-_og0', 'k0_cm0_wm136_og0')]
    pairs['Backpack + CoCoMix vs plain $d{=}152$ (width-matched)'] = [('k4_cm2_topk_og0', 'k0_cm0_wm152_og0')]
    pairs['plain $d{=}152$ vs $d{=}128$'] = [('k0_cm0_wm152_og0', 'k0_cm0_-_og0')]
    out, rows = {}, []
    for name, ps in pairs.items():
        d = torch.stack([ce[a + '_s0'] - ce[b + '_s0'] for a, b in ps]).mean(0).double()
        out[name] = {'effect': d.mean().item(), 'sem': sem(d.numpy()), 'n_pairs': len(ps), 'n_tokens': len(d)}
        rows.append(f"{name} & {len(ps)} & {d.mean().item():+.4f} $\\pm$ {sem(d.numpy()):.4f} \\\\")
    write_tex('ti_contrasts', '\n'.join([r'\begin{tabular}{@{}lrr@{}}', r'\toprule', r'contrast (A vs B) & pairs & $\Delta$CE (nats) $\pm$ s.e. \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    dump('contrasts', out)
    return out


def saequal(keys=('P', 'CM', 'BP', 'BC')):
    """Post-hoc SAE metrics, TopK vs OrtSAE, standard errors over tokens / windows / latents / letters."""
    E = errbars()
    mets = [('ev', 'explained variance'), ('ce_recovered', 'CE recovered'), ('max_cos', 'max cosine (lower = more distinct)'), ('absorption', 'first-letter absorption'), ('probe_f1', 'first-letter probe F1')]
    names = {'P': 'plain', 'CM': 'CoCoMix', 'BP': 'Backpack', 'BC': 'BP+CM'}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.0), gridspec_kw={'hspace': .45, 'wspace': .3})
    ax = axs.flatten()
    w, out, rows = .38, {}, []
    for a, (k, ttl) in zip(ax[:5], mets):
        for gi, (g, c, lb) in enumerate(((0., BLUE, 'TopK SAE'), (.25, ORANGE, 'OrtSAE'))):
            v = np.array([E[N[key]][f'sae_g{g}'][k] for key in keys])
            a.bar(np.arange(len(keys)) + (gi - .5) * w, v[:, 0], w, yerr=v[:, 1], color=c, label=lb, error_kw={'lw': .8, 'capsize': 1.5})
        a.set_xticks(range(len(keys)), [names[k_] for k_ in keys], fontsize=6.5, rotation=25)
        a.set_title(ttl, fontsize=7.5)
    ax[5].axis('off')
    h, l = ax[0].get_legend_handles_labels()
    ax[5].legend(h, l, fontsize=7, loc='center')
    for key in keys:
        rows.append(f"{ {'P': 'plain', 'CM': 'CoCoMix', 'BP': 'Backpack', 'BC': 'Backpack + CoCoMix'}[key] } & " + ' & '.join(f"{fmt(E[N[key]]['sae_g0.0'][k], 3)} / {fmt(E[N[key]]['sae_g0.25'][k], 3)}" for k, _ in mets) + r' \\')
        out[key] = {k: [E[N[key]]['sae_g0.0'][k], E[N[key]]['sae_g0.25'][k]] for k, _ in mets}
    write_tex('ti_sae', '\n'.join([r'\begin{tabular}{@{}lccccc@{}}', r'\toprule', r'LM & EV & CE recovered & max-cos & absorption & probe F1 \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    save(fig, 'ri_sae')
    dump('saequal', out)
    return out


def saefull():
    """Post-hoc SAE metrics of all 23 models, TopK / OrtSAE, with standard errors."""
    import report_data as rd
    E = errbars()
    rows = []
    for n in sorted(E, key=lambda n: (rd.plots.parse(n)['k'], rd.plots.parse(n)['cm'], n)):
        cells = ' & '.join(f"{E[n]['sae_g0.0'][k][0]:.3f} / {E[n]['sae_g0.25'][k][0]:.3f}" for k in ('ev', 'ce_recovered', 'max_cos', 'absorption', 'probe_f1'))
        err = ' & '.join(f"{E[n]['sae_g0.0'][k][1]:.4f} / {E[n]['sae_g0.25'][k][1]:.4f}" for k in ('ev', 'ce_recovered', 'max_cos', 'absorption', 'probe_f1'))
        rows.append(f"{rd.tex_label(n[:-3])} & {cells} \\\\\n & {err} \\\\")
    write_tex('ti_sae_full', '\n'.join([r'\begin{tabular}{@{}lccccc@{}}', r'\toprule', r'model & EV & CE recovered & max-cos & absorption & probe F1 \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    return {}


def export():
    """Copy the interpretability figures into the report folder."""
    os.makedirs(f'{HERE}/report/figures', exist_ok=True)
    for f in glob.glob(f'{FIG}/ri_*.pdf'):
        shutil.copy(f, f'{HERE}/report/figures/')
    return sorted(os.path.basename(f) for f in glob.glob(f'{FIG}/ri_*.pdf'))


SEED_ANALYSES = ['units', 'geometry', 'steer', 'senses', 'gender', 'concepts', 'wnorm', 'absorb', 'saequal', 'common', 'reliance', 'faith']
ANALYSES = ['common', 'reliance', 'faith', 'units', 'geometry', 'steer', 'senses', 'decomp', 'gender', 'concepts', 'cgen', 'wnorm', 'sdecomp', 'absorb', 'quality', 'contrasts', 'saequal', 'saefull', 'export']

if __name__ == '__main__':
    todo = sys.argv[1:] or (SEED_ANALYSES if SEED else ANALYSES)
    for name in todo:
        print(name, flush=True)
        r = globals()[name]()
        print(json.dumps(r, default=float)[:600], flush=True)
