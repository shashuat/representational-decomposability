"""Correctness tests: `python test_all.py` (or pytest). Tiny CPU configs on synthetic data; `E2E=1` adds the slow grid.py end-to-end run.

Each test targets a way the method could be silently wrong: future leakage, broken decomposition, wrong interleave layout, penalty maths,
SAE sparsity / norm, checkpoint round-trip, metric definitions, plotting on the real artifact schema."""
import contextlib
import glob
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import traceback
from unittest import SkipTest

TMP = tempfile.mkdtemp()
os.environ['DATA'] = f'{TMP}/data'  # must be set before importing the project modules
os.makedirs(f'{TMP}/data')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

import data  # noqa: E402
import evaluate as ev  # noqa: E402
import grid  # noqa: E402
import label  # noqa: E402
import plots  # noqa: E402
import sae as S  # noqa: E402
import train  # noqa: E402
from model import LM, ort  # noqa: E402

torch.manual_seed(0)
random.seed(0)
T = 16


def fake_data():
    """Synthetic 'stories' -> BPE -> bins for the four splits. Returns the vocabulary size."""
    words = 'the a doctor nurse teacher farmer baker said that he she was tired and happy dog ball boy girl ran to park home . ,'.split()
    stories = {k: [' '.join(random.choice(words) for _ in range(60)) for _ in range(n)] for k, n in {'train': 300, 'sae': 120, 'dev': 30, 'test': 80}.items()}
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer, tk.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    tk.train_from_iterator(stories['train'], trainers.BpeTrainer(vocab_size=600, special_tokens=['<eos>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk.save(f'{TMP}/data/tok.json')
    for k, texts in stories.items():
        np.concatenate([np.array(e.ids + [0], dtype=np.uint16) for e in tk.encode_batch(texts)]).tofile(f'{TMP}/data/{k}.bin')
    return tk.get_vocab_size()


V = fake_data()
DIMS = {'V': V, 'd': 32, 'T': T, 'L': 2, 'H': 2}
VARIANTS = ({}, {'k': 4}, {'C': 8}, {'C': 8, 'mix': 1}, {'k': 4, 'C': 8, 'mix': 1})


def mk(big=True, **kw):
    """big=True: weights ~ N(0, .3) so logits are O(1). At the std-0.02 init Backpack logits are ~1e-6 and equality tests would pass vacuously."""
    m = LM(**{**DIMS, **kw}).eval()
    for p in m.parameters() if big else ():
        if p.dim() > 1:
            p.data.normal_(0, .3)
    return m


def rnd(n=2):
    return torch.randint(1, V, (n, T))


# ---------------------------------------------------------------- model

def test_causality():  # changing tokens >= 9 must not move logits at positions < 9, in every variant
    for kw in VARIANTS:
        m = mk(**kw)
        x = rnd()
        x2 = x.clone()
        x2[:, 9:] = (x2[:, 9:] + 1) % V
        with torch.no_grad():
            a, b = m(x)[0], m(x2)[0]
        assert torch.allclose(a[:, :9], b[:, :9], atol=1e-5), f'future leak {kw}'
        assert (a[:, 9:] - b[:, 9:]).abs().max() > 1e-4, f'inputs ignored {kw}'


def test_backpack_matches_reference():  # SDPA implementation == explicit per-sense softmax-weighted sum of sense vectors, then tied readout
    m = mk(k=4)
    x = rnd()
    k, dh = 4, 8
    with torch.no_grad():
        h = m.ln_f(m.blocks[1](m.mid(x)))
        senses = m.sense(m.E(x)).view(2, T, k, -1)
        q, kk = m.Wq(h).view(2, T, k, -1), m.Wk(h).view(2, T, k, -1)
        out = torch.zeros(2, T, DIMS['d'])
        for ell in range(k):
            for i in range(T):
                w = torch.softmax((q[:, i, ell, None] * kk[:, :i + 1, ell]).sum(-1) / math.sqrt(dh), -1)
                out[:, i] += (w[..., None] * senses[:, :i + 1, ell]).sum(1)
        ref = out @ m.E.weight.T
        assert ref.abs().max() > .1 and torch.allclose(m(x)[0], ref, atol=1e-4)


def test_sense_layernorm_option():  # sln=1: LayerNorm in front of the sense net; reference equality and param count (+2d) still hold, sln=0 is unchanged
    base, ln = mk(k=4), mk(k=4, sln=1)
    d = DIMS['d']
    assert sum(p.numel() for p in ln.parameters()) - sum(p.numel() for p in base.parameters()) == 2 * d
    assert [n for n, _ in base.sense.named_parameters()][0] == '0.weight' and isinstance(ln.sense[0], torch.nn.LayerNorm)
    x = rnd()
    with torch.no_grad():
        S = ln.sense(ln.E(x))
        assert torch.allclose(S, ln.sense[1:](torch.nn.functional.layer_norm(ln.E(x), (d,), ln.sense[0].weight, ln.sense[0].bias)), atol=1e-5)
        assert ln(x)[0].shape == base(x)[0].shape


def test_backpack_linear_in_sense_scales():  # exact decomposition: logits are linear in per-(token, sense) scales
    m = mk(k=4)
    x = rnd()
    s1, s2 = torch.rand(2, T, 4), torch.rand(2, T, 4)
    with torch.no_grad():
        assert torch.allclose(m(x, sc=s1)[0] + m(x, sc=s2)[0], m(x, sc=s1 + s2)[0], atol=1e-5)
        assert torch.allclose(m(x, sc=torch.ones(2, T, 4))[0], m(x)[0], atol=1e-6)
        assert m(x, sc=torch.zeros(2, T, 4))[0].abs().max() < 1e-6


def test_sense_decomposition_sums_to_logits():  # what fig_decomp plots: keep one sense at a time, contributions add up to the full logits
    m = mk(k=4)
    x = rnd(1)
    with torch.no_grad():
        parts = torch.stack([m(x, sc=torch.eye(4)[ell].expand(1, T, 4))[0] for ell in range(4)])
        assert parts.abs().max() > .1 and torch.allclose(parts.sum(0), m(x)[0], atol=1e-4)


def test_concept_head_and_loss():
    m = mk(big=False, C=8, mix=1, Kc=2)
    x, y = rnd(), rnd()
    lab = torch.randint(0, 8, (2, T, 3))
    with torch.no_grad():
        logits, o = m(x, y, lab=lab)
        assert logits.shape == (2, T, V)  # the LM reads T concept slots, not 2T
        assert torch.allclose(o['lc'], -F.log_softmax(o['z'], -1).gather(-1, lab).mean())
        assert abs(o['lc'].item() - math.log(8)) < .1  # untrained head ~ uniform
        z = torch.zeros(2, T, 8)
        z[..., 3] = 5
        assert (m(x, zadd=z)[0] - logits).abs().max() > 1e-4  # with mixing, steering is causal
        m0 = mk(big=False, C=8, mix=0)
        assert torch.allclose(m0(x, zadd=z)[0], m0(x)[0])  # concept-loss-only: the head is not on the forward path


def test_interleave_layout():  # the stream entering block 1 is (h1, c1, h2, c2, ...) with c_t = W(TopK(z_t))
    m = mk(C=8, mix=1, Kc=2)
    x = rnd()
    captured = []
    m.blocks[1].register_forward_hook(lambda mod, inp, out: captured.append(inp[0]))
    with torch.no_grad():
        _, o = m(x)
        h = m.mid(x)
        v, i = o['z'].topk(2, -1)
        c = m.W(torch.zeros_like(o['z']).scatter(-1, i, v))
    assert captured[0].shape == (2, 2 * T, DIMS['d'])
    assert torch.allclose(captured[0][:, 0::2], h, atol=1e-6) and torch.allclose(captured[0][:, 1::2], c, atol=1e-6)


def test_ort_penalty():
    v = torch.tensor([[1., 0], [.6, .8], [0, 1.]])  # cosines: (0,1)=.6, (1,2)=.8, (0,2)=0 -> row maxima .6, .8, .8
    assert ort(torch.eye(8)) < 1e-9 and abs(ort(torch.ones(8, 4)) - 1) < 1e-6
    assert abs(ort(v) - (.36 + .64 + .64) / 3) < 1e-6
    assert abs(ort(7 * v) - ort(v)) < 1e-6 and abs(ort(torch.stack([v, v])) - ort(v)) < 1e-6
    W = torch.randn(10, 6, requires_grad=True)
    ort(W, K=2).backward()
    assert W.grad.isfinite().all() and W.grad.abs().sum() > 0


def test_penalty_is_optimisable():  # model.pen() can be driven down through the sense net and the compression layer
    m = mk(big=False, k=4, C=8, mix=1).train()
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    p0 = float(m.pen())
    for _ in range(150):
        opt.zero_grad()
        m.pen().backward()
        opt.step()
    assert float(m.pen()) < .5 * p0, (p0, float(m.pen()))


def test_param_budget():  # real dimensions: baseline ~1M parameters, every mechanism adds parameters
    n = lambda **kw: sum(p.numel() for p in LM(V=4096, **kw).parameters())
    base = n()
    print(f'  params base {base / 1e6:.2f}M | k=4 {n(k=4) / 1e6:.2f}M | C=512 {n(C=512) / 1e6:.2f}M | both {n(k=4, C=512, mix=1) / 1e6:.2f}M')
    assert .8e6 < base < 1.1e6 and n(k=4) > base and n(C=512) > base
    assert n(k=4) < n(d=152) and n(d=136) < n(d=152)  # width-matched baselines bracket the mechanisms' sizes


def test_overfit():  # every variant can memorise one fixed batch (catches dead gradients / wiring bugs)
    for kw in VARIANTS:
        torch.manual_seed(1)
        m = mk(big=False, **kw).train()
        x, y = rnd(4), rnd(4)
        lab = torch.randint(0, 8, (4, T, 2))
        opt = torch.optim.Adam(m.parameters(), 5e-3)
        first = None
        for _ in range(200):
            _, o = m(x, y, lab=lab if m.C else None)
            first = first or o['ce'].item()
            opt.zero_grad()
            (o['ce'] + o.get('lc', 0)).backward()
            opt.step()
        assert o['ce'].item() < .5 * first, (kw, first, o['ce'].item())


# ---------------------------------------------------------------- SAE / OrtSAE

def planted(n=4096, d=32, m=64, K=2, cor=0.):
    """x = sparse non-negative combination of m unit atoms; cor > 0 makes the atoms share a common direction."""
    atoms = F.normalize(torch.randn(m, d) + cor * torch.randn(1, d) * 3, dim=-1)
    codes = torch.zeros(n, m).scatter_(1, torch.rand(n, m).topk(K, -1).indices, torch.rand(n, K) + .5)
    return codes @ atoms


def mean_max_cos(sae):
    W = F.normalize(sae.Wd, dim=-1)
    return (W @ W.T).fill_diagonal_(-1).max(1).values.mean().item()


def test_sae_sparsity_norm_fit():
    X = planted()
    sae = S.train_sae(X, 64, 2, steps=800, bs=512, lr=2e-3)
    x_hat, c, _ = sae(X)
    assert ((c > 0).sum(-1) <= 2).all() and torch.allclose(sae.Wd.norm(dim=-1), torch.ones(64), atol=1e-4)
    ev_ = 1 - ((x_hat - X) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()
    assert ev_ > .6, ev_


def test_ortsae_lowers_decoder_cosine():  # same seed and data, only gamma differs; dictionary with correlated atoms
    X = planted(cor=1.)
    a, b = S.train_sae(X, 64, 2, 0., 600, 512, 2e-3), S.train_sae(X, 64, 2, 1., 600, 512, 2e-3)
    assert mean_max_cos(b) < mean_max_cos(a) - .02, (mean_max_cos(a), mean_max_cos(b))


def test_attribution_chain_rule():  # dL/dcode == (dL/dh) @ W_dec^T when h = code @ W_dec (the transposition label.attrib relies on)
    lm = mk(big=False)
    x, y = rnd(4), rnd(4)
    Wd = F.normalize(torch.randn(16, DIMS['d']), dim=-1)
    c = torch.rand(4, T, 16).requires_grad_()
    gc, = torch.autograd.grad(lm(x, y, h=c @ Wd)[1]['ce'], c)
    h = (c @ Wd).detach().requires_grad_()
    gh, = torch.autograd.grad(lm(x, y, h=h)[1]['ce'], h)
    assert torch.allclose(gc, gh @ Wd.T, atol=1e-6)
    sae = S.train_sae(lm.mid(rnd(32)).detach().flatten(0, 1), 16, 4, steps=5, bs=64)
    labels = label.attrib(lm, sae, x, y, 2)
    assert labels.shape == (4, T, 2) and labels.dtype == torch.int16 and labels.min() >= 0 and labels.max() < 16
    assert not torch.equal(labels, label.attrib(lm, sae, x, y, 2, sign=-1))


# ---------------------------------------------------------------- data / pipeline / metrics

def test_windows_and_splits():
    for split in ('train', 'sae', 'dev', 'test'):
        X, Y = data.windows(split, T)
        assert X.shape == Y.shape and X.shape[1] == T and torch.equal(X.flatten()[1:], Y.flatten()[:-1]), split
    c = data.carve(list(range(100)), list(range(1000, 1010)), (50, 20, 10))
    assert [len(c[k]) for k in ('train', 'sae', 'dev', 'test')] == [50, 20, 10, 10]
    sets = [set(v) for v in c.values()]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(4) for j in range(i + 1, 4))


def test_grid():
    cfgs = list(grid.cfgs())
    assert len(cfgs) == 63 + 6 and len({c['name'] for c in cfgs}) == 69
    assert any(c['k'] == 0 and c['C'] == 0 and c['og'] == 0 and 'd' not in c for c in cfgs)  # plain baseline present
    for c in cfgs:
        if '_wm' in c['name']:
            continue
        cm = c['name'].split('_')[1]
        assert (c['C'] > 0) == (cm != 'cm0') and bool(c['lab']) == (cm in ('cm1', 'cm2')) and not (c['og'] and not (c['k'] or c['mix']))
    for c in cfgs[::4]:
        m = LM(**{**DIMS, **c}).eval()
        labels = torch.randint(0, c['C'], (2, T, 4)) if c['C'] else None
        with torch.no_grad():
            assert m(rnd(), rnd(), lab=labels)[0].shape == (2, T, V)


def test_train_then_eval_roundtrip():  # step 1 writes a checkpoint; step 2 re-loads it and recomputes everything
    ld = f'{TMP}/lab'
    os.makedirs(ld, exist_ok=True)
    for split in ('train', 'dev', 'test'):
        torch.save(torch.randint(0, 8, (data.windows(split, T)[0].shape[0], T, 2)).short(), f'{ld}/{split}.pt')
    cfg = {**train.DEF, **DIMS, 'name': 'rt', 'out': f'{TMP}/runs', 'steps': 4, 'bs': 8, 'warm': 1, 'k': 4, 'C': 8, 'mix': 1, 'lab': ld, 'og': .1}
    train.main(cfg)
    ckpt = f'{TMP}/runs/rt/ckpt.pt'
    assert os.path.exists(ckpt) and torch.load(ckpt)['cfg'] == cfg
    with open(f'{TMP}/runs/rt/log.jsonl') as f:
        recs = [json.loads(line) for line in f]
    assert len(recs) >= 2 and all(k in recs[-1] for k in ('ce', 'dev_ce', 'elapsed_s', 'tok_s', 'lr')) and recs[-1]['elapsed_s'] >= recs[0]['elapsed_s'] > 0
    with open(f'{TMP}/runs/rt/timing.json') as f:
        tm = json.load(f)
    assert tm['steps'] == 4 and tm['train_s'] > 0 and tm['total_s'] >= tm['train_s'] and tm['tok_s'] > 0 and tm['first_step_s'] <= tm['train_s']
    ev.main(ckpt, gs=(0., .25), m=16, K=2, steps=5, bs=64)
    with open(f'{TMP}/runs/rt/eval_test.json') as f:
        r = json.load(f)
    m = LM(**torch.load(ckpt)['cfg']).eval()
    m.load_state_dict(torch.load(ckpt)['sd'])
    X, Y = data.windows('test', T)
    with torch.no_grad():
        ce = np.mean([F.cross_entropy(m(X[i:i + 8])[0].flatten(0, 1), Y[i:i + 8].flatten()).item() for i in range(0, len(X) - len(X) % 8, 8)])
    assert abs(r['ce'] - ce) < 1e-3 and abs(r['ppl'] - math.exp(r['ce'])) < 1e-6 and r['n_params'] == sum(p.numel() for p in m.parameters())
    assert 0 <= r['concept_hit'] <= 1 and np.isfinite(r['steer_kl']) and 'sense_cos' in r and 'sense_top8' in r
    assert r['gender_prof'] and all(len(p['zero']) == 4 for p in r['gender_prof'].values()) and len(r['gender_kl_zero_sense']) == 4
    for g in ('sae_g0.0', 'sae_g0.25'):
        s = r[g]
        assert all(np.isfinite(v) for v in s.values() if not isinstance(v, dict))
        assert 0 <= s['absorption'] <= 1 and 0 <= s['probe_f1'] <= 1 and 0 <= s['dead'] <= 1 and s['l0'] <= 2 + 1e-6
        assert s['per_letter'] and all(0 <= f1 <= 1 and 0 <= ab <= 1 for f1, ab in s['per_letter'].values())
    tt = r['time']
    assert tt['total_s'] >= tt['lm_metrics_s'] + tt['acts_s'] and all(v >= 0 for v in tt.values()) and r['sae_g0.25']['time']['train_sae_s'] > 0
    assert torch.load(f'{TMP}/runs/rt/sae_decoders.pt')[0.0].shape == (16, DIMS['d'])


def test_early_stopping_and_best_checkpoint():  # a dev CE that stops improving ends the run early and the saved weights are the best-dev step
    cfg = {**train.DEF, **DIMS, 'name': 'es', 'out': f'{TMP}/runs', 'steps': 2000, 'es': 1, 'bs': 8, 'warm': 1, 'lr': 1e-9, 'es_start': 0,
           'patience': 200, 'min_delta': 1., 'comp': 0}
    train.main(cfg)  # lr ~ 0 and min_delta 1.0: no gain of 1 nat after the first log line -> stops at step 200
    with open(f'{TMP}/runs/es/timing.json') as f:
        tm = json.load(f)
    assert tm['early_stopped'] and tm['steps_run'] == 201 and tm['best_step'] <= tm['steps_run'] and tm['steps'] == 2000
    with open(f'{TMP}/runs/es/log.jsonl') as f:
        recs = [json.loads(line) for line in f]
    assert recs[-1]['step'] == 200 and abs(recs[-1]['best_dev_ce'] - min(r['dev_ce'] for r in recs)) < 1e-9
    full = {**cfg, 'name': 'es_off', 'steps': 150, 'es': 0}
    train.main(full)
    with open(f'{TMP}/runs/es_off/timing.json') as f:
        assert not json.load(f)['early_stopped']


def test_lm_metrics_and_probe():
    m = mk(big=False, C=8).to(ev.dev)
    X, Y = (t[:16].to(ev.dev) for t in data.windows('test', T))
    r = ev.lm_metrics(m, X, Y, None)
    with torch.no_grad():
        ref = F.cross_entropy(m(X)[0].flatten(0, 1), Y.flatten()).item()
        labels = m(X)[1]['z'].topk(3, -1).indices  # labels equal to the head's own top-3 -> precision@Ka must be exactly 1
    assert abs(r['ce'] - ref) < 1e-4 and abs(r['ppl'] - math.exp(ref)) < 1e-2
    assert abs(ev.lm_metrics(m, X, Y, labels)['concept_hit'] - 1) < 1e-6
    lab = torch.arange(150) % 3
    A = F.one_hot(lab, 3).float() @ torch.randn(3, 8) + .05 * torch.randn(150, 8)
    C = F.one_hot(lab, 8).float()  # latent c <-> letter c
    p = ev.probe(A, C, lab)
    assert p['probe_f1'] > .99 and p['absorption'] < 1e-6 and p['probe_dense_acc'] > .99 and set(p['per_letter']) == {'a', 'b', 'c'}
    C2 = C.clone()
    C2[(lab == 0) & (torch.arange(150) % 2 == 0), 0] = 0  # latent 0 silent on half of the letter-0 tokens -> absorption (.5 + 0 + 0) / 3
    assert abs(ev.probe(A, C2, lab)['absorption'] - .5 / 3) < .02


def check_plot_outputs(figdir, resdir):
    got = set(os.listdir(figdir))
    for f in ('lm', 'effects', 'curves', 'concept', 'gender', 'sae', 'pareto', 'letters', 'heatmap', 'ckpt', 'decomp', 'timing'):
        assert f'fig_{f}.pdf' in got, (f, sorted(got))
    got = set(os.listdir(resdir))
    for f in ('aggregates.csv', 'results.md', 'effects.md', 'senses.md', 'concepts.md', 'timing.md', 'summary.txt'):
        assert f in got, (f, sorted(got))


def test_plots_run_on_fake_runs():
    buf = io.StringIO()
    plots.fake_runs(f'{TMP}/fake', DIMS)
    with contextlib.redirect_stdout(buf):
        plots.main(f'{TMP}/fake', f'{TMP}/figs', f'{TMP}/res')
    assert 'WARNING' not in buf.getvalue(), buf.getvalue()
    check_plot_outputs(f'{TMP}/figs', f'{TMP}/res')
    with open(f'{TMP}/res/effects.md') as f:
        assert 'Backpack k=4 vs k=0' in f.read()


def test_grid_end_to_end():  # E2E=1: the real orchestration (grid.py subprocesses) on synthetic tokens, then plots + text aggregates
    if not os.environ.get('E2E'):
        raise SkipTest('slow; set E2E=1')
    env = {**os.environ, 'OUT': f'{TMP}/e2e', 'SEEDS': '0'}
    src = os.path.dirname(os.path.abspath(__file__))
    go = lambda *a: subprocess.run([sys.executable, 'grid.py', *a], cwd=src, env=env, check=True, capture_output=True, text=True).stdout
    tr = [f'train.{k}={v}' for k, v in {'V': V, 'd': 32, 'T': T, 'steps': 6, 'bs': 8, 'warm': 2}.items()]
    n = int(go('list').split()[0])
    assert n == 23
    go('train', *tr, 'label.m=64', 'label.K=4', 'label.steps=10')
    assert all(os.path.exists(f'{TMP}/e2e/teacher/{t}/{s}.pt') for t in ('topk', 'ort') for s in ('train', 'dev', 'test'))
    go('eval', *tr, 'eval.sae={"m": 64, "K": 4, "steps": 10, "bs": 256}')
    assert len(glob.glob(f'{TMP}/e2e/*/eval_test.json')) == n
    agg = go('agg')
    assert 'k4_cm2_ort_og0.1' in agg and 'wm136' in agg
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        plots.main(f'{TMP}/e2e', f'{TMP}/e2e_figs', f'{TMP}/e2e_res')
    assert 'WARNING' not in buf.getvalue(), buf.getvalue()
    check_plot_outputs(f'{TMP}/e2e_figs', f'{TMP}/e2e_res')
    with open(f'{TMP}/e2e_res/concepts.md') as f:
        assert 'purity' in f.read()


if __name__ == '__main__':
    bad = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith('test_') and callable(f)):
        try:
            fn()
            print('PASS', name)
        except SkipTest as e:
            print('SKIP', name, e)
        except Exception as e:
            bad += 1
            print('FAIL', name, repr(e))
            traceback.print_exc(limit=3)
    sys.exit(bad)
