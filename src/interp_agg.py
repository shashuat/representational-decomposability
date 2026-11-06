"""Five-seed figures and tables of the eight main models (reads the per-seed json written by interp_viz.py with SEED=k; no GPU).
Error bars = std / sqrt(number of seeds) over the seeds present; seed dots are drawn where they help. Also the training curves and the 23-model figures stay in interp_viz.py.
  python interp_agg.py [analysis ...]   ->  figures_30k/interp/ri_*.pdf, report/generated/ti_*.tex  (then python interp_viz.py export)"""
import glob
import json
import math
import os
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np

import interp_viz as V
from interp_viz import AQUA, BLUE, INK, MUTED, ORANGE, VIOLET, save, tex, write_tex

assert V.SEED == 0, 'run the aggregator without SEED'
OUT, RUNS = V.OUT, V.RUNS
NAMES = {'P': 'plain', 'CC': 'concept CE only', 'CM': 'CoCoMix', 'CMo': 'CoCoMix + pen.', 'BP': 'Backpack', 'BPo': 'Backpack + pen.', 'BC': 'Backpack + CoCoMix', 'BCo': 'BP + CM + pen.'}


def rd(p):
    with open(p) as f:
        return json.load(f)


def seeds(name):
    """Per-seed json objects of one analysis (seed 0 first)."""
    out = []
    for s in range(5):
        p = f'{OUT}/{name}.json' if s == 0 else f'{OUT}/seed{s}/{name}.json'
        if os.path.exists(p):
            out.append(rd(p))
    return out


def ms(v):
    """mean and std / sqrt(n) over the seeds (zero error for a single seed)."""
    v = np.asarray(v, float)
    m = v.mean(0)
    return m, (v.std(0, ddof=1) / math.sqrt(len(v)) if len(v) > 1 else np.zeros_like(m))


def errs():
    """errbars.json of every seed present, keyed by seed."""
    d = {0: rd(f'{OUT}/errbars.json')}
    for p in glob.glob(f'{OUT}/errbars_s*.json'):
        d[int(p.rsplit('_s', 1)[1][:-5])] = rd(p)
    return dict(sorted(d.items()))


def pm(v, p=3):
    m, e = ms(v)
    return f'{float(m):.{p}f} $\\pm$ {float(e):.{p}f}' if len(v) > 1 else f'{float(m):.{p}f}'


def dots(ax, x, v, c, w=.12):
    ax.scatter(x + np.linspace(-w, w, len(v)), v, s=7, color=c, alpha=.55, lw=0, zorder=3)


# ---------------------------------------------------------------- common example

def common():
    """ri_common: (a) the same prompt through all eight models, (b) the goal mass each lever reaches within the collateral budget. ti_common: table."""
    S = seeds('common')
    n = len(S)
    models = ['P', 'CC', 'CM', 'CMo', 'BP', 'BPo', 'BC', 'BCo']
    c0 = S[0]['models']
    toks = ['was', 'saw', 'had', 'looked', 'wanted', 'said', 'barked']
    M = np.zeros((len(models), len(toks)))
    for i, k in enumerate(models):
        d = dict(c0[k]['base_top'])
        M[i] = [d.get(t, np.nan) for t in toks]
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[.8, 1.25], wspace=.62)
    a = fig.add_subplot(gs[0])
    a.imshow(np.nan_to_num(M), cmap='Blues', vmin=0, vmax=.7, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                a.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white' if M[i, j] > .4 else INK)
    a.set_xticks(range(len(toks)), toks, rotation=45, ha='right', fontsize=6.5)
    a.set_yticks(range(len(models)), [NAMES[k] for k in models], fontsize=6.5)
    a.grid(False)
    a.set_title('(a) next word after the prompt (top-5 of each)', fontsize=7.5)
    b = fig.add_subplot(gs[1])
    rows = [(k, kind) for kind in ('sense', 'concept', 'latent') for k in models if kind in c0[k]['levers']]
    col = {'sense': AQUA, 'concept': ORANGE, 'latent': VIOLET}
    for r, (k, kind) in enumerate(rows):
        base = ms([s['models'][k]['p_goal_base'] for s in S])[0]
        b.plot([base, base], [r - .35, r + .35], color=MUTED, lw=1)
        best = [s['models'][k]['levers'][kind]['best']['p_goal'] for s in S]
        read = [s['models'][k]['levers'][kind]['read']['p_goal'] if 'read' in s['models'][k]['levers'][kind] else s['models'][k]['p_goal_base'] for s in S]
        for v, fill, off in ((best, 'none', .12), (read, col[kind], -.12)):
            m, e = ms(v)
            b.errorbar(m, r + off, xerr=e, fmt='o', ms=4.2, mfc=fill, mec=col[kind], color=col[kind], capsize=1.5, elinewidth=.8)
    b.set_yticks(range(len(rows)), [f'{NAMES[k]} / {kind}' for k, kind in rows], fontsize=6)
    b.invert_yaxis()
    b.set_xlabel(f'probability of the goal words ("barked wagged ran ...")\nwithin {S[0]["budget"]:.1f} nats of collateral KL', fontsize=7)
    b.set_title('(b) pushing the prompt toward the goal', fontsize=7.5)
    b.plot([], [], 'o', color=INK, ms=4, label='unit found by reading')
    b.plot([], [], 'o', mfc='none', color=INK, ms=4, label='best unit of the interface')
    b.plot([], [], '|', color=MUTED, ms=7, label='before editing')
    b.legend(fontsize=6, loc='lower right')
    b.grid(axis='y', visible=False)
    save(fig, 'ri_common')
    names = NAMES
    rows_t = []
    lv = {'sense': 'sense of \\emph{dog}', 'concept': 'concept', 'latent': 'SAE latent'}
    for k in ('P', 'CM', 'BP', 'BC'):
        for kind, r in c0[k]['levers'].items():
            rd_ = r.get('read')
            vr = [s['models'][k]['levers'][kind]['read']['p_goal'] if 'read' in s['models'][k]['levers'][kind] else s['models'][k]['p_goal_base'] for s in S]
            vb = [s['models'][k]['levers'][kind]['best']['p_goal'] for s in S]
            rows_t.append(f"{names[k]} & {lv[kind]} & " + (tex(' '.join(rd_['desc'][:4])) if rd_ else '--') + f" & {pm(vr, 2)} & {pm(vb, 2)} & {tex(' '.join(rd_['top_after'])) if rd_ else '--'} \\\\")
    write_tex('ti_common', '\n'.join([r'\begin{tabular}{@{}lllrrl@{}}', r'\toprule', r'model & lever & unit found by reading (seed 0) & goal mass & best unit & next word after \\', r'\midrule', *rows_t, r'\bottomrule', r'\end{tabular}']) + '\n')
    return {'n_seeds': n, 'rows': len(rows)}


# ---------------------------------------------------------------- intervention response

def steer():
    S = seeds('steer')
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
    styles = {'sense': (ax[0], {'BP': (AQUA, 'Backpack'), 'BPo': (BLUE, 'Backpack + penalty'), 'BC': (ORANGE, 'Backpack + CoCoMix')}, 'sense scale'),
              'concept': (ax[1], {'CM': (ORANGE, 'CoCoMix'), 'BC': (AQUA, 'Backpack + CoCoMix'), 'CC': (VIOLET, 'concept CE only')}, 'concept logit shift'),
              'latent': (ax[2], {'P_TopK': (BLUE, 'plain, TopK SAE'), 'P_OrtSAE': (VIOLET, 'plain, OrtSAE'), 'CM_TopK': (ORANGE, 'CoCoMix, TopK SAE'), 'CM_OrtSAE': (AQUA, 'CoCoMix, OrtSAE')}, 'latent strength (x mean active code)')}
    out = {}
    for kind, (a, cm, xl) in styles.items():
        for key, (c, lb) in cm.items():
            d = [s[f'{kind}|{key}'] for s in S]
            x = d[0]['x']
            on, on_e = ms([q['on'] for q in d])
            off, off_e = ms([q['off'] for q in d])
            a.errorbar(x, on, on_e, color=c, marker='o', ms=3, lw=1.2, capsize=2, label=lb)
            a.fill_between(x, off - off_e, off + off_e, color=c, alpha=.12, lw=0)
            a.plot(x, off, color=c, ls=':', lw=1)
            out[f'{kind}|{key}'] = {'x': x, 'on': on.tolist(), 'on_sem': on_e.tolist(), 'off': off.tolist()}
        a.set_xlabel(xl)
    ax[0].set_ylabel('change in log target mass')
    ax[1].plot([], [], color=MUTED, ls=':', lw=1, label='dotted: other units\' targets')
    for a, t in zip(ax, ('Backpack senses', 'CoCoMix concepts', 'post-hoc SAE latents')):
        a.set_title(t, fontsize=8)
        a.legend(fontsize=6.2, loc='upper left')
    save(fig, 'ri_steer')
    return {'n_seeds': len(S)}


# ---------------------------------------------------------------- geometry

def geometry():
    S = seeds('geometry')
    panels = [('SAE decoder rows', ('TopK SAE', 'OrtSAE')), ('senses of one word', ('no penalty', 'penalty')), ('concept directions', ('no penalty', 'penalty'))]
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.3), sharey=True)
    for a, (t, nms) in zip(ax, panels):
        for i, (nm, c) in enumerate(zip(nms, (BLUE, ORANGE))):
            v = [s[f'{t}|{nm}'][0] for s in S]
            m, e = ms(v)
            a.bar(i, m, .6, yerr=e, color=c, error_kw={'lw': .8, 'capsize': 2})
            dots(a, i, v, INK)
        r = ms([s[f'{t}|random'][0] for s in S])[0]
        a.axhline(r, color=MUTED, ls=':', lw=1)
        a.text(1.45, r, ' random', fontsize=6, va='bottom', ha='right', color=MUTED)
        a.set_xticks([0, 1], nms, fontsize=7)
        a.set_title(t, fontsize=7.5)
        a.axhline(0, color=MUTED, lw=.6)
        a.grid(axis='x', visible=False)
    ax[0].set_ylabel('cosine to nearest other unit')
    save(fig, 'ri_geometry')
    return {'n_seeds': len(S)}


# ---------------------------------------------------------------- readability, roles, gender, SAE, absorption

def units():
    S = seeds('units')
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.4), gridspec_kw={'width_ratios': [1.1, 1]})
    sets = [([('P_TopK', BLUE), ('P_OrtSAE', VIOLET), ('CM_TopK', ORANGE), ('CM_OrtSAE', ORANGE), ('BP_TopK', AQUA), ('BP_OrtSAE', AQUA)], 'post-hoc SAE latents',
             ['plain\nTopK', 'plain\nOrt', 'CoCoMix\nTopK', 'CoCoMix\nOrt', 'Backpack\nTopK', 'Backpack\nOrt']),
            ([('CC_concept', VIOLET), ('CM_concept', ORANGE), ('BC_concept', AQUA), ('teacher', MUTED)], 'concept units', ['CE only', 'CoCoMix', 'BP + CM', 'teacher\nSAE'])]
    for a, (items, t, labs) in zip(ax, sets):
        for i, (k, c) in enumerate(items):
            v = [s[k]['purity_mean'] for s in S]
            m, e = ms(v)
            a.bar(i, m, .65, yerr=e, color=c, alpha=.5 if k.endswith('OrtSAE') else 1, error_kw={'lw': .8, 'capsize': 2})
            dots(a, i, v, INK)
        a.set_xticks(range(len(items)), labs, fontsize=6, rotation=30 if len(items) > 4 else 0)
        a.set_title(t, fontsize=7.5)
        a.grid(axis='x', visible=False)
    ax[0].set_ylabel('top-token share of activation mass')
    save(fig, 'ri_units')
    return {'n_seeds': len(S)}


def roles():
    S = seeds('senses')
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    cm = plt.matplotlib.colors.LinearSegmentedColormap.from_list('d', [BLUE, '#f2f2f0', ORANGE])
    for a, (key, ttl) in zip(ax, (('BP', 'Backpack'), ('BPo', 'Backpack + penalty'), ('BC', 'Backpack + CoCoMix'))):
        R = []
        for s in S:
            m = np.array(s['roles'][key]['mean'])
            R.append(m[np.argsort(-m[:, 0])])  # senses ordered by how strongly they promote their own token
        mu, se = ms(np.stack(R))
        a.imshow(mu * [[.1, 1, 1]], cmap=cm, vmin=-.5, vmax=.5, aspect='auto')
        for i in range(mu.shape[0]):
            for j in range(3):
                a.text(j, i, f'{mu[i, j]:.2f}' + (f'\n$\\pm${se[i, j]:.2f}' if len(S) > 1 else ''), ha='center', va='center', fontsize=6.5)
        a.set_xticks(range(3), ['own token\n(z-score)', 'next token\n(bigram corr)', 'related words\n(PMI corr)'], fontsize=6.5)
        a.set_yticks(range(4), ['copy sense', 'sense 2', 'sense 3', 'sense 4'] if key == 'BP' else [], fontsize=6.5)
        a.set_title(ttl, fontsize=8)
        a.grid(False)
    save(fig, 'ri_roles')
    return {'n_seeds': len(S)}


def gender():
    S = seeds('gender')
    cfg = [('BP', 'Backpack', AQUA), ('BPo', '+ penalty', BLUE), ('BC', '+ CoCoMix', ORANGE), ('BCo', '+ CoCoMix + pen.', VIOLET)]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={'wspace': .4})
    for i, (k, t, c) in enumerate(cfg):
        red = [max(s[k]['reduction_mean']) for s in S]
        kl = [s[k]['kl_mean'][int(np.argmax(s[k]['reduction_mean']))] for s in S]
        for a, v in ((ax[0], red), (ax[1], kl)):
            m, e = ms(v)
            a.bar(i, m, .65, yerr=e, color=c, error_kw={'lw': .8, 'capsize': 2})
            dots(a, i, v, INK)
        ax[2].scatter(kl, red, s=14, color=c, label=t, alpha=.85, lw=0)
    for a in ax[:2]:
        a.set_xticks(range(4), [c[1] for c in cfg], fontsize=6, rotation=25)
        a.grid(axis='x', visible=False)
    ax[0].set_ylabel('he/she gap removed by\nthe best single sense')
    ax[1].set_ylabel('collateral KL (nats)')
    ax[2].set_xlabel('collateral KL (nats)')
    ax[2].set_ylabel('gap removed')
    ax[2].legend(fontsize=6)
    for a, t in zip(ax, ('(a) effect of zeroing one sense', '(b) collateral change', '(c) one dot per trained model')):
        a.set_title(t, fontsize=7.5)
    save(fig, 'ri_gender')
    return {'n_seeds': len(S)}


def saequal():
    E = errs()
    keys, names = ('P', 'CM', 'BP', 'BC'), {'P': 'plain', 'CM': 'CoCoMix', 'BP': 'Backpack', 'BC': 'BP+CM'}
    mets = [('ev', 'explained variance'), ('ce_recovered', 'CE recovered'), ('max_cos', 'max cosine (lower = more distinct)'), ('absorption', 'first-letter absorption'), ('probe_f1', 'first-letter probe F1')]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 3.8), gridspec_kw={'hspace': .5, 'wspace': .3})
    ax = axs.flatten()
    w = .38
    for a, (k, ttl) in zip(ax[:5], mets):
        for gi, (g, c, lb) in enumerate(((0., BLUE, 'TopK SAE'), (.25, ORANGE, 'OrtSAE'))):
            for i, key in enumerate(keys):
                nm = V.CELLS[key]
                v = [e[f'{nm}_s{si}'][f'sae_g{g}'][k][0] for si, e in E.items() if f'{nm}_s{si}' in e]
                m, e = ms(v)
                a.bar(i + (gi - .5) * w, m, w, yerr=e, color=c, label=lb if i == 0 else None, error_kw={'lw': .8, 'capsize': 1.5})
                dots(a, i + (gi - .5) * w, v, INK, .07)
        a.set_xticks(range(4), [names[k_] for k_ in keys], fontsize=6.5, rotation=25)
        a.set_title(ttl, fontsize=7.5)
        a.grid(axis='x', visible=False)
    ax[5].axis('off')
    h, l = ax[0].get_legend_handles_labels()
    ax[5].legend(h, l, fontsize=7, loc='center')
    save(fig, 'ri_sae')
    return {'n_seeds': len(E)}


def absorb():
    S = seeds('absorb')
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4))
    for i, (k, c) in enumerate((('P', BLUE), ('CM', ORANGE), ('BP', AQUA))):
        v = [s[k]['diff'][0] for s in S]
        m, e = ms(v)
        ax.bar(i, m, .6, yerr=e, color=c, error_kw={'lw': .8, 'capsize': 2})
        dots(ax, i, v, INK)
    ax.axhline(0, color=MUTED, lw=.8)
    ax.set_xticks(range(3), ['plain', 'CoCoMix', 'Backpack'], fontsize=7)
    ax.set_ylabel('absorption: OrtSAE $-$ TopK')
    ax.grid(axis='x', visible=False)
    save(fig, 'ri_absorb')
    return {'n_seeds': len(S)}


def reliance():
    S = seeds('reliance')
    fig, ax = plt.subplots(1, 1, figsize=(3.3, 2.3))
    for i, (k, c) in enumerate((('CM', ORANGE), ('CMo', '#f5b9a0'), ('BC', BLUE), ('BCo', '#9cc3f0'))):
        v = [s[k]['d_ce_shuffled'] for s in S]
        m, e = ms(v)
        ax.bar(i, m, .65, yerr=e, color=c, error_kw={'lw': .8, 'capsize': 2})
        dots(ax, i, v, INK)
    ax.set_xticks(range(4), ['CoCoMix', '+ pen.', 'BP + CM', '+ pen.'], fontsize=6.5)
    ax.set_ylabel('loss increase when concepts\nare shuffled across stories (nats)', fontsize=7)
    ax.grid(axis='x', visible=False)
    save(fig, 'ri_reliance')
    return {'n_seeds': len(S)}


def faith():
    """ri_faith: predicted versus actual effect of removing one unit (seed-0 scatter) and the correlation over seeds."""
    S = seeds('faith')
    pts = S[0]['points']
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.4), gridspec_kw={'width_ratios': [1, 1, 1.15], 'wspace': .38})
    for a, (k, t, c) in zip(ax[:2], (('P|latent', '(a) plain LM: post-hoc SAE latent', BLUE), ('BP|sense', '(b) Backpack: (sense, source word) term', AQUA))):
        p, y = np.array(pts[k])
        a.scatter(p, y, s=5, color=c, alpha=.5, lw=0)
        lim = [min(p.min(), y.min()), max(p.max(), y.max())]
        a.plot(lim, lim, color=INK, lw=.7, ls=':')
        a.set_title(t, fontsize=7)
        a.set_xlabel('effect predicted by the explanation', fontsize=7)
    ax[0].set_ylabel('effect of actually removing it\n(change in the predicted logit)', fontsize=7)
    cols = [('P|latent', 'plain\nlatent', BLUE), ('CM|latent', 'CoCoMix\nlatent', ORANGE), ('BP|latent', 'Backpack\nlatent', AQUA), ('BC|latent', 'BP+CM\nlatent', VIOLET), ('BP|sense', 'Backpack\nsense term', AQUA), ('BC|sense', 'BP+CM\nsense term', VIOLET)]
    for i, (k, _, c) in enumerate(cols):
        v = [s['stats'][k]['r'] for s in S]
        m, e = ms(v)
        ax[2].bar(i, m, .65, yerr=e, color=c, hatch='' if 'sense' in k else '//', alpha=.9, error_kw={'lw': .8, 'capsize': 2})
        dots(ax[2], i, v, INK)
    ax[2].set_xticks(range(len(cols)), [c[1].replace('\n', ' ') for c in cols], fontsize=6, rotation=35, ha='right')
    ax[2].set_ylabel('correlation, predicted vs actual', fontsize=7)
    ax[2].set_title('(c) over trained models', fontsize=7)
    ax[2].grid(axis='x', visible=False)
    save(fig, 'ri_faith')
    return {k: [round(float(ms([s['stats'][k]['r'] for s in S])[0]), 4)] for k in S[0]['stats']}


# ---------------------------------------------------------------- cost: quality table and training curves

def quality_main():
    """ti_quality_main: eight models over the seeds present: perplexity, paired CE difference to the plain model of the same seed, concept head precision@4, steering KL, parameters and relative training time."""
    E = errs()
    rows, out = [], {}
    t0 = rd(f'{RUNS}/{V.N["P"]}/timing.json')['train_s']
    for key, c in V.CELLS.items():
        g = lambda f: [e[f'{c}_s{i}'][f][0] for i, e in E.items() if f'{c}_s{i}' in e and f in e[f'{c}_s{i}']]
        ppl, dce, hit, kl = g('ppl'), g('d_ce_vs_base'), g('concept_hit'), g('steer_kl')
        cfg = V.load_eval(f'{c}_s0')
        t = rd(f'{RUNS}/{c}_s0/timing.json')['train_s'] / t0
        rows.append(f"{NAMES[key]} & {cfg['n_params'] / 1e6:.2f} & {t:.1f}$\\times$ & {pm(ppl, 2)} & {'--' if key == 'P' else pm(dce, 3)} & {pm(hit, 2) if hit else '--'} & {pm(kl, 2) if kl else '--'} \\\\")
        out[key] = {'ppl': ms(ppl), 'dce': ms(dce), 'n_seeds': len(ppl)}
    write_tex('ti_quality_main', '\n'.join([r'\begin{tabular}{@{}lrrrrrr@{}}', r'\toprule', r'model & params (M) & train time & test ppl & $\Delta$CE vs plain & concept hit@4 & steering KL \\', r'\midrule', *rows, r'\bottomrule', r'\end{tabular}']) + '\n')
    return out


def curves():
    """ri_curves: dev cross-entropy over training (mean over seeds) and the gap to the plain model of the same seed."""
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    col = {'P': INK, 'CC': VIOLET, 'CM': ORANGE, 'CMo': '#f5b9a0', 'BP': AQUA, 'BPo': '#8fdcc0', 'BC': BLUE, 'BCo': '#9cc3f0'}
    L = {}
    for key, c in V.CELLS.items():
        runs = []
        for s in range(5):
            p = f'{RUNS}/{c}_s{s}/log.jsonl'
            if os.path.exists(p) and os.path.exists(f'{RUNS}/{c}_s{s}/eval_test.json'):
                rows = [json.loads(ln) for ln in pathlib.Path(p).read_text().splitlines()]
                rows = [r for r in rows if 'dev_ce' in r and r['step'] > 0]
                runs.append((np.array([r['step'] for r in rows]), np.array([r['dev_ce'] for r in rows])))
        L[key] = runs
    for key, runs in L.items():
        if not runs:
            continue
        n = min(len(r[0]) for r in runs)
        st = runs[0][0][:n]
        Y = np.stack([r[1][:n] for r in runs])
        P = np.stack([r[1][:n] for r in L['P']][:len(runs)]) if len(L['P']) >= len(runs) else None
        m, e = ms(Y)
        sel = st >= 3000
        sm = lambda y: np.convolve(np.pad(y, 2, mode='edge'), np.ones(5) / 5, mode='valid')
        lab = NAMES[key]
        ls = '--' if key.endswith('o') else '-'
        ax[0].plot(st[sel], sm(m)[sel], color=col[key], lw=1.1, ls=ls, label=lab)
        if P is not None and key != 'P':
            gm, ge = ms(Y - P)
            ax[1].plot(st[sel], sm(gm)[sel], color=col[key], lw=1.1, ls=ls)
            ax[1].fill_between(st[sel], sm(gm - ge)[sel], sm(gm + ge)[sel], color=col[key], alpha=.15, lw=0)
    ax[0].set_ylabel('dev cross-entropy')
    ax[1].set_ylabel('gap to the plain model (nats)')
    ax[1].axhline(0, color=MUTED, lw=.8)
    for a in ax:
        a.set_xlabel('training step')
    ax[0].legend(fontsize=5.8, ncol=2)
    save(fig, 'ri_curves')
    return {k: len(v) for k, v in L.items()}


ANALYSES = ['common', 'faith', 'steer', 'geometry', 'units', 'roles', 'gender', 'saequal', 'absorb', 'reliance', 'quality_main', 'curves']

if __name__ == '__main__':
    for name in sys.argv[1:] or ANALYSES:
        print(name, flush=True)
        print(json.dumps(globals()[name](), default=float)[:300], flush=True)
