"""Comparison figures and text aggregates built from saved artifacts only:
runs/<name>/{eval_test.json, log.jsonl, sae_decoders.pt, ckpt.pt} and runs/teacher/<t>/test.pt.

  python plots.py [runs] [figdir] [resdir]     figures -> figdir (pdf + png);  text -> resdir (aggregates.csv, results.md, effects.md,
                                                senses.md, concepts.md, summary.txt)
Seeds are the unit of replication: error bars are std over seeds; CIs are bootstrap over paired seeds (crude at n = 3)."""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

import grid
import paths
from data import D, windows
from model import LM

plt.switch_backend('Agg')
SAE = ('0.0', '0.25')
SAE_NAME = {'0.0': 'TopK SAE', '0.25': 'OrtSAE'}
SAE_KEYS = 'ev l0 dead ce_recovered kl_score mean_max_cos meta_ev probe_dense_acc probe_f1 absorption'.split()
KEYS = 'ce acc ppl n_params concept_ce concept_hit steer_kl sense_cos sense_maxcos2 gender_bias gender_best_reduction'.split()
UP = 'acc concept_hit steer_kl gender_best_reduction ev ce_recovered kl_score probe_f1'.split()  # higher is better
DOWN = 'ce ppl concept_ce sense_maxcos2 sense_cos mean_max_cos meta_ev dead absorption'.split()  # lower is better
EFFECT_METRICS = ['ce', 'concept_hit', 'steer_kl', 'gender_best_reduction', 'sae0.0_ev', 'sae0.0_absorption', 'sae0.0_mean_max_cos',
                  'sae0.0_meta_ev', 'sae0.25_ev', 'sae0.25_absorption']


def direction(col):
    """+1 higher is better, -1 lower is better, 0 undefined (conventions and caveats: project/goals/plots_plan.md)."""
    base = col.split('_', 1)[-1] if col.startswith('sae') else col
    return 1 if base in UP else -1 if base in DOWN else 0


def parse(name):
    """'k4_cm2_ort_og0.1_s1' -> cell and factors."""
    cell, seed = name.rsplit('_s', 1)
    k, cm, teacher, og = cell.split('_')
    return {'cell': cell, 'seed': int(seed), 'k': int(k[1:]), 'cm': int(cm[2:]), 'teacher': teacher, 'og': float(og[2:])}


def cell_name(k, cm, teacher, og):
    return f'k{k}_cm{cm}_{teacher}_og{og:g}'


def label(cell):
    p = parse(cell + '_s0')
    if p['teacher'].startswith('wm'):
        return 'baseline d=' + p['teacher'][2:]
    parts = [f"BP{p['k']}"] * bool(p['k'])
    parts += [['', 'CE', 'CoCoMix', 'mixOnly'][p['cm']] + ('(ort)' if p['teacher'] == 'ort' else '')] * bool(p['cm'])
    parts += ['ortho'] * bool(p['og'])
    return '+'.join(parts) or 'baseline'


def order(cells):
    """Readable cell order: baselines, Backpack, CoCoMix modes, ortho."""
    def key(cell):
        q = parse(cell + '_s0')
        return q['k'], q['cm'], q['teacher'], q['og']
    return sorted(set(cells), key=key)


def colors(cells):
    return {c: plt.cm.tab20(i % 20) for i, c in enumerate(order(cells))}


def save(fig, figdir, name):
    fig.tight_layout()
    fig.savefig(f'{figdir}/{name}.pdf')
    fig.savefig(f'{figdir}/{name}.png', dpi=130)
    plt.close(fig)


def load(out):
    """One row per run: parsed name, scalar metrics, flattened SAE metrics, and the raw JSON under 'raw'."""
    rows = []
    for path in sorted(glob.glob(f'{out}/*/eval_test.json')):
        with open(path) as f:
            r = json.load(f)
        row = {**parse(r['name']), **{k: r[k] for k in KEYS if k in r}, 'raw': r}
        row.update({f't_eval_{k}': v for k, v in r.get('time', {}).items()})
        if os.path.exists(tpath := os.path.join(os.path.dirname(path), 'timing.json')):
            with open(tpath) as f:
                row.update({f't_train_{k}': v for k, v in json.load(f).items()})
        for g in SAE:
            row.update({f'sae{g}_{k}': v for k, v in r.get(f'sae_g{g}', {}).items() if k in SAE_KEYS})
        rows.append(row)
    return pd.DataFrame(rows)


def load_model(out, name):
    ck = torch.load(f'{out}/{name}/ckpt.pt', map_location='cpu')
    model = LM(**ck['cfg']).eval()
    model.load_state_dict(ck['sd'])
    return model


# ---------------------------------------------------------------- figures from eval JSONs / logs

def fig_lm(df, figdir):
    a = df.groupby('cell').agg(m=('ppl', 'mean'), s=('ppl', 'std'), n=('n_params', 'first')).sort_values('m')
    C = colors(df.cell)
    fig, ax = plt.subplots(figsize=(7, .3 * len(a) + 1))
    ax.barh([label(c) for c in a.index], a.m, xerr=a.s.fillna(0), color=[C[c] for c in a.index])
    for i, (c, r) in enumerate(a.iterrows()):
        ax.plot(df[df.cell == c].ppl, [i] * (df.cell == c).sum(), 'k.', ms=3)
        ax.text(r.m, i, f' {r.n / 1e6:.2f}M', va='center', fontsize=6)
    ax.set_xlim(a.m.min() * .97)
    ax.set_xlabel('test perplexity (lower is better); label = #params')
    save(fig, figdir, 'fig_lm')


def fig_effects(df, figdir):
    sub = df[(df.og == 0) & df.teacher.isin(['topk', '-'])]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for ax, col in zip(axes, ['ce', 'sae0.0_absorption', 'sae0.25_absorption']):
        for k in (0, 4):
            g = sub[sub.k == k].groupby('cm')[col].agg(['mean', 'std'])
            ax.errorbar(g.index, g['mean'], g['std'].fillna(0), marker='o', capsize=2, label=f'Backpack k={k}')
        ax.set_xticks(range(4))
        ax.set_xticklabels(['none', 'CE only', 'CE+mix', 'mix only'], fontsize=7)
        ax.set_title(col, fontsize=9)
    axes[0].legend(fontsize=7)
    save(fig, figdir, 'fig_effects')


def fig_curves(out, figdir):
    rows = []
    for path in glob.glob(f'{out}/*/log.jsonl'):
        name = os.path.basename(os.path.dirname(path))
        if '_s' in name:
            with open(path) as f:
                rows += [{**json.loads(line), **parse(name)} for line in f]
    d = pd.DataFrame(rows)
    C = colors(d.cell)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in zip(axes, ['dev_ce', 'lc'], ['dev CE (monitoring only)', 'concept CE (training)']):
        g = d[d.cm.isin([1, 2])] if col == 'lc' else d
        for c, x in g.groupby(['cell', 'step'])[col].mean().reset_index().groupby('cell'):
            ax.plot(x.step, x[col], color=C[c], lw=1, label=label(c))
        ax.set_xlabel('step')
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=5, ncol=2)
    save(fig, figdir, 'fig_curves')


def fig_timing(out, figdir):
    """Wall-clock view: dev CE against training time per cell (does a mechanism buy quality per second?) and ms/step per cell."""
    rows, tim = [], []
    for path in glob.glob(f'{out}/*/log.jsonl'):
        name = os.path.basename(os.path.dirname(path))
        if '_s' in name:
            with open(path) as f:
                rows += [{**json.loads(line), **parse(name)} for line in f]
    d = pd.DataFrame(rows)
    C = colors(d.cell)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c, x in d.groupby(['cell', 'step'])[['elapsed_s', 'dev_ce', 'ce']].mean().reset_index().groupby('cell'):
        axes[0].plot(x.elapsed_s, x.dev_ce, color=C[c], lw=1, label=label(c))
        axes[1].plot(x.elapsed_s, x.ce, color=C[c], lw=1)
    axes[0].set_xlabel('training wall time (s)')
    axes[0].set_ylabel('dev CE')
    axes[0].set_title('dev CE vs wall time (monitoring only)', fontsize=9)
    axes[0].legend(fontsize=5, ncol=2)
    axes[1].set_xlabel('training wall time (s)')
    axes[1].set_ylabel('train CE (100-step mean)')
    axes[1].set_title('train CE vs wall time', fontsize=9)
    for path in glob.glob(f'{out}/*/timing.json'):
        name = os.path.basename(os.path.dirname(path))
        if '_s' in name:
            with open(path) as f:
                tim.append({**json.load(f), **parse(name)})
    t = pd.DataFrame(tim).groupby('cell').ms_per_step.mean().reindex(order(d.cell.unique()))
    axes[2].barh([label(c) for c in t.index], t.values, color=[C[c] for c in t.index])
    axes[2].set_xlabel('ms / step (excl. compile and dev eval)')
    axes[2].tick_params(labelsize=6)
    axes[2].set_title('training cost per step', fontsize=9)
    save(fig, figdir, 'fig_timing')


def timing_md(df):
    """Seed-mean wall-clock table per cell: training (s, ms/step, tokens/s, compile) and evaluation (s, SAE share)."""
    cols = {'t_train_train_s': 'train s', 't_train_ms_per_step': 'ms/step', 't_train_tok_s': 'tok/s', 't_train_first_step_s': 'compile+1st s',
            't_train_peak_mem_gb': 'peak GB', 't_eval_total_s': 'eval s', 't_eval_lm_metrics_s': 'lm_metrics s'}
    have = [c for c in cols if c in df.columns]
    if not have:
        return 'no timing found\n'
    g = df.groupby('cell')[have].mean().reindex(order(df.cell.unique()))
    lines = ['| cell | ' + ' | '.join(cols[c] for c in have) + ' |', '|' + '---|' * (len(have) + 1)]
    lines += [f'| {label(c)} | ' + ' | '.join('' if np.isnan(v) else f'{v:.1f}' for v in row) + ' |' for c, row in zip(g.index, g.values)]
    tot = {c: df[c].sum() for c in ('t_train_total_s', 't_eval_total_s') if c in df.columns}
    return '\n'.join(lines) + '\n\nsum over runs (s): ' + json.dumps({k: round(v) for k, v in tot.items()}) + '\n'


def fig_concept(df, figdir):
    s = df[df.cm > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, col in zip(axes, ('concept_hit', 'steer_kl')):
        g = s.groupby('cell')[col].agg(['mean', 'std']).reindex(order(s.cell)).dropna(subset=['mean'])
        ax.bar([label(i) for i in g.index], g['mean'], yerr=g['std'].fillna(0))
        ax.set_title(col, fontsize=9)
        ax.tick_params(axis='x', rotation=70, labelsize=6)
    save(fig, figdir, 'fig_concept')


def fig_gender(df, figdir):
    s = df[df.k > 0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
    g = s.groupby('cell').gender_best_reduction.agg(['mean', 'std']).reindex(order(s.cell))
    axes[0].bar([label(i) for i in g.index], g['mean'], yerr=g['std'].fillna(0))
    axes[0].tick_params(axis='x', rotation=60, labelsize=6)
    axes[0].set_title('best single-sense zeroing: bias reduction', fontsize=9)
    for _, r in s.iterrows():
        if 'gender_kl_zero_sense' in r.raw:
            axes[1].scatter(r.raw['gender_kl_zero_sense'], r.raw['gender_bias_zero_sense'], c='C1' if r.og else 'C0', s=12, alpha=.7)
    axes[1].set_xlabel('KL cost of zeroing a sense')
    axes[1].set_ylabel('|log p(he)/p(she)| after')
    axes[1].set_title('per sense (orange = ortho penalty)', fontsize=9)
    for j, cell in enumerate(('k4_cm0_-_og0', 'k4_cm0_-_og0.1')):  # per-profession effect, seed-mean
        rs = [r.raw['gender_prof'] for _, r in s[s.cell == cell].iterrows() if 'gender_prof' in r.raw]
        if rs:
            profs = sorted(set.intersection(*[set(p) for p in rs]))
            x = np.arange(len(profs))
            axes[2].bar(x + .4 * j, [np.mean([min(p[w]['zero']) for p in rs]) for w in profs], .4, label=f'{label(cell)}: best sense zeroed')
            if j == 0:
                axes[2].plot(x, [np.mean([p[w]['base'] for p in rs]) for w in profs], 'k_', ms=12, label='before')
            axes[2].set_xticks(x)
            axes[2].set_xticklabels(profs, rotation=70, fontsize=6)
    axes[2].set_title('per profession |log p(he)/p(she)|', fontsize=9)
    axes[2].legend(fontsize=6)
    save(fig, figdir, 'fig_gender')


def fig_sae(df, figdir):
    metrics = 'ev ce_recovered kl_score mean_max_cos meta_ev probe_f1 absorption dead'.split()
    cells = order(df.cell)
    x = np.arange(len(cells))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for ax, m in zip(axes.flat, metrics):
        for j, g in enumerate(SAE):
            s = df.groupby('cell')[f'sae{g}_{m}'].agg(['mean', 'std']).reindex(cells)
            ax.errorbar(x + .2 * j, s['mean'], s['std'].fillna(0), fmt='o', ms=3, label=SAE_NAME[g])
        ax.set_title(m, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([label(c) for c in cells], rotation=80, fontsize=5)
    axes[0, 0].legend(fontsize=7)
    save(fig, figdir, 'fig_sae')


def fig_pareto(df, figdir):
    C = colors(df.cell)
    fig, ax = plt.subplots(figsize=(5, 4))
    for g, marker in zip(SAE, 'o^'):
        for c, r in df.groupby('cell')[[f'sae{g}_ev', f'sae{g}_absorption']].mean().iterrows():
            ax.scatter(r.iloc[0], r.iloc[1], marker=marker, color=C[c], s=25)
        ax.scatter([], [], marker=marker, c='k', label=SAE_NAME[g])
    ax.set_xlabel('explained variance (higher better)')
    ax.set_ylabel('absorption (lower better)')
    ax.legend(fontsize=7)
    save(fig, figdir, 'fig_pareto')


def fig_letters(df, figdir):
    """Per-letter probe F1 and absorption proxy for the baseline LM, TopK vs OrtSAE (seed-mean)."""
    rs = [r.raw for _, r in df[df.cell == 'k0_cm0_-_og0'].iterrows()]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    for ax, idx, title in zip(axes, (0, 1), ('1-sparse probe F1', 'absorption proxy (lower better)')):
        for j, g in enumerate(SAE):
            per = [r[f'sae_g{g}']['per_letter'] for r in rs]
            letters = sorted(set.intersection(*[set(p) for p in per]))
            ax.bar(np.arange(len(letters)) + .4 * j, [np.mean([p[c][idx] for p in per]) for c in letters], .4, label=SAE_NAME[g])
            ax.set_xticks(np.arange(len(letters)))
            ax.set_xticklabels(letters)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    save(fig, figdir, 'fig_letters')


def fig_heatmap(df, figdir):
    cols = [c for c in KEYS + [f'sae{g}_{k}' for g in SAE for k in SAE_KEYS] if c in df and direction(c) and c != 'n_params']
    m = df.groupby('cell')[cols].mean()
    z = (m - m.mean()) / m.std().replace(0, np.nan) * pd.Series({c: direction(c) for c in cols})
    z = z.loc[z.mean(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(.45 * len(cols) + 3, .3 * len(z) + 1.5))
    im = ax.imshow(z.values, cmap='RdBu', vmin=-2, vmax=2, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=80, fontsize=6)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([label(c) for c in z.index], fontsize=6)
    fig.colorbar(im, label='z-score across cells (blue = better)')
    save(fig, figdir, 'fig_heatmap')


# ---------------------------------------------------------------- figures from checkpoints

def fig_ckpt(out, figdir):
    """Weight-space evidence: mean k x k sense cosine, concept-compression column norms, SAE decoder max-cos histograms."""
    exists = lambda n: os.path.exists(f'{out}/{n}/ckpt.pt')
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, name in zip(axes[:2], ('k4_cm0_-_og0_s0', 'k4_cm0_-_og0.1_s0')):
        if exists(name):
            m = load_model(out, name)
            S = F.normalize(m.sense(m.E.weight).view(m.E.num_embeddings, m.k, -1), dim=-1)
            ax.imshow((S @ S.mT).mean(0).detach(), vmin=-1, vmax=1, cmap='RdBu')
            ax.set_title(f'mean sense cosine\n{label(name[:-3])}', fontsize=8)
    for name in ('k0_cm2_topk_og0_s0', 'k0_cm2_topk_og0.1_s0'):
        if exists(name):
            axes[2].hist(load_model(out, name).W.weight.norm(dim=0).detach().numpy(), bins=30, alpha=.6, label=label(name[:-3]))
    axes[2].set_title('concept-compression column norms', fontsize=8)
    axes[2].legend(fontsize=6)
    dec = f'{out}/k0_cm0_-_og0_s0/sae_decoders.pt'
    if os.path.exists(dec):
        W = torch.load(dec)
        max_cos = lambda w: (F.normalize(w, dim=-1) @ F.normalize(w, dim=-1).T).fill_diagonal_(-1).max(1).values.numpy()
        for g, w in W.items():
            axes[3].hist(max_cos(w), bins=30, alpha=.5, label=SAE_NAME[str(g)])
        axes[3].hist(max_cos(torch.randn_like(next(iter(W.values())))), bins=30, histtype='step', color='k', label='random rows')
        axes[3].legend(fontsize=6)
        axes[3].set_title('SAE decoder max-cos (baseline LM)', fontsize=8)
    save(fig, figdir, 'fig_ckpt')


def fig_decomp(out, figdir, name='k4_cm0_-_og0_s0', n_pos=8):
    """Exact Backpack decomposition: contribution of each sense to the logit of the model's top token at confident positions.
    Keeping a single sense (via the sc intervention) isolates it; by linearity the contributions sum to the full logit."""
    if not os.path.exists(f'{out}/{name}/ckpt.pt'):
        return
    m = load_model(out, name)
    tk = Tokenizer.from_file(f'{D}/tok.json')
    x = windows('test', m.P.num_embeddings)[0][:1]
    T = x.shape[1]
    with torch.no_grad():
        logits = m(x)[0][0]
        parts = torch.stack([m(x, sc=torch.eye(m.k)[ell].expand(1, T, m.k))[0][0] for ell in range(m.k)])  # (k, T, V)
    pos = logits.softmax(-1).max(-1).values[8:].topk(n_pos).indices.sort().values + 8
    top = logits.argmax(-1)[pos]
    contrib = parts[:, pos, top]
    err = (contrib.sum(0) - logits[pos, top]).abs().max().item()
    tok = lambda i: tk.id_to_token(int(i)).replace('Ġ', '_')
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for ell in range(m.k):
        ax.bar(np.arange(n_pos) + .2 * ell, contrib[ell], .2, label=f'sense {ell}')
    ax.set_xticks(np.arange(n_pos))
    ax.set_xticklabels([f'{tok(x[0, p])}→{tok(t)}' for p, t in zip(pos, top)], rotation=40, fontsize=7)
    ax.set_title(f'{label(name[:-3])}: per-sense logit contributions (max |sum - logit| = {err:.1e})', fontsize=9)
    ax.legend(fontsize=7)
    save(fig, figdir, 'fig_decomp')


# ---------------------------------------------------------------- text aggregates

def boot_ci(x, n=2000):
    rng = np.random.default_rng(0)
    return np.percentile([rng.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5])


def results_table(df):
    headline = ('ppl', 'acc', 'concept_hit', 'steer_kl', 'gender_best_reduction', 'sae0.0_ev', 'sae0.0_absorption', 'sae0.25_ev', 'sae0.25_absorption')
    cols = [c for c in headline if c in df]
    base = df[df.cell == 'k0_cm0_-_og0'].set_index('seed').ppl
    lines = ['| cell | n | params | ' + ' | '.join(cols) + ' | Δppl vs baseline [95% CI] |', '|' + '---|' * (len(cols) + 4)]
    for c in order(df.cell):
        g = df[df.cell == c]
        d = (g.set_index('seed').ppl - base).dropna()
        ci = boot_ci(d.values) if len(d) > 1 else [np.nan] * 2
        cells = [f'{g[k].mean():.3f}±{g[k].std():.3f}' if g[k].notna().any() else '-' for k in cols]
        lines.append(f'| {label(c)} | {len(g)} | {g.n_params.iloc[0] / 1e6:.2f}M | ' + ' | '.join(cells) + f' | {d.mean():+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |')
    return '\n'.join(lines) + '\n'


def factor_effects(df):
    """Matched-pair main effects: for every pair of cells differing in exactly one factor, the difference of seed-mean metrics; mean over pairs."""
    m = df[~df.teacher.str.startswith('wm')].groupby('cell')[[c for c in EFFECT_METRICS if c in df]].mean()
    p = {c: parse(c + '_s0') for c in m.index}
    pairs = {
        'Backpack k=4 vs k=0': lambda q: cell_name(0, q['cm'], q['teacher'], q['og']) if q['k'] else None,
        'concept CE only vs none': lambda q: cell_name(q['k'], 0, '-', q['og']) if q['cm'] == 1 else None,
        'CE+interleave vs CE only': lambda q: cell_name(q['k'], 1, q['teacher'], q['og']) if q['cm'] == 2 else None,
        'interleave only vs none': lambda q: cell_name(q['k'], 0, '-', q['og']) if q['cm'] == 3 else None,
        'OrtSAE teacher vs TopK teacher': lambda q: cell_name(q['k'], q['cm'], 'topk', q['og']) if q['teacher'] == 'ort' else None,
        'ortho penalty 0.1 vs 0': lambda q: cell_name(q['k'], q['cm'], q['teacher'], 0) if q['og'] else None,
    }
    lines = ['| factor (A vs B) | n pairs | ' + ' | '.join(m.columns) + ' |', '|' + '---|' * (len(m.columns) + 2)]
    for name, partner in pairs.items():
        diffs = [m.loc[c] - m.loc[partner(q)] for c, q in p.items() if partner(q) in m.index]
        if diffs:
            d = pd.DataFrame(diffs)
            lines.append(f'| {name} | {len(d)} | ' + ' | '.join(f'{d[c].mean():+.4f}' if d[c].notna().any() else '-' for c in m.columns) + ' |')
    return '\n'.join(lines) + '\n'


def senses_md(df):
    """Top-8 vocabulary tokens of each sense (seed 0, Backpack cells without / with the ortho penalty)."""
    out = []
    for cell in ('k4_cm0_-_og0', 'k4_cm0_-_og0.1'):
        for _, row in df[(df.cell == cell) & (df.seed == 0)].iterrows():
            if 'sense_top8' in row.raw:
                out.append(f'### {label(cell)} (seed 0)\n')
                for w, senses in row.raw['sense_top8'].items():
                    out.append(f'**{w}**\n' + '\n'.join(f'* sense {i}: ' + ' '.join(t.replace('Ġ', '_') for t in toks) for i, toks in enumerate(senses)) + '\n')
    return '\n'.join(out) or 'no Backpack runs found\n'


def concepts_md(out, n_top=10):
    """What do the teacher concepts fire on? Per teacher: concept -> token counts on the test split, token purity (share of label events
    covered by each concept's most frequent token), and the top tokens of the most frequent concepts. Returns (markdown, {teacher: purity})."""
    tk = Tokenizer.from_file(f'{D}/tok.json')
    lines, purity = [], {}
    for t in ('topk', 'ort'):
        path = f'{out}/teacher/{t}/test.pt'
        if not os.path.exists(path):
            continue
        lab = torch.load(path).long()
        X = windows('test', lab.shape[1])[0][:len(lab)]
        V, n_c = int(X.max()) + 1, int(lab.max()) + 1
        counts = torch.bincount((lab * V + X[..., None]).flatten(), minlength=n_c * V).view(n_c, V).float()
        tot = counts.sum(1)
        purity[t] = (counts.max(1).values.sum() / tot.sum()).item()
        lines.append(f'### {t} teacher: token purity {purity[t]:.3f}, concepts used {(tot > 0).sum().item()}/{n_c}\n')
        for c in tot.topk(n_top).indices.tolist():
            top = counts[c].topk(8)
            toks = ' '.join(f"{tk.id_to_token(i).replace('Ġ', '_')}({v / tot[c]:.2f})" for v, i in zip(top.values.tolist(), top.indices.tolist()))
            lines.append(f'* concept {c} ({int(tot[c])} events): {toks}')
        lines.append('')
    return '\n'.join(lines) or 'no teacher labels found\n', purity


def write_text(df, out, resdir):
    """Text aggregates for reading / filling the report: aggregates.csv (mean, std, count per cell and metric), results.md, effects.md,
    senses.md, concepts.md, timing.md, summary.txt."""
    os.makedirs(resdir, exist_ok=True)
    skip = ('raw', 'cell', 'teacher', 'seed', 'k', 'cm', 'og')
    agg = df.groupby('cell')[[c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]].agg(['mean', 'std', 'count'])
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg.to_csv(f'{resdir}/aggregates.csv')
    concepts, purity = concepts_md(out)
    table, effects = results_table(df), factor_effects(df)
    for name, text in (('results.md', table), ('effects.md', effects), ('senses.md', senses_md(df)), ('concepts.md', concepts), ('timing.md', timing_md(df))):
        with open(f'{resdir}/{name}', 'w') as f:
            f.write(text)
    best = df.groupby('cell').ppl.mean().sort_values()
    with open(f'{resdir}/summary.txt', 'w') as f:
        f.write(f"runs: {len(df)}  cells: {df.cell.nunique()}  seeds: {sorted(df.seed.unique())}\n")
        f.write('best cells by test ppl (seed-mean): ' + ', '.join(f'{label(c)}={v:.3f}' for c, v in best.head(5).items()) + '\n')
        f.write('teacher token purity: ' + json.dumps(purity) + '\n\nmatched-pair effects (A minus B, seed-mean metrics):\n' + effects + '\nheadline table:\n' + table)


def main(out=paths.RUNS, figdir=paths.FIGS, resdir=paths.RESULTS):
    os.makedirs(figdir, exist_ok=True)
    df = load(out)
    jobs = [(fig_lm, df), (fig_effects, df), (fig_curves, out), (fig_timing, out), (fig_concept, df), (fig_gender, df), (fig_sae, df), (fig_pareto, df),
            (fig_letters, df), (fig_heatmap, df), (fig_ckpt, out), (fig_decomp, out)]
    for fn, arg in jobs:
        try:
            fn(arg, figdir)
        except Exception as e:  # partial grids are expected mid-run: report, never hide
            print(f'WARNING {fn.__name__} failed: {e!r}')
    try:
        write_text(df, out, resdir)
    except Exception as e:
        print(f'WARNING write_text failed: {e!r}')


def fake_runs(out, dims, seeds=(0, 1)):
    """Schema-faithful random artifacts (no training) to test the plotting code."""
    rng = np.random.default_rng(0)
    sae = {'ev': .8, 'l0': 16., 'dead': .05, 'ce_recovered': .9, 'kl_score': .9, 'mean_max_cos': .5, 'meta_ev': .6,
           'probe_dense_acc': .9, 'probe_f1': .6, 'absorption': .3}
    letters = {c: [.6, .3] for c in 'abcdefgh'}
    for c in grid.cfgs():
        if c['seed'] not in seeds:
            continue
        run_dir = f"{out}/{c['name']}"
        os.makedirs(run_dir, exist_ok=True)
        ce, k = 1.6 + .05 * rng.random(), c['k']
        r = {'name': c['name'], 'split': 'test', 'n_params': int(1e6 + 1e5 * k), 'ce': ce, 'acc': .5, 'ppl': float(np.exp(ce)),
             **{f'sae_g{g}': {**{a: v * (1 + .1 * rng.standard_normal()) for a, v in sae.items()}, 'per_letter': letters} for g in SAE}}
        if c['C']:
            r.update(concept_ce=3., concept_hit=.4 + .1 * rng.random(), steer_kl=.01 * rng.random())
        if k:
            prof = {w: {'base': 1., 'zero': list(rng.random(k)), 'kl': list(rng.random(k) * .1)} for w in ('doctor', 'nurse', 'chef')}
            r.update(sense_cos=.3, sense_maxcos2=.2, gender_bias=1., gender_best_reduction=rng.random(), gender_bias_zero_sense=list(rng.random(k)),
                     gender_kl_zero_sense=list(rng.random(k) * .1), gender_prof=prof, sense_top8={'he': [['_he', '_she'] * 4] * k})
        r['time'] = {'total_s': 20., 'lm_metrics_s': 2., 'load_s': 1.}
        with open(f'{run_dir}/eval_test.json', 'w') as f:
            json.dump(r, f)
        with open(f'{run_dir}/log.jsonl', 'w') as f:
            f.write('\n'.join(json.dumps({'step': s, 'ce': 3 - s / 100, 'lc': 2., 'dev_ce': 3 - s / 100 + .1 * rng.random(), 'elapsed_s': s / 10, 'tok_s': 1e6})
                              for s in range(0, 300, 100)))
        with open(f'{run_dir}/timing.json', 'w') as f:
            json.dump({'train_s': 30., 'ms_per_step': 15. + k, 'tok_s': 1e6, 'first_step_s': 5., 'peak_mem_gb': 2., 'total_s': 40.}, f)
        if c['name'] in ('k4_cm0_-_og0_s0', 'k4_cm0_-_og0.1_s0', 'k0_cm2_topk_og0_s0', 'k0_cm2_topk_og0.1_s0', 'k0_cm0_-_og0_s0'):
            cfg = {**c, **dims}
            torch.save({'cfg': cfg, 'sd': LM(**cfg).state_dict()}, f'{run_dir}/ckpt.pt')
            torch.save({0.0: torch.randn(16, dims['d']), 0.25: torch.randn(16, dims['d'])}, f'{run_dir}/sae_decoders.pt')


if __name__ == '__main__':
    main(*sys.argv[1:])
