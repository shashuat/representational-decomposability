"""Report tables from the real run artifacts: writes LaTeX row fragments to ../report/generated/*.tex (one per table body) and nseeds.tex.

  python report_tables.py [--runs DIR] [--results DIR] [--out DIR]     (CPU only; safe while the GPU is busy)

Defaults: runs = paths.RUNS, results = paths.RESULTS (reserved for text aggregates; the tables only read runs), out = ../report/generated.

Only seeds for which every grid cell has an eval JSON ("complete seeds") enter the tables. Cells are mean$\\pm$std over those seeds
(bare value when n=1). Re-run after the grid grows."""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import paths
import plots
from sae import train_sae

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'report', 'generated')
BOLD_OK = 'ce ppl acc concept_ce concept_hit gender_best_reduction ev ce_recovered kl_score dead probe_f1 absorption gender_kl_best'.split()  # direction unambiguous
LOWER = 'ce ppl concept_ce dead absorption gender_kl_best'.split()
TEACHER = {'-': '---', 'topk': 'TopK', 'ort': 'OrtSAE'}
MODE = {0: 'none', 1: 'CE only', 2: 'CE + interleave', 3: 'interleave only'}


class Data:
    """Per-(cell, seed) metric lookup restricted to complete seeds."""

    def __init__(self, runs):
        df = plots.load(runs)
        if df.empty:  # no evals yet: every cell renders as '---'
            df = pd.DataFrame({'cell': pd.Series(dtype=str), 'seed': pd.Series(dtype=int)})
        n_cells = df.cell.nunique()
        self.seeds = sorted(s for s, g in df.groupby('seed') if g.cell.nunique() == n_cells)
        self.df = df[df.seed.isin(self.seeds)].set_index(['cell', 'seed'], drop=False).sort_index()
        self.all = df  # every run present (timing table)

    def row(self, cell, seed):
        return self.df.loc[(cell, seed)]

    def vals(self, cell, get):
        """One value per complete seed (NaN-free); get: column name or function(row) -> float."""
        out = []
        for s in self.seeds:
            if (cell, s) not in self.df.index:
                continue
            r = self.row(cell, s)
            v = r.get(get, np.nan) if isinstance(get, str) else get(r)
            if v is not None and np.isfinite(v):
                out.append(float(v))
        return out


def fmt(vals, p=3, sci=False):
    if not vals:
        return '---'
    m, s = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else None
    f = (lambda x: f'{x:.{p}g}') if sci else (lambda x: f'{x:.{p}f}')
    return f(m) if s is None else f'{f(m)}$\\pm${f(s)}'


def cell_text(d, cell, get, key=None, p=3, sci=False):
    return fmt(d.vals(cell, get), p, sci)


def bold_best(rows, cols):
    """rows: list of lists of (text, mean|None, key|None); bold the best mean per column among columns whose key has a clear direction."""
    for c in cols:
        cand = [(r[c][1], i) for i, r in enumerate(rows) if r[c][1] is not None and r[c][2] in BOLD_OK]
        if len(cand) > 1:
            lo = rows[cand[0][1]][c][2] in LOWER
            best = (min if lo else max)(cand)[1]
            t, m, k = rows[best][c]
            rows[best][c] = (f'\\textbf{{{t}}}', m, k)


def cell_item(d, cell, get, key=None, p=3):
    v = d.vals(cell, get)
    return fmt(v, p), (float(np.mean(v)) if v else None), key


def params(d, cell):
    v = d.vals(cell, 'n_params')
    return f'{int(v[0]):,}'.replace(',', '{,}') if v else '---'


def gender_stats(r):
    """(mean reduction over senses, KL of the best sense) from the stored per-sense lists."""
    g = r['raw']
    if 'gender_bias_zero_sense' not in g:
        return np.nan, np.nan
    z, kl, b = np.array(g['gender_bias_zero_sense']), np.array(g['gender_kl_zero_sense']), max(g['gender_bias'], 1e-9)
    return float((1 - z / b).mean()), float(kl[z.argmin()])


def write(outdir, name, rows):
    os.makedirs(outdir, exist_ok=True)
    with open(f'{outdir}/{name}.tex', 'w') as f:
        f.write('\n'.join(rows) + '\n')


def tab_lm(d):
    spec = [('Baseline', 'k0_cm0_-_og0'), ('+ Backpack ($k{=}4$)', 'k4_cm0_-_og0'), ('+ concept CE only (TopK teacher)', 'k0_cm1_topk_og0'),
            ('+ CE + interleave (TopK teacher)', 'k0_cm2_topk_og0'), ('+ interleave only', 'k0_cm3_-_og0'),
            ('+ CE + interleave, OrtSAE teacher', 'k0_cm2_ort_og0'), ('Backpack + CE + interleave (TopK teacher)', 'k4_cm2_topk_og0'),
            ('Backpack, $\\gamma_o{=}0.1$', 'k4_cm0_-_og0.1'), ('Width-matched ($d{=}136$, Backpack)', 'k0_cm0_wm136_og0'),
            ('Width-matched ($d{=}152$, Backpack + CoCoMix)', 'k0_cm0_wm152_og0')]
    rows = []
    for name, cell in spec:
        delta = [d.row(cell, s).ce - d.row('k0_cm0_-_og0', s).ce for s in d.seeds if (cell, s) in d.df.index]
        rows.append([(name, None, None), (params(d, cell), None, None), cell_item(d, cell, 'ce', 'ce'), cell_item(d, cell, 'ppl', 'ppl', 2),
                     cell_item(d, cell, 'acc', 'acc'), ('---' if cell == 'k0_cm0_-_og0' else fmt(delta, 3), None, None)])
    bold_best(rows, [2, 3, 4])
    return [' & '.join(t for t, _, _ in r) + ' \\\\' for r in rows]


def tab_bp(d):
    rows = []
    for name, cm_cell in (('none', 'k4_cm0_-_og0'), ('none', 'k4_cm0_-_og0.1'), ('CE + interleave', 'k4_cm2_topk_og0'), ('CE + interleave', 'k4_cm2_topk_og0.1')):
        rows.append(bp_row(d, cm_cell, [(name, None, None), ('0.1' if cm_cell.endswith('0.1') else '0', None, None)]))
    bold_best(rows, [2, 6, 8])
    return [' & '.join(t for t, _, _ in r) + ' \\\\' for r in rows]


def bp_row(d, cell, head):
    return head + [cell_item(d, cell, 'ce', 'ce'), cell_item(d, cell, 'sense_cos', 'sense_cos'), cell_item(d, cell, 'sense_maxcos2', 'sense_maxcos2'),
                   cell_item(d, cell, 'gender_bias', 'gender_bias'), cell_item(d, cell, 'gender_best_reduction', 'gender_best_reduction'),
                   cell_item(d, cell, lambda r: gender_stats(r)[0], 'gender_mean_reduction'),
                   cell_item(d, cell, lambda r: gender_stats(r)[1], 'gender_kl_best')]


def random_floor(m=1024, d=128, draws=5):
    """Mean max-cos and Meta-SAE EV of m random unit vectors in R^d (same Meta-SAE as evaluate.sae_eval); CPU."""
    mc, ev = [], []
    for s in range(draws):
        torch.manual_seed(100 + s)
        W = F.normalize(torch.randn(m, d), dim=-1)
        mc.append((W @ W.T).fill_diagonal_(-1).max(1).values.mean().item())
        meta = train_sae(W, 256, 4, 0., 1000, 1024)
        with torch.no_grad():
            ev.append(1 - ((meta(W)[0] - W) ** 2).sum().item() / ((W - W.mean(0)) ** 2).sum().item())
    return mc, ev


SAE_COLS = [('ev', 'ev', 3), ('l0', 'l0', 1), ('dead', 'dead', 3), ('ce_recovered', 'ce_recovered', 3), ('kl_score', 'kl_score', 3),
            ('mean_max_cos', 'mean_max_cos', 3), ('meta_ev', 'meta_ev', 3), ('probe_f1', 'probe_f1', 3), ('absorption', 'absorption', 3)]


def tab_sae(d):
    fam = [('Baseline', 'k0_cm0_-_og0'), ('Backpack', 'k4_cm0_-_og0'), ('CE + interleave', 'k0_cm2_topk_og0'), ('Backpack + CE + interleave', 'k4_cm2_topk_og0')]
    rows = []
    for name, cell in fam:
        for g, gname in (('0.0', 'TopK'), ('0.25', 'OrtSAE')):
            rows.append([(name if g == '0.0' else '', None, None), (gname, None, None)] +
                        [cell_item(d, cell, f'sae{g}_{k}', key, p) for k, key, p in SAE_COLS])
    bold_best(rows, list(range(2, 11)))
    out = [' & '.join(t for t, _, _ in r) + ' \\\\' for r in rows]
    mc, ev = random_floor()
    out.append('\\multicolumn{2}{@{}l}{Random-direction floor} & --- & --- & --- & --- & --- & ' + fmt(mc, 3) + ' & ' + fmt(ev, 3) + ' & --- & --- \\\\')
    return out


def full_rows():
    """(k, mode, teacher, og) in the order of the appendix tables."""
    r = [(0, 0, '-', 0)] + [(0, 1, t, 0) for t in ('topk', 'ort')] + [(0, 2, t, og) for t in ('topk', 'ort') for og in (0, .1)] + [(0, 3, '-', og) for og in (0, .1)]
    r += [(4, 0, '-', og) for og in (0, .1)] + [(4, 1, t, og) for t in ('topk', 'ort') for og in (0, .1)]
    r += [(4, 2, t, og) for t in ('topk', 'ort') for og in (0, .1)] + [(4, 3, '-', og) for og in (0, .1)]
    return r


def head(k, cm, t, og):
    return [(str(k), None, None), (MODE[cm], None, None), (TEACHER[t], None, None), (f'{og:g}', None, None)]


def tab_full_lm(d):
    rows = []
    for k, cm, t, og in full_rows():
        cell = plots.cell_name(k, cm, t, og)
        steer = ('n/a', None, None) if cm == 1 else cell_item(d, cell, 'steer_kl', 'steer_kl')
        rows.append(head(k, cm, t, og) + [(params(d, cell), None, None), cell_item(d, cell, 'ce', 'ce'), cell_item(d, cell, 'ppl', 'ppl', 2),
                                          cell_item(d, cell, 'acc', 'acc'), cell_item(d, cell, 'concept_ce', 'concept_ce'),
                                          cell_item(d, cell, 'concept_hit', 'concept_hit'), steer])
    bold_best(rows, [5, 6, 7])
    out = [' & '.join(t for t, _, _ in r) + ' \\\\' for r in rows]
    for name, cell in (('plain, width-matched to $k{=}4$', 'k0_cm0_wm136_og0'), ('plain, width-matched to $k{=}4$ + CoCoMix', 'k0_cm0_wm152_og0')):
        out.append(f'\\multicolumn{{4}}{{@{{}}l}}{{{name}}} & {params(d, cell)} & {cell_text(d, cell, "ce")} & {cell_text(d, cell, "ppl", p=2)} & '
                   f'{cell_text(d, cell, "acc")} & --- & --- & --- \\\\')
    return out


def tab_full_bp(d):
    rows = []
    for k, cm, t, og in full_rows():
        if k == 4:
            cell = plots.cell_name(k, cm, t, og)
            r = bp_row(d, cell, [(MODE[cm], None, None), (TEACHER[t], None, None), (f'{og:g}', None, None)])
            rows.append(r[:3] + r[4:])  # drop the CE column (it is in the LM table)
    bold_best(rows, [6, 8])
    return [' & '.join(t for t, _, _ in r) + ' \\\\' for r in rows]


def label_freq(runs, teacher):
    """Label-frequency baseline: concept CE of the train-label marginal and precision@Ka of always predicting the Ka most frequent train labels, on test."""
    if not all(os.path.exists(f'{runs}/teacher/{teacher}/{s}.pt') for s in ('train', 'test')):
        return None
    tr, te = (torch.load(f'{runs}/teacher/{teacher}/{s}.pt').long() for s in ('train', 'test'))
    C = int(max(tr.max(), te.max())) + 1
    f = (tr.flatten().bincount(minlength=C) + 1).float()
    f /= f.sum()
    ce = -f.log()[te].mean().item()
    top = f.topk(te.shape[-1]).indices
    hit = (top[None, None, :, None] == te[..., None, :]).any(-1).float().mean().item()
    return ce, hit


def tab_concept(d, runs):
    out = []
    for t in ('topk', 'ort'):
        ce, hit = label_freq(runs, t) or (np.nan, np.nan)
        out.append(f'\\multicolumn{{3}}{{@{{}}l}}{{Label-frequency baseline ({TEACHER[t]} labels)}} & {ce:.3f} & {hit:.3f} & --- & --- \\\\'.replace('nan', '---'))
    for k, cm, t in ((0, 1, 'topk'), (0, 2, 'topk'), (0, 2, 'ort'), (4, 2, 'topk'), (4, 2, 'ort')):
        cell = plots.cell_name(k, cm, t, 0)
        steer = 'n/a' if cm == 1 else cell_text(d, cell, 'steer_kl')
        out.append(f'{k} & {MODE[cm]} & {TEACHER[t]} & {cell_text(d, cell, "concept_ce")} & {cell_text(d, cell, "concept_hit")} & {steer} & '
                   f'{cell_text(d, cell, "ce")} \\\\')
    return out


def tab_full_sae(d):
    rows = []
    for k, cm, t, og in full_rows():
        cell = plots.cell_name(k, cm, t, og)
        r = head(k, cm, t, og)
        for g in ('0.0', '0.25'):
            r += [cell_item(d, cell, f'sae{g}_{m}', m) for m in ('ev', 'ce_recovered', 'mean_max_cos', 'absorption')]
        rows.append(r)
    for cols in ([4, 5, 7], [8, 9, 11]):  # bold within each SAE block, not across the two
        bold_best(rows, cols)
    return [' & '.join(t for t, _, _ in r) + ' \\\\' for r in rows]


def tab_circ(d):
    recs = []
    for _, r in d.df.iterrows():
        for g in ('0.0', '0.25'):
            recs.append({'g': g, **{k: r.get(f'sae{g}_{k}', np.nan) for k in ('mean_max_cos', 'absorption', 'probe_f1', 'ce_recovered', 'meta_ev')}})
    x = pd.DataFrame(recs, columns=['g', 'mean_max_cos', 'absorption', 'probe_f1', 'ce_recovered', 'meta_ev'])
    out = []
    for name, a, b in (('mean max-cos vs.\\ absorption proxy', 'mean_max_cos', 'absorption'), ('mean max-cos vs.\\ letter-probe F1', 'mean_max_cos', 'probe_f1'),
                       ('mean max-cos vs.\\ CE recovered', 'mean_max_cos', 'ce_recovered'), ('Meta-SAE EV vs.\\ absorption proxy', 'meta_ev', 'absorption')):
        cells = [x[a].corr(x[b], method='spearman')] + [x[x.g == g][a].corr(x[x.g == g][b], method='spearman') for g in ('0.0', '0.25')]
        out.append(f'{name} & ' + ' & '.join('---' if np.isnan(c) else f'{c:+.2f}' for c in cells) + ' \\\\')
    return out, len(x)


def tab_timing(d):
    """Whole tabular (header included) so the columns can follow what was logged: 'steps run' / 'best step' appear when early stopping logged them."""
    t = d.all
    cols = {'t_train_steps_run': 'steps run', 't_train_best_step': 'best step', 't_train_train_s': 'train s', 't_train_ms_per_step': 'ms/step',
            't_train_tok_s': 'Mtok/s', 't_train_first_step_s': 'compile s', 't_eval_total_s': 'eval s'}
    have = [c for c in cols if c in t.columns]
    head = '\\begin{tabular}{@{}lr' + 'r' * len(have) + '@{}}\\toprule\ncell & runs & ' + ' & '.join(cols[c] for c in have) + ' \\\\\\midrule'
    if not have or t.empty:
        return [head, f'\\multicolumn{{{len(have) + 2}}}{{c}}{{no runs yet}} \\\\', '\\bottomrule\\end{tabular}']
    g = t.groupby('cell')[have].mean().reindex(plots.order(t.cell.unique()))
    n = t.groupby('cell').size()
    scale = {'t_train_tok_s': 1e6}
    ints = ('t_train_steps_run', 't_train_best_step', 't_train_train_s', 't_eval_total_s')
    num = lambda c, v: '---' if np.isnan(v) else f'{v:.0f}' if c in ints else f'{v / scale[c]:.2f}' if c in scale else f'{v:.1f}'
    return [head] + [f'{plots.label(c)} & {n[c]} & ' + ' & '.join(num(k, r[k]) for k in have) + ' \\\\' for c, r in g.iterrows()] + ['\\bottomrule\\end{tabular}']


def main(runs=paths.RUNS, outdir=OUTDIR, results=paths.RESULTS):
    d = Data(runs)
    os.makedirs(outdir, exist_ok=True)
    seeds = ','.join(map(str, d.seeds))
    with open(f'{outdir}/nseeds.tex', 'w') as f:
        f.write(f'\\newcommand{{\\NSEEDS}}{{{len(d.seeds)}}}\n\\newcommand{{\\SEEDLIST}}{{{seeds or "none yet"}}}\n')
    circ, n_circ = tab_circ(d)
    for name, rows in (('tab_lm', tab_lm(d)), ('tab_bp', tab_bp(d)), ('tab_sae', tab_sae(d)), ('tab_full_lm', tab_full_lm(d)), ('tab_full_bp', tab_full_bp(d)),
                       ('tab_concept', tab_concept(d, runs)), ('tab_full_sae', tab_full_sae(d)), ('tab_circ', circ), ('tab_timing', tab_timing(d))):
        write(outdir, name, rows)
    with open(f'{outdir}/circ_n.tex', 'w') as f:
        f.write(f'\\newcommand{{\\CIRCN}}{{{n_circ}}}\n')
    print(f'runs={runs}  complete seeds {d.seeds}; tables -> {outdir}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default=paths.RUNS)
    p.add_argument('--results', default=paths.RESULTS)
    p.add_argument('--out', default=OUTDIR)
    a = p.parse_args()
    main(a.runs, a.out, a.results)
