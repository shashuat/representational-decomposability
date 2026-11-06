"""Report tables and facts from the two grids: 30k steps, seed 0 (the headline numbers) and 10k steps, 5 seeds (statistical support for the claims).

  python report_data.py [--main RUNS_30K] [--support RUNS_10K] [--out REPORT/generated] [--md RESULTS_DIR]     (CPU only)

Writes complete `tabular` bodies to report/generated/*.tex (\\input them), macros to nseeds.tex, and the text files REPORT_CLAIMS.md / REPORT_FACTS.md.
A claim is 'supported' when its sign at 30k (seed 0) matches the 10k mean and the 10k 95% t-interval over seeds excludes 0; null claims need
both budgets compatible with no effect. Pairing across seeds is by seed index only (crude), so intervals are indicative."""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import t as tdist

import paths
import plots
import report_tables as rt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 'k0_cm0_-_og0'
MODE = ['', 'concept CE', 'CE + interleave', 'interleave only']


def tex_label(cell):
    q = plots.parse(cell + '_s0')
    if q['teacher'].startswith('wm'):
        return f"Baseline $d{{=}}{q['teacher'][2:]}$"
    parts = ['Backpack $k{=}4$'] * bool(q['k']) + [MODE[q['cm']] + (f" ({'OrtSAE' if q['teacher'] == 'ort' else 'TopK'})" if q['cm'] in (1, 2) else '')] * bool(q['cm'])
    return ' + '.join(parts + ['ortho'] * bool(q['og'])) or 'Baseline'


class Grid:
    """One grid: per-run frame with derived columns, pivot access by (seed, cell)."""

    def __init__(self, runs):
        df = plots.load(runs)
        g = df.raw.map(lambda raw: rt.gender_stats({'raw': raw}))
        df['gender_mean_red'], df['gender_kl_best'] = [x[0] for x in g], [x[1] for x in g]
        for k in ('ev', 'absorption', 'mean_max_cos', 'ce_recovered', 'probe_f1', 'meta_ev'):  # within-run OrtSAE minus TopK
            df[f'd_{k}'] = df[f'sae0.25_{k}'] - df[f'sae0.0_{k}']
        self.df, self.runs = df, runs
        self.seeds = sorted(df.seed.unique())
        self.cells = plots.order(df.cell.unique())

    def P(self, col):
        return self.df.pivot(index='seed', columns='cell', values=col)

    def one(self, cell, col, seed=0):
        v = self.df[(self.df.cell == cell) & (self.df.seed == seed)][col]
        return float(v.iloc[0]) if len(v) and pd.notna(v.iloc[0]) else np.nan


def ci(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


t = tdist


def pm(g, cell, col, p=3):
    v = g.P(col)[cell].dropna().values if cell in g.P(col) else []
    if not len(v):
        return '---'
    return f'{v.mean():.{p}f}' + (f'$\\pm${v.std(ddof=1):.{p}f}' if len(v) > 1 else '')


def f(x, p=3, plus=False):
    return '---' if x is None or np.isnan(x) else (f'{x:+.{p}f}' if plus else f'{x:.{p}f}')


def npar(x):
    return '---' if np.isnan(x) else f'{int(x):,}'.replace(',', '{,}')


def tab(header, rows, cols):
    return '\\begin{tabular}{@{}' + cols + '@{}}\\toprule\n' + header + ' \\\\\\midrule\n' + ' \\\\\n'.join(rows) + ' \\\\\\bottomrule\n\\end{tabular}\n'


PAIRS = {  # name -> partner(q) giving the cell an effect is measured against (None: not part of this contrast), as in plots.factor_effects
    'Backpack $k{=}4$ vs $k{=}0$': lambda q: plots.cell_name(0, q['cm'], q['teacher'], q['og']) if q['k'] else None,
    'concept CE only vs none': lambda q: plots.cell_name(q['k'], 0, '-', q['og']) if q['cm'] == 1 else None,
    'CE + interleave vs CE only': lambda q: plots.cell_name(q['k'], 1, q['teacher'], q['og']) if q['cm'] == 2 else None,
    'interleave only vs none': lambda q: plots.cell_name(q['k'], 0, '-', q['og']) if q['cm'] == 3 else None,
    'OrtSAE vs TopK teacher': lambda q: plots.cell_name(q['k'], q['cm'], 'topk', q['og']) if q['teacher'] == 'ort' else None,
    'ortho penalty $\\gamma_o{=}0.1$ vs 0': lambda q: plots.cell_name(q['k'], q['cm'], q['teacher'], 0) if q['og'] else None,
}


def contrast(g, name, col):
    """Per-seed matched-pair effect (mean over the cell pairs of the contrast) for a column; returns a Series indexed by seed and the number of pairs."""
    P = g.P(col)
    cells = [c for c in P.columns if not plots.parse(c + '_s0')['teacher'].startswith('wm')]
    pairs = [(c, PAIRS[name](plots.parse(c + '_s0'))) for c in cells]
    pairs = [(a, b) for a, b in pairs if b in P.columns]
    return pd.concat([P[a] - P[b] for a, b in pairs], axis=1).mean(axis=1), len(pairs)


def width_contrast(g, cell, col):
    P = g.P(col)
    return P[cell] - P[BASE]


def verdict(v30, m10, h10, sd10, kind):
    if np.isnan(v30) or np.isnan(m10):
        return 'n/a'
    if kind == 'null':
        if abs(m10) <= h10 and abs(v30) <= 2 * sd10:
            return 'no effect at either budget'
        return 'small, same sign at both budgets' if np.sign(v30) == np.sign(m10) and abs(m10) > h10 else 'effect at one budget only'
    same, sig = np.sign(v30) == np.sign(m10), abs(m10) > h10
    return 'supported' if same and sig else ('same sign, 10k interval includes 0' if same else 'signs differ')


def claims(g30, g10):
    """Rows: (group, claim, contrast name or callable, column, scale, kind, digits)."""
    rows = []
    cm = lambda name, col, label, kind='effect', grp='LM quality': (grp, label, name, col, kind)  # noqa: E731
    for name, kind in (('Backpack $k{=}4$ vs $k{=}0$', 'effect'), ('concept CE only vs none', 'null'), ('CE + interleave vs CE only', 'effect'),
                       ('interleave only vs none', 'effect'), ('OrtSAE vs TopK teacher', 'null'), ('ortho penalty $\\gamma_o{=}0.1$ vs 0', 'null')):
        rows.append(cm(name, 'ce', name + ': test CE', kind))
    rows.append(('LM quality', 'Wider plain model $d{=}136$ vs $d{=}128$: test CE', ('w', 'k0_cm0_wm136_og0'), 'ce', 'effect'))
    rows.append(('LM quality', 'Wider plain model $d{=}152$ vs $d{=}128$: test CE', ('w', 'k0_cm0_wm152_og0'), 'ce', 'effect'))
    rows.append(('LM quality', 'Interaction interleave $\\times$ Backpack (TopK): test CE', ('did', 'k4_cm2_topk_og0', 'k4_cm1_topk_og0', 'k0_cm2_topk_og0', 'k0_cm1_topk_og0'), 'ce', 'effect'))
    rows.append(('Concept interface', 'CE + interleave vs CE only: concept hit', 'CE + interleave vs CE only', 'concept_hit', 'effect'))
    rows.append(('Concept interface', 'CE + interleave vs CE only: steering KL', 'CE + interleave vs CE only', 'steer_kl', 'effect'))
    rows.append(('Concept interface', 'OrtSAE vs TopK teacher: concept hit', 'OrtSAE vs TopK teacher', 'concept_hit', 'null'))
    rows.append(('Concept interface', 'OrtSAE vs TopK teacher: steering KL', 'OrtSAE vs TopK teacher', 'steer_kl', 'null'))
    rows.append(('Backpack interface', 'ortho penalty vs none: best-sense gender reduction', 'ortho penalty $\\gamma_o{=}0.1$ vs 0', 'gender_best_reduction', 'null'))
    rows.append(('Backpack interface', 'ortho penalty vs none: sense max-cos$^2$', 'ortho penalty $\\gamma_o{=}0.1$ vs 0', 'sense_maxcos2', 'effect'))
    for k, lab, kind in (('ev', 'explained variance', 'effect'), ('absorption', 'absorption proxy', 'effect'), ('ce_recovered', 'CE recovered', 'effect'),
                         ('probe_f1', 'letter-probe F1', 'effect'), ('mean_max_cos', 'mean decoder max-cosine', 'effect'), ('meta_ev', 'Meta-SAE EV', 'effect')):
        rows.append(('Post-hoc SAE', f'OrtSAE ($\\gamma{{=}}0.25$) vs TopK SAE: {lab}', ('run', f'd_{k}'), f'd_{k}', kind))
    out = []
    for grp, lab, how, col, kind in rows:
        if isinstance(how, str):
            s30, n30 = contrast(g30, how, col)
            s10, n10 = contrast(g10, how, col)
        elif how[0] == 'w':
            s30, s10, n30 = width_contrast(g30, how[1], col), width_contrast(g10, how[1], col), 1
        elif how[0] == 'did':  # difference of differences (a - b) - (c - d): interaction of two switches
            did = lambda g: g.P(col)[how[1]] - g.P(col)[how[2]] - g.P(col)[how[3]] + g.P(col)[how[4]]  # noqa: E731
            s30, s10, n30 = did(g30), did(g10), 1
        else:
            s30, s10, n30 = g30.df.groupby('seed')[col].mean(), g10.df.groupby('seed')[col].mean(), 23
        v30, m10, h10, sd10 = float(s30.iloc[0]), float(s10.mean()), ci(s10), float(s10.std(ddof=1))
        out.append({'group': grp, 'claim': lab, 'col': col, 'n': n30, 'v30': v30, 'm10': m10, 'h10': h10, 'sd10': sd10, 'kind': kind,
                    'verdict': verdict(v30, m10, h10, sd10, kind), 's10': list(map(float, s10.values))})
    return out


def tab_claims(cl):
    rows = []
    for c in cl:
        p = 3 if c['col'] not in ('ce',) else 4
        rows.append(f"{c['claim']} & {f(c['v30'], p, True)} & {f(c['m10'], p, True)} [{f(c['m10'] - c['h10'], p, True)}, {f(c['m10'] + c['h10'], p, True)}] & {c['verdict']}")
    return tab('Claim (A vs B; effect = A $-$ B) & 30k, seed 0 & 10k, mean over 5 seeds [95\\% interval] & verdict', rows, 'lrrl')


def families():
    return {'concept CE only': ['k0_cm1_topk_og0', 'k0_cm1_ort_og0'], 'CE + interleave (CoCoMix)': [f'k0_cm2_{t}_og{o}' for t in ('topk', 'ort') for o in (0, 0.1)],
            'interleave only': ['k0_cm3_-_og0', 'k0_cm3_-_og0.1'], 'Backpack $k{=}4$': ['k4_cm0_-_og0', 'k4_cm0_-_og0.1'],
            'Backpack + concept CE': [f'k4_cm1_{t}_og{o}' for t in ('topk', 'ort') for o in (0, 0.1)],
            'Backpack + CE + interleave': [f'k4_cm2_{t}_og{o}' for t in ('topk', 'ort') for o in (0, 0.1)],
            'Backpack + interleave only': ['k4_cm3_-_og0', 'k4_cm3_-_og0.1'],
            'plain $d{=}136$ / $d{=}152$': ['k0_cm0_wm136_og0', 'k0_cm0_wm152_og0']}


def family_gap(g, cells, ppl='ppl'):
    P = g.P(ppl)
    return (P[cells].mean(axis=1) - P[BASE])


def tab_budget(g30, g10):
    rows = []
    for name, cs in families().items():
        a, b = family_gap(g10, cs), family_gap(g30, cs)
        rows.append(f"{name} & {len(cs)} & {f(float(a.iloc[0]), 2, True)} & {f(a.mean(), 2, True)}$\\pm${a.std(ddof=1):.2f} & {f(float(b.iloc[0]), 2, True)}")
    return tab('Model family (mean over its cells) & cells & 10k, seed 0 & 10k, 5 seeds & 30k, seed 0', rows, 'lrrrr'), {n: (family_gap(g10, cs), family_gap(g30, cs)) for n, cs in families().items()}


LM_ROWS = ['k0_cm0_-_og0', 'k4_cm0_-_og0', 'k0_cm1_topk_og0', 'k0_cm2_topk_og0', 'k0_cm3_-_og0', 'k0_cm2_ort_og0', 'k4_cm2_topk_og0', 'k4_cm0_-_og0.1',
           'k0_cm0_wm136_og0', 'k0_cm0_wm152_og0']


def tab_lm(g30, g10):
    rows = []
    for c in LM_ROWS:
        d30, d10 = g30.one(c, 'ppl') - g30.one(BASE, 'ppl'), (g10.P('ppl')[c] - g10.P('ppl')[BASE]).values
        rows.append(f"{tex_label(c)} & {npar(g30.one(c, 'n_params'))} & {f(g30.one(c, 'ce'))} & {f(g30.one(c, 'ppl'), 2)} & {f(g30.one(c, 'acc'))} & "
                    f"{f(d30, 2, True) if c != BASE else '---'} & {pm(g10, c, 'ppl', 2)} & "
                    + (f"{f(d10.mean(), 2, True)} [{f(d10.mean() - ci(d10), 2, True)}, {f(d10.mean() + ci(d10), 2, True)}]" if c != BASE else '---'))
    return tab('Model & Params & CE & PPL & Acc & $\\Delta$PPL & PPL (5 seeds) & $\\Delta$PPL (5 seeds) [95\\%]', rows, 'lrrrrrrr')


def tab_full_lm(g30, g10):
    rows = []
    for c in g30.cells:
        rows.append(f"{tex_label(c)} & {npar(g30.one(c, 'n_params'))} & {f(g30.one(c, 'ce'))} & {f(g30.one(c, 'ppl'), 2)} & {f(g30.one(c, 'acc'))} & {pm(g10, c, 'ppl', 2)} & "
                    f"{f(g30.one(c, 'concept_hit'))} & {f(g30.one(c, 'steer_kl'))} & {pm(g10, c, 'concept_hit')} & {pm(g10, c, 'steer_kl')}")
    return tab('Cell & Params & CE & PPL & Acc & PPL (5 seeds) & hit@4 & steer KL & hit@4 (5 seeds) & steer KL (5 seeds)', rows, 'lrrrrrrrrr')


def tab_bp(g30, g10):
    rows = []
    for c in [c for c in g30.cells if plots.parse(c + '_s0')['k'] and plots.parse(c + '_s0')['cm'] in (0, 2) and plots.parse(c + '_s0')['teacher'] in ('-', 'topk')]:
        rows.append(f"{tex_label(c)} & {f(g30.one(c, 'ce'))} & {f(g30.one(c, 'sense_cos'), 2)} & {f(g30.one(c, 'sense_maxcos2'), 2)} & {f(g30.one(c, 'gender_bias'), 2)} & "
                    f"{f(g30.one(c, 'gender_best_reduction'), 2)} & {f(g30.one(c, 'gender_mean_red'), 2)} & {f(g30.one(c, 'gender_kl_best'), 3)} & {pm(g10, c, 'gender_best_reduction', 2)} & {pm(g10, c, 'gender_mean_red', 2)}")
    return tab('Cell & CE & sense cos & sense max-cos$^2$ & bias $b_0$ & best red. & mean red. & KL best & best red. (5 seeds) & mean red. (5 seeds)', rows, 'lrrrrrrrrr')


def tab_full_bp(g30, g10):
    rows = [f"{tex_label(c)} & {f(g30.one(c, 'gender_best_reduction'), 2)} & {f(g30.one(c, 'gender_mean_red'), 2)} & {f(g30.one(c, 'gender_kl_best'), 3)} & {f(g30.one(c, 'sense_maxcos2'), 2)} & "
            f"{pm(g10, c, 'gender_best_reduction', 2)} & {pm(g10, c, 'sense_maxcos2', 2)}" for c in g30.cells if plots.parse(c + '_s0')['k']]
    return tab('Cell & best red. & mean red. & KL best & max-cos$^2$ & best red. (5 seeds) & max-cos$^2$ (5 seeds)', rows, 'lrrrrrr')


def tab_concept(g30, g10, runs):
    rows = []
    for c in [c for c in g30.cells if plots.parse(c + '_s0')['cm'] in (1, 2) and plots.parse(c + '_s0')['og'] == 0]:
        rows.append(f"{tex_label(c)} & {f(g30.one(c, 'concept_ce'))} & {f(g30.one(c, 'concept_hit'))} & {f(g30.one(c, 'steer_kl'))} & {pm(g10, c, 'concept_hit')} & {pm(g10, c, 'steer_kl')}")
    lf = {t: rt.label_freq(runs, t) for t in ('topk', 'ort')}
    base = [f"Label-frequency baseline ({'TopK' if t == 'topk' else 'OrtSAE'} labels) & {f(v[0])} & {f(v[1])} & --- & --- & ---" for t, v in lf.items() if v]
    return tab('Cell & concept CE & hit@4 & steer KL & hit@4 (5 seeds) & steer KL (5 seeds)', base + rows, 'lrrrrr'), lf


SAE_M = ['ev', 'l0', 'dead', 'ce_recovered', 'kl_score', 'mean_max_cos', 'meta_ev', 'probe_f1', 'absorption']


def tab_sae(g30):
    rows = []
    for c in ('k0_cm0_-_og0', 'k4_cm0_-_og0', 'k0_cm2_topk_og0', 'k4_cm2_topk_og0'):
        for gg, n in (('0.0', 'TopK'), ('0.25', 'OrtSAE')):
            rows.append(f"{tex_label(c)} & {n} & " + ' & '.join(f(g30.one(c, f'sae{gg}_{m}'), 1 if m == 'l0' else 3) for m in SAE_M))
    mc, ev = rt.random_floor()
    rows.append(f"\\multicolumn{{2}}{{@{{}}l}}{{Random-direction floor}} & --- & --- & --- & --- & --- & {np.mean(mc):.3f} & {np.mean(ev):.3f} & --- & ---")
    return tab('Model & SAE & EV & L0 & dead & CE rec. & KL score & max-cos & Meta-EV & probe F1 & absorption', rows, 'llrrrrrrrrr'), (np.mean(mc), np.mean(ev))


def tab_full_sae(g30, g10):
    rows = []
    for c in g30.cells:
        rows.append(f"{tex_label(c)} & " + ' & '.join(f(g30.one(c, f'sae{gg}_{m}')) for gg in ('0.0', '0.25') for m in ('ev', 'absorption', 'mean_max_cos', 'ce_recovered'))
                    + f" & {pm(g10, c, 'sae0.0_absorption')} & {pm(g10, c, 'sae0.25_absorption')}")
    return tab('Cell & EV & abs. & max-cos & CE rec. (TopK) & EV & abs. & max-cos & CE rec. (OrtSAE) & abs. TopK (5 seeds) & abs. OrtSAE (5 seeds)', rows, 'lrrrrrrrrrr')


def tab_circ(g10):
    recs = [{'g': gg, **{k: r.get(f'sae{gg}_{k}', np.nan) for k in ('mean_max_cos', 'absorption', 'probe_f1', 'ce_recovered', 'meta_ev')}} for _, r in g10.df.iterrows() for gg in ('0.0', '0.25')]
    x = pd.DataFrame(recs)
    rows = []
    for name, a, b in (('mean max-cos vs.\\ absorption proxy', 'mean_max_cos', 'absorption'), ('mean max-cos vs.\\ letter-probe F1', 'mean_max_cos', 'probe_f1'),
                       ('mean max-cos vs.\\ CE recovered', 'mean_max_cos', 'ce_recovered'), ('Meta-SAE EV vs.\\ absorption proxy', 'meta_ev', 'absorption')):
        cs = [x[a].corr(x[b], method='spearman')] + [x[x.g == gg][a].corr(x[x.g == gg][b], method='spearman') for gg in ('0.0', '0.25')]
        rows.append(f'{name} & ' + ' & '.join('---' if np.isnan(c) else f'{c:+.2f}' for c in cs))
    return tab('Pair & all SAEs & TopK & OrtSAE', rows, 'lrrr'), len(x)


def tab_timing(g30, g10):
    rows = [f"{tex_label(c)} & {f(g30.one(c, 't_train_train_s'), 0)} & {f(g30.one(c, 't_train_ms_per_step'), 1)} & {f(g30.one(c, 't_train_tok_s') / 1e6, 2)} & {f(g30.one(c, 't_eval_total_s'), 0)} & "
            f"{pm(g10, c, 't_train_train_s', 0)} & {pm(g10, c, 't_train_ms_per_step', 1)}" for c in g30.cells]
    return tab('Cell & train s (30k) & ms/step & Mtok/s & eval s & train s (10k, 5 seeds) & ms/step (10k)', rows, 'lrrrrrr')


def write(out, name, text):
    with open(os.path.join(out, name), 'w') as fh:
        fh.write(text)


def md_claims(cl, budget, lf, floor, n10, g30, g10):
    L = ['# Claims: 30k seed 0 (headline) with 10k 5-seed significance\n',
         'Effect = A minus B on the test split, matched-pair contrasts (mean over the cell pairs differing only in that factor). 30k: one seed. 10k: mean over 5 seeds with a 95% t-interval '
         '(seed-paired, crude). Verdicts: *supported* = same sign and the 10k interval excludes 0; *no effect at either budget* = both within noise (30k within 2 seed-std of 0).\n',
         '| group | claim | 30k seed 0 | 10k mean [95% CI] | n pairs | verdict |', '|---|---|---|---|---|---|']
    for c in cl:
        p = 4 if c['col'] == 'ce' else 3
        L.append(f"| {c['group']} | {c['claim'].replace('$', '').replace(chr(92), '')} | {f(c['v30'], p, True)} | {f(c['m10'], p, True)} [{f(c['m10'] - c['h10'], p, True)}, {f(c['m10'] + c['h10'], p, True)}] | {c['n']} | {c['verdict']} |")
    L += ['\n## Gap to the plain baseline in test ppl by model family (10k seed 0 / 10k 5-seed mean ± std / 30k seed 0)\n', '| family | 10k seed 0 | 10k 5 seeds | 30k seed 0 |', '|---|---|---|---|']
    for n, (a, b) in budget.items():
        L.append(f"| {n.replace('$', '').replace(chr(92), '')} | {float(a.iloc[0]):+.2f} | {a.mean():+.2f} ± {a.std(ddof=1):.2f} | {float(b.iloc[0]):+.2f} |")
    L += [f"\nLabel-frequency baseline for concept hit@4 (always predict the 4 most frequent labels): {', '.join(f'{k}: CE {v[0]:.3f}, hit {v[1]:.3f}' for k, v in lf.items() if v)}.",
          f'Random-direction floor (m=1024, d=128): mean max-cos {floor[0]:.3f}, Meta-SAE EV {floor[1]:.3f}.', f'SAE-geometry Spearman table uses {n10} SAEs (10k grid).\n']
    return '\n'.join(L)


def main(main_runs, support_runs, out, md):
    g30, g10 = Grid(main_runs), Grid(support_runs)
    os.makedirs(out, exist_ok=True)
    cl = claims(g30, g10)
    bt, budget = tab_budget(g30, g10)
    ct, lf = tab_concept(g30, g10, main_runs)
    st, floor = tab_sae(g30)
    circ, n10 = tab_circ(g10)
    for name, text in (('tab_lm.tex', tab_lm(g30, g10)), ('tab_claims.tex', tab_claims(cl)), ('tab_budget.tex', bt), ('tab_bp.tex', tab_bp(g30, g10)), ('tab_concept.tex', ct),
                       ('tab_sae.tex', st), ('tab_full_lm.tex', tab_full_lm(g30, g10)), ('tab_full_bp.tex', tab_full_bp(g30, g10)), ('tab_full_sae.tex', tab_full_sae(g30, g10)),
                       ('tab_circ.tex', circ), ('tab_timing.tex', tab_timing(g30, g10))):
        write(out, name, text)
    tm = {'ms': g30.df.t_train_ms_per_step.mean()}
    write(out, 'nseeds.tex', f"\\newcommand{{\\NSEEDS}}{{{len(g10.seeds)}}}\\newcommand{{\\NMAIN}}{{{len(g30.seeds)}}}\\newcommand{{\\StepsMain}}{{30{{,}}000}}\\newcommand{{\\StepsSupport}}{{10{{,}}000}}\n")
    write(md, 'REPORT_CLAIMS.md', md_claims(cl, budget, lf, floor, n10, g30, g10))
    with open(os.path.join(md, 'REPORT_CLAIMS.json'), 'w') as fh:
        json.dump(cl, fh, indent=1)
    print(f'wrote tables to {out}, claims to {md}; main seeds {g30.seeds}, support seeds {g10.seeds}; {tm}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--main', default=os.path.join(paths.ROOT, 'runs_30k'))
    p.add_argument('--support', default=paths.RUNS)
    p.add_argument('--out', default=os.path.join(HERE, 'report', 'generated'))
    p.add_argument('--md', default=os.path.join(paths.ROOT, 'results'))
    a = p.parse_args()
    main(a.main, a.support, a.out, a.md)
