"""Report figures: 30k steps, seed 0 are the headline numbers; the 10k, 5-seed grid is overlaid as the statistical support. CPU only.

  python report_figs.py [--main RUNS_30K] [--support RUNS_10K] [--out FIGDIR]  ->  rf_*.pdf / rf_*.png

Colour = model class (blue plain baselines, orange concept head on a plain LM, green Backpack k=4); marker shape = concept feedback (circle none/CE only, square interleave);
filled = 30k seed 0, hollow with bar = 10k mean +- std over 5 seeds. First three slots of the reference palette, one axis per panel, thin marks."""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import paths
import plots
import report_data as rd

BLUE, ORANGE, GREEN, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e6e5e1'
plt.switch_backend('Agg')
plt.rcParams.update({'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'text.color': INK, 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': .6, 'axes.axisbelow': True, 'figure.facecolor': 'white',
                     'axes.facecolor': 'white', 'legend.frameon': False, 'pdf.fonttype': 42})


def cls(cell):
    q = plots.parse(cell + '_s0')
    return 'plain' if q['teacher'].startswith('wm') or (q['k'] == 0 and q['cm'] == 0) else ('k0' if q['k'] == 0 else 'bp')


COL = {'plain': BLUE, 'k0': ORANGE, 'bp': GREEN}
CLASS_NAME = {'plain': 'plain LM', 'k0': 'concept head (k=0)', 'bp': 'Backpack k=4'}


def mk(cell):
    q = plots.parse(cell + '_s0')
    return 's' if q['cm'] in (2, 3) else 'o'


def plain_label(cell):
    q = plots.parse(cell + '_s0')
    if q['teacher'].startswith('wm'):
        return f"plain d={q['teacher'][2:]}"
    parts = ['Backpack'] * bool(q['k']) + [['', 'CE only', 'CE+interleave', 'interleave only'][q['cm']] + (f" ({'Ort' if q['teacher'] == 'ort' else 'TopK'})" if q['cm'] in (1, 2) else '')] * bool(q['cm'])
    return ' + '.join(parts + ['ortho'] * bool(q['og'])) or 'baseline'


def save(fig, out, name):
    fig.savefig(f'{out}/{name}.pdf', bbox_inches='tight')
    fig.savefig(f'{out}/{name}.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def class_legend(ax, extra=(), loc='lower right'):
    h = [plt.Line2D([], [], marker='o', ls='', color=COL[k], label=CLASS_NAME[k]) for k in COL] + [plt.Line2D([], [], marker='s', ls='', color=MUTED, label='interleave')] + list(extra)
    ax.legend(handles=h, loc=loc, fontsize=7, handletextpad=.3)


def fig_lm(g30, g10, out):
    cells = sorted(g30.cells, key=lambda c: -g30.one(c, 'ppl'))
    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    for i, c in enumerate(cells):
        v10 = g10.P('ppl')[c].dropna().values
        ax.plot([v10.mean(), g30.one(c, 'ppl')], [i, i], color=GRID, lw=1.2, zorder=1)
        ax.errorbar(v10.mean(), i, xerr=v10.std(ddof=1), fmt=mk(c), mfc='white', mec=COL[cls(c)], ecolor=COL[cls(c)], elinewidth=.9, ms=5, capsize=2, zorder=2)
        ax.plot(g30.one(c, 'ppl'), i, mk(c), color=COL[cls(c)], ms=5.5, zorder=3)
    ax.axvline(g30.one(rd.BASE, 'ppl'), color=MUTED, lw=.8, ls=':')
    ax.set_yticks(range(len(cells)), [plain_label(c) for c in cells])
    ax.set_xlabel('test perplexity (lower is better)')
    ax.grid(axis='y', visible=False)
    class_legend(ax, [plt.Line2D([], [], marker='o', ls='', color=INK, label='30k steps, seed 0'), plt.Line2D([], [], marker='o', ls='', mfc='white', mec=INK, label='10k steps, mean $\\pm$ std (5 seeds)')], 'upper right')
    save(fig, out, 'rf_lm')


def fig_budget(g30, g10, out):
    fam = rd.families()
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    for i, (n, cs) in enumerate(fam.items()):
        a, b = rd.family_gap(g10, cs), rd.family_gap(g30, cs)
        k = cls(cs[0]) if n != 'plain $d{=}136$ / $d{=}152$' else 'plain'
        y = len(fam) - 1 - i
        ax.plot([a.mean(), float(b.iloc[0])], [y, y], color=GRID, lw=1.4, zorder=1)
        ax.errorbar(a.mean(), y, xerr=a.std(ddof=1), fmt='o', mfc='white', mec=COL[k], ecolor=COL[k], elinewidth=.9, ms=6, capsize=2, zorder=2)
        ax.plot(float(b.iloc[0]), y, 'o', color=COL[k], ms=6.5, zorder=3)
        ax.annotate(f"{a.mean():+.2f} $\\to$ {float(b.iloc[0]):+.2f}", xy=(1.01, y), xycoords=('axes fraction', 'data'), va='center', fontsize=6.5, color=MUTED, annotation_clip=False)
    ax.axvline(0, color=MUTED, lw=.8)
    ax.set_yticks(range(len(fam)), [n.replace('$', '').replace('{=}', '=').replace('\\', '') for n in list(fam)[::-1]])
    ax.set_xlim(-0.8, 4.7)
    ax.set_xlabel('gap to the plain baseline in test perplexity (ppl; family mean over its cells)')
    ax.annotate('10k $\\to$ 30k', xy=(1.01, len(fam) - .3), xycoords=('axes fraction', 'data'), fontsize=6.5, color=INK, annotation_clip=False)
    ax.grid(axis='y', visible=False)
    class_legend(ax, [plt.Line2D([], [], marker='o', ls='', color=INK, label='30k steps, seed 0'), plt.Line2D([], [], marker='o', ls='', mfc='white', mec=INK, label='10k steps, mean $\\pm$ std')], 'center right')
    save(fig, out, 'rf_budget')


def strip(s):
    return s.replace('$', '').replace('{=}', '=').replace('\\gamma_o', 'gamma_o').replace('\\gamma', 'gamma').replace('\\', '').replace('^2', '²')


def fig_claims(cl, out):
    """Forest plot of the like-scaled claims: LM-quality effects (nats of test CE) and post-hoc SAE behaviour (OrtSAE minus TopK, metric units)."""
    panels = [('Effect on test CE (nats): A minus B', [c for c in cl if c['group'] == 'LM quality'], lambda t: strip(t.replace(': test CE', ''))),
              ('Post-hoc SAE: OrtSAE minus TopK', [c for c in cl if c['group'] == 'Post-hoc SAE' and c['col'] in ('d_ev', 'd_absorption', 'd_ce_recovered', 'd_probe_f1')],
               lambda t: strip(t.split(': ')[-1]))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={'width_ratios': [1.25, 1], 'wspace': .85})
    for ax, (ttl, rows, lab) in zip(axes, panels):
        rows = rows[::-1]
        for i, c in enumerate(rows):
            ax.errorbar(c['m10'], i, xerr=c['h10'], fmt='o', color=BLUE, ms=4.5, elinewidth=1.1, capsize=2.5, zorder=2)
            ax.plot(c['v30'], i, 'D', color=ORANGE, ms=5, zorder=3)
        ax.axvline(0, color=MUTED, lw=.8)
        ax.set_yticks(range(len(rows)), [lab(c['claim']) for c in rows], fontsize=6.5)
        ax.set_title(ttl, fontsize=8, loc='left')
        ax.grid(axis='y', visible=False)
    axes[0].legend(handles=[plt.Line2D([], [], marker='D', ls='', color=ORANGE, label='30k steps, seed 0'), plt.Line2D([], [], marker='o', ls='-', color=BLUE, label='10k steps: mean, 95% CI over 5 seeds')],
                   loc='upper left', bbox_to_anchor=(0, -.12), fontsize=6.5, ncol=2)
    save(fig, out, 'rf_claims')


def fig_curves(runs, out):
    sel = [('k0_cm0_-_og0', '-'), ('k0_cm0_wm152_og0', '--'), ('k0_cm1_topk_og0', '-'), ('k0_cm2_topk_og0', '--'), ('k4_cm0_-_og0', '-'), ('k4_cm2_topk_og0', '--')]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ends = {}
    for c, ls in sel:
        with open(f'{runs}/{c}_s0/log.jsonl') as fh:
            L = [json.loads(line) for line in fh]
        st, dv = np.array([r['step'] for r in L]), np.array([r['dev_ce'] for r in L])
        for ax, m in zip(axes, (st >= 0, st >= 6000)):
            ax.plot(st[m], dv[m], color=COL[cls(c)], ls=ls, lw=1.2)
        ends[c] = dv[-1]
    ys, last = {}, -9.
    for c in sorted(ends, key=ends.get):  # spread the end labels so they do not overlap
        ys[c] = max(ends[c], last + .022)
        last = ys[c]
    for c, y in ys.items():
        axes[1].text(30000 + 350, y, plain_label(c), color=MUTED, fontsize=6.3, va='center')
    axes[0].set_ylim(1.9, 4.5)
    axes[1].set_ylim(1.95, 2.3)
    axes[1].set_xlim(6000, 41000)
    axes[0].set_title('dev cross-entropy during training (seed 0)', fontsize=8, loc='left')
    axes[1].set_title('zoom on the last 24k steps', fontsize=8, loc='left')
    for ax in axes:
        ax.set_xlabel('step')
    axes[0].set_ylabel('dev CE (nats)')
    save(fig, out, 'rf_curves')


def fig_interfaces(g30, g10, lf, out):
    ccells = [f'k{k}_cm{m}_{t}_og0' for k in (0, 4) for m in (1, 2) for t in ('topk', 'ort')]
    gcells = ['k4_cm0_-_og0', 'k4_cm1_topk_og0', 'k4_cm1_ort_og0', 'k4_cm2_topk_og0', 'k4_cm2_ort_og0', 'k4_cm3_-_og0']
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0), gridspec_kw={'wspace': .45})
    for ax, col, ttl in zip(axes[:2], ('concept_hit', 'steer_kl'), ('concept recovery: precision@4', 'steering KL (+5 on one concept logit)')):
        for i, c in enumerate(ccells):
            v = g10.P(col)[c].dropna().values
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt=mk(c), mfc='white', mec=COL[cls(c)], ecolor=COL[cls(c)], ms=5, capsize=2)
            ax.plot(i, g30.one(c, col), mk(c), color=COL[cls(c)], ms=5.5)
        ax.set_xticks(range(len(ccells)), [plain_label(c).replace('Backpack + ', 'BP + ').replace(' (TopK)', ' T').replace(' (Ort)', ' O') for c in ccells], rotation=60, ha='right', fontsize=6)
        ax.set_title(ttl, fontsize=7.5, loc='left')
        ax.grid(axis='x', visible=False)
    if lf.get('topk'):
        axes[0].axhline(lf['topk'][1], color=MUTED, ls=':', lw=1)
        axes[0].text(len(ccells) - .5, lf['topk'][1] + .005, 'label-frequency baseline', ha='right', fontsize=6.3, color=MUTED)
    ax = axes[2]
    for i, c in enumerate(gcells):
        v = g10.P('gender_best_reduction')[c].dropna().values
        ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt='o', mfc='white', mec=COL['bp'], ecolor=COL['bp'], ms=5, capsize=2)
        ax.plot(i, g30.one(c, 'gender_best_reduction'), 'o', color=COL['bp'], ms=5.5)
        ax.plot(i, g30.one(c, 'gender_mean_red'), 'x', color=MUTED, ms=5)
    ax.set_xticks(range(len(gcells)), [plain_label(c).replace('Backpack + ', 'BP + ').replace('Backpack', 'BP').replace(' (TopK)', ' T').replace(' (Ort)', ' O') for c in gcells], rotation=60, ha='right', fontsize=6)
    ax.set_title('gender gap removed by zeroing one sense', fontsize=7.5, loc='left')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='x', visible=False)
    ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=COL['bp'], label='best sense, 30k'), plt.Line2D([], [], marker='o', ls='', mfc='white', mec=COL['bp'], label='best sense, 10k mean $\\pm$ std'),
                       plt.Line2D([], [], marker='x', ls='', color=MUTED, label='mean over senses, 30k')], fontsize=6, loc='upper left', ncol=1)
    save(fig, out, 'rf_interfaces')


def fig_sae(g30, g10, floor, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), gridspec_kw={'width_ratios': [1.1, 1.5], 'wspace': .95})
    ax = axes[0]
    fam = [('k0_cm0_-_og0', 'baseline'), ('k4_cm0_-_og0', 'Backpack'), ('k0_cm2_topk_og0', 'CE +\ninterleave'), ('k4_cm2_topk_og0', 'Backpack\n+ CE +\ninterleave')]
    w = .36
    for i, (c, _) in enumerate(fam):
        for j, (gg, mfc, lab) in enumerate((('0.0', COL[cls(c)], 'TopK SAE'), ('0.25', 'white', 'OrtSAE'))):
            ax.bar(i + (j - .5) * w, g30.one(c, f'sae{gg}_mean_max_cos'), w * .92, color=mfc, edgecolor=COL[cls(c)], lw=1.1, label=lab if i == 0 else None)
    ax.axhline(floor[0], color=MUTED, ls=':', lw=1)
    ax.text(-.5, floor[0] + .008, 'random directions', ha='left', fontsize=6.3, color=MUTED)
    ax.set_xticks(range(len(fam)), [n for _, n in fam], fontsize=6.5)
    ax.set_ylabel('mean decoder max-cosine')
    ax.set_title('geometry (falls by construction)', fontsize=7.5, loc='left')
    ax.legend(fontsize=6.5, loc='upper center', bbox_to_anchor=(.5, -.2), ncol=2)
    ax.set_ylim(0, .42)
    ax.grid(axis='x', visible=False)
    ax = axes[1]
    cells = sorted(g30.cells, key=lambda c: g30.one(c, 'sae0.0_absorption'))
    for i, c in enumerate(cells):
        a, b = g30.one(c, 'sae0.0_absorption'), g30.one(c, 'sae0.25_absorption')
        ax.plot([a, b], [i, i], color=GRID, lw=1.2, zorder=1)
        ax.plot(a, i, mk(c), color=COL[cls(c)], ms=4.5, zorder=2)
        ax.plot(b, i, mk(c), mfc='white', mec=COL[cls(c)], ms=4.5, zorder=3)
    ax.set_yticks(range(len(cells)), [plain_label(c) for c in cells], fontsize=5.6)
    ax.set_xlabel('first-letter absorption proxy (lower is better)')
    ax.set_title('behaviour: TopK SAE (filled) vs OrtSAE (hollow)', fontsize=7.5, loc='left')
    ax.grid(axis='y', visible=False)
    save(fig, out, 'rf_sae')


def main(main_runs, support_runs, out):
    g30, g10 = rd.Grid(main_runs), rd.Grid(support_runs)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(paths.RESULTS, 'REPORT_CLAIMS.json')) as fh:
        cl = json.load(fh)
    lf = {t: rd.rt.label_freq(main_runs, t) for t in ('topk', 'ort')}
    floor = rd.rt.random_floor()
    floor = (float(np.mean(floor[0])), float(np.mean(floor[1])))
    fig_lm(g30, g10, out)
    fig_budget(g30, g10, out)
    fig_claims(cl, out)
    fig_curves(main_runs, out)
    fig_interfaces(g30, g10, lf, out)
    fig_sae(g30, g10, floor, out)
    print('wrote rf_*.pdf/png to', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--main', default=os.path.join(paths.ROOT, 'runs_30k'))
    p.add_argument('--support', default=paths.RUNS)
    p.add_argument('--out', default=os.path.join(paths.ROOT, 'figures_30k'))
    a = p.parse_args()
    main(a.main, a.support, a.out)
