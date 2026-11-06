"""Factorial ablation driver.

Factors: Backpack k in {0, 4} x CoCoMix mode cm in {0 none, 1 concept-CE only, 2 CE + interleave, 3 interleave only}
x concept teacher in {TopK SAE, OrtSAE} (only where labels are used) x orthogonality penalty og in {0, G} (only where it touches a parameter),
plus width-matched baselines (d = 136 / 152 ~ parameter count of Backpack / Backpack + CoCoMix) so "more parameters" is separable
from "mechanism".

Usage: python grid.py {list|train|eval|agg} [train.K=V label.K=V eval.K=V ...]
Env:   OUT (runs dir, default paths.RUNS = $RDP_ROOT/runs), SEEDS (default "0 1 2"), SLN (default 1).
`train` first trains the baseline, builds both teachers from it, then every cell. Passthrough example:
  python grid.py train train.steps=20 label.m=64 eval.sae='{"m": 64, "K": 4, "steps": 20}'
"""
import glob
import itertools
import json
import os
import subprocess
import sys

import paths

OUT, G = os.environ.get('OUT', paths.RUNS), .1
SLN = int(os.environ.get('SLN', '1'))  # LayerNorm in front of the Backpack sense net (released-code faithful; SLN=0 reproduces the first 3000-step grid)
SEEDS = tuple(map(int, os.environ.get('SEEDS', '0 1 2').split()))
WIDTH_MATCHED = (136, 152)
ONLY = tuple(filter(None, os.environ.get('ONLY', '').split(',')))  # keep only these cells (name without the seed suffix), e.g. the eight main models
BASE = 'k0_cm0_-_og0_s0'  # the baseline run that also serves as the concept teacher


def cfgs():
    """All run configurations (dicts of train.py flags); the name encodes the cell: k{k}_cm{cm}_{teacher}_og{og}_s{seed}."""
    for s, k, cm, t, og in itertools.product(SEEDS, (0, 4), (0, 1, 2, 3), ('topk', 'ort'), (0, G)):
        uses_labels = cm in (1, 2)
        if (not uses_labels and t == 'ort') or (og and not (k or cm >= 2)):
            continue
        yield {'name': f'k{k}_cm{cm}_{t if uses_labels else "-"}_og{og}_s{s}', 'seed': s, 'k': k, 'C': 512 * (cm > 0), 'mix': int(cm >= 2),
               'lam': .1 * uses_labels, 'og': og, 'sln': SLN * (k > 0), 'lab': f'{OUT}/teacher/{t}' if uses_labels else ''}
    for s, d in itertools.product(SEEDS, WIDTH_MATCHED):
        yield {'name': f'k0_cm0_wm{d}_og0_s{s}', 'seed': s, 'k': 0, 'C': 0, 'mix': 0, 'lam': 0., 'og': 0, 'lab': '', 'd': d, 'sln': 0}


def run(*cmd):
    subprocess.run([sys.executable, *cmd], check=True)


def flags(cfg):
    return [x for k, v in cfg.items() for x in (f'--{k}', str(v))]


def passthrough(prefix):
    """Extra CLI tokens 'prefix.key=value' -> ['--key', 'value'] appended to that stage's command."""
    return [x for a in sys.argv[2:] if a.startswith(prefix + '.') for x in (f"--{a[len(prefix) + 1:].split('=')[0]}", a.split('=', 1)[1])]


def train(configs):
    base = f'{OUT}/{BASE}/ckpt.pt'
    if not os.path.exists(base):
        run('train.py', '--name', BASE, '--out', OUT, *passthrough('train'))
    for teacher, g in (('topk', 0.), ('ort', .25)):
        if not os.path.exists(f'{OUT}/teacher/{teacher}/test.pt'):
            run('label.py', '--teacher', base, '--out', f'{OUT}/teacher/{teacher}', '--g', str(g), *passthrough('label'))
    for c in configs:
        if not os.path.exists(f"{OUT}/{c['name']}/ckpt.pt"):
            run('train.py', *flags(c), '--out', OUT, *passthrough('train'))


def evaluate(configs):
    for c in configs:
        if not os.path.exists(f"{OUT}/{c['name']}/eval_test.json"):
            run('evaluate.py', '--ckpt', f"{OUT}/{c['name']}/ckpt.pt", *passthrough('eval'))


def agg(keys=('ce', 'acc', 'concept_hit', 'gender_best_reduction')):
    """Print the seed-mean of the headline metrics per cell (full tables: plots.py)."""
    cells = {}
    for f in glob.glob(f'{OUT}/*/eval_test.json'):
        with open(f) as fh:
            r = json.load(fh)
        cells.setdefault(r['name'].rsplit('_s', 1)[0], []).append(r)
    mean = lambda v: sum(v) / len(v)
    for name, rs in sorted(cells.items()):
        top = [f'{k}={mean([r[k] for r in rs]):.4f}' for k in keys if k in rs[0]]
        sae = [f'{g}.{k}={mean([r[g][k] for r in rs]):.3f}' for g in ('sae_g0.0', 'sae_g0.25') for k in ('ev', 'ce_recovered', 'mean_max_cos', 'absorption')]
        print(name, len(rs), *top, *sae)


if __name__ == '__main__':
    configs, cmd = [c for c in cfgs() if not ONLY or c['name'].rsplit('_s', 1)[0] in ONLY], sys.argv[1] if len(sys.argv) > 1 else 'list'
    if cmd == 'list':
        print(len(configs), 'configs')
        for c in configs:
            print(c['name'])
    elif cmd == 'train':
        train(configs)
    elif cmd == 'eval':
        evaluate(configs)
    elif cmd == 'agg':
        agg()
