# Code guide

Everything is in `src/`. Python 3.11, `torch>=2.1`, `tokenizers`, `datasets`, `numpy`, plus `pandas` and `matplotlib` for the plots.
`ruff check .` is clean (`ruff.toml` allows compact one-liners and short mathematical names).

## Where artifacts go

All heavy outputs live under `$RDP_ROOT` (see `paths.py`). If unset, `/Data/shash/red_dec` is used when it exists, otherwise `./artifacts` next to the repo.

```
$RDP_ROOT/data/      tokenizer and {train,sae,dev,test}.bin
$RDP_ROOT/runs/      <run>/{ckpt.pt, log.jsonl, eval_test.json}, teacher/ SAEs and labels
$RDP_ROOT/figures/   plots
$RDP_ROOT/results/   text aggregates (summary.txt, aggregates.csv, ...)
```

## Files

| file | role |
|---|---|
| `paths.py` | all output locations |
| `data.py` | tokenizer, story-level splits, non-overlapping windows |
| `model.py` | `LM`: baseline / Backpack (`k`) / CoCoMix (`C`, `mix`) / orthogonality penalty (`og`); intervention arguments for sense scales and concept steering |
| `sae.py` | TopK SAE with AuxK dead-latent revival, unit-norm decoder, optional OrtSAE penalty |
| `label.py` | concept teacher: SAE on the baseline residual and attribution-based top-`Ka` labels per window |
| `train.py` | step 1: train and write a checkpoint |
| `evaluate.py` | step 2: reload the checkpoint in a fresh process and compute all metrics |
| `grid.py` | the factorial grid: configurations, orchestration, aggregation |
| `run_grid.sh` | resumable train, eval and plot stage (`TAG=_30k STEPS=30000` for the 30k-step grid) |
| `interp_stats.py` | standard errors of every metric and post-hoc SAE |
| `interp_viz.py`, `interp_agg.py` | the interpretability probes and the figures and tables of the report (single-seed and four-seed) |
| `plots.py`, `report_data.py`, `report_figs.py`, `report_tables.py` | ablation plots, aggregates and LaTeX table fragments |
| `fig_arch.py` | architecture schematic |
| `test_all.py` | 21 tests |

## Reproduce

```bash
cd src
python test_all.py                 # synthetic tokens, CPU is enough
E2E=1 python test_all.py           # plus the whole grid end to end (train, teachers, cells, eval, aggregate, plots)
python data.py                     # download and tokenise TinyStories
python grid.py list                # list the configurations
python grid.py train               # baseline -> teacher SAEs -> attribution labels -> all runs
python grid.py eval                # one fresh process per checkpoint
python grid.py agg                 # mean over seeds per cell
python plots.py                    # figures and text aggregates
```

The 30,000-step grid: `TAG=_30k STEPS=30000 ./run_grid.sh seed0 "0"`. Seeds are chosen with `SEEDS="0 1 2 4"` and cells with `ONLY=<cell,...>`.

A single run:

```bash
python train.py --k 4 --C 512 --mix 1 --lab $RDP_ROOT/runs/teacher/ort --og 0.1 --name my_run
python evaluate.py --ckpt $RDP_ROOT/runs/my_run/ckpt.pt
```

Smoke test of the whole pipeline with tiny settings:

```bash
SEEDS="0" OUT=/tmp/r python grid.py train train.steps=20 train.d=32 label.m=64 eval.sae='{"m":64,"K":4,"steps":20}'
```

## Tests

`test_all.py` checks, among others: causality, that the Backpack equals an explicit reference implementation, that sense interventions are linear,
the interleave layout, the penalty maths, SAE sparsity and fit, that the penalty lowers decoder cosine, the attribution chain rule, checkpoint round trips and the probe code.
