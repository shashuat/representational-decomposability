"""Single source of truth for where heavy artifacts live. Override the root with RDP_ROOT; falls back to ../ if the root does not exist.

ROOT/data       tokenizer + {train,sae,dev,test}.bin
ROOT/runs       <name>/{ckpt.pt, log.jsonl, eval_test.json, sae_decoders.pt}, teacher/{topk,ort}/{sae.pt, train|dev|test.pt}
ROOT/figures    plots (pdf + png)
ROOT/results    text aggregates (summary.txt, aggregates.csv, *.md)
ROOT/logs       stdout of long jobs
ROOT/hf_cache   HuggingFace cache (datasets via cache_dir; hub downloads via the conda env var HF_HOME)"""
import os

_default = '/Data/shash/red_dec'
ROOT = os.environ.get('RDP_ROOT', _default if os.path.isdir(_default) else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'artifacts'))
DATA, RUNS, FIGS, RESULTS, LOGS, HF = (os.path.join(ROOT, d) for d in ('data', 'runs', 'figures', 'results', 'logs', 'hf_cache'))
