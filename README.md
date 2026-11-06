# Representational Decomposability

**Studying interpretability in LLMs using representational decomposability.**

Post-hoc interpretability tools (probes, sparse autoencoders) approximate the computation they explain.
This project instead builds interpretability *into* a small language model and asks what each mechanism buys and what it costs.
It combines three published ideas in one decoder-only transformer trained from scratch on TinyStories:

| granularity | mechanism | idea |
|---|---|---|
| token | **Backpack LMs** (Hewitt et al., 2023) | logits are an exact sum of context-free per-token sense vectors |
| concept | **CoCoMix** (Tack et al., 2025) | the model predicts, and is fed back, concepts from an SAE dictionary |
| feature | **OrtSAE** (Korznikov et al., 2025) | an orthogonality penalty pushes dictionary vectors apart |

Each mechanism is an independent switch in a single forward pass (`src/model.py`), so eight main configurations
(four seeds each) and fifteen further variants can be compared with one shared set of probes for readability,
attribution faithfulness, intervention selectivity and training cost.

## Findings

- Exact additive decomposition (Backpack) gives faithful attributions and selective interventions that learned interfaces do not.
- The concept channel is writable, but its efficacy is bounded by the teacher's vocabulary.
- Orthogonality lowers the similarity between units without improving their readability.
- The perplexity cost comes from mechanisms that *reroute* computation, not from added supervision, and it is recovered by widening the plain model.

Full write-up: [`report/representational-decomposability-report.pdf`](report/representational-decomposability-report.pdf) (17 pages).

## Repository layout

```
src/        model, SAE, data, training, evaluation, analysis and plotting code, plus 21 tests
docs/       method, experiments and code guide
report/     the report (PDF)
```

| doc | content |
|---|---|
| [`docs/method.md`](docs/method.md) | the model, the three switches, the concept teacher and the penalty |
| [`docs/experiments.md`](docs/experiments.md) | data, the model grid, evaluation protocol, results summary and limits |
| [`docs/code-guide.md`](docs/code-guide.md) | what every file in `src/` does and how to reproduce a run |

## Quick start

```bash
cd src
pip install -r requirements.txt matplotlib pandas   # torch>=2.1, tokenizers, datasets, numpy
python test_all.py                                  # 21 tests on synthetic tokens, no download needed
python data.py                                      # TinyStories -> 4k BPE -> disjoint story-level splits
python grid.py train && python grid.py eval         # train the grid, then evaluate each checkpoint
```

Artifacts (data, checkpoints, figures, results) go under `$RDP_ROOT`; see [`docs/code-guide.md`](docs/code-guide.md).

## Scale and caveats

The model has 2 layers, 2 heads, width 128 and about 1M parameters, trained on TinyStories on a single GPU.
Conclusions are about this scale only; whether the separation between the mechanisms holds on larger models,
with stronger concept teachers, is left as future work.

## Acknowledgements

Supervised by Professor Mehwish Alam (Institut Polytechnique de Paris).

Author: Shashwat Sharma.
