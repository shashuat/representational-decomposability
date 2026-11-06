# Experiments

## Data

[TinyStories](https://arxiv.org/abs/2305.07759), tokenised with a 4096-token byte-level BPE fitted on the first 100k stories.
Stories are concatenated and cut into non-overlapping 256-token windows. Four disjoint story-level splits:

| split | stories | use |
|---|---|---|
| train | 300k (69.4M tokens) | language-model training only |
| sae | 20k | fitting post-hoc SAEs only |
| dev | 5k | picking the lowest-CE checkpoint only |
| test | 21,990 (official validation split) | every reported number |

## Training

Every model is trained from scratch for 30,000 steps (batch 128 x 256 tokens, about 983M tokens or 14 passes over the training set),
learning rate 2e-3 with warmup and cosine decay, `torch.compile`, and the best-dev-CE checkpoint is kept.
The eight main configurations use four seeds each; the other 15 variants (23 models in all) use one seed.
Figures report the mean with standard error across seeds.

## Evaluation protocol

| property | question |
|---|---|
| readability | which tokens does a unit stand for, how word-like is it? |
| attribution | does the explanation of a prediction agree with the effect of removing the unit? |
| intervention | how does the next-token distribution change when a unit is scaled or shifted, and how much do unrelated tokens change? |
| cost | test perplexity and its trajectory during training |

## Quality cost (test perplexity, mean over four seeds)

| model | params (M) | train time | test ppl |
|---|---|---|---|
| plain | 0.95 | 1.0x | 7.62 |
| concept CE only | 1.09 | 1.1x | 7.66 |
| CoCoMix | 1.09 | 1.6x | 7.74 |
| Backpack | 1.07 | 1.4x | 8.04 |
| Backpack + CoCoMix | 1.20 | 1.9x | 8.25 |

Penalised variants match their unpenalised counterparts within noise. Adding supervision (concept CE) is almost free; the cost appears
where computation is rerouted (CoCoMix feedback, Backpack readout), and wider plain models recover it.

## What the probes show

- **Backpack**: every trained model has one copy sense and one next-token sense. Scaling a sense changes the probability of its own tokens smoothly while the other senses' tokens barely move.
- **CoCoMix**: the concept head predicts the teacher's concepts well above chance (hit@4 about 0.37 to 0.39), and the channel is writable, but steering is bounded by the teacher's vocabulary.
- **OrtSAE penalty**: decoder cosine similarity drops, but absorption and readability do not improve.

## Limits

Single small model (about 1M parameters, 2 layers) on one synthetic dataset; no sweeps over `k`, the penalty weight, the number of labels or the
dictionary size; no topic control, knowledge editing or SAEBench evaluation. The sense chosen in the gender-editing demo was selected on test prompts, so that
experiment shows the intervention exists rather than that a sense denotes "gender".

The full numbers, figures and discussion are in the [report](../report/representational-decomposability-report.pdf).
