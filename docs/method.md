# Method

One decoder-only language model, three independent switches. All of it lives in `src/model.py`, `src/sae.py` and `src/label.py`.

## Base model

Pre-LayerNorm transformer, 2 layers, 2 heads, width `d = 128`, context 256, byte-level BPE vocabulary of 4096 tokens
(about 0.95M parameters). Training loss is next-token cross-entropy (CE).

## Switch 1: Backpack senses (`k`)

Each token gets `k` context-free sense vectors produced by a small shared feed-forward network (with a LayerNorm at its input,
as in the released Backpack code). The logits are

```
logits(t) = sum over positions j, senses l of  alpha[t, j, l] * sense_l(x_j) @ E^T
```

where `alpha` are contextual weights from attention. The sum is exact, so the contribution of every sense of every source token
to every logit can be read off and sense scales can be intervened on linearly. The tests check that the per-sense contributions
sum to the full logit.

## Switch 2: CoCoMix concepts (`C`, `mix`)

A concept head after block 0 predicts which concepts a *teacher* would select for the current window:

1. A TopK sparse autoencoder (SAE) is fitted on the residual stream of a trained plain model (the teacher).
2. For every window, the latents whose activation most changes the next-token loss (activation times gradient, top `Ka = 4`)
   become the concept labels.
3. The student has a concept head trained with CE on these labels (`cm=1`, "concept CE only").
4. With `mix=1` the compressed concept vector is also interleaved into the hidden stream before the remaining blocks (`cm=2`, `cm=3`).

In the unified decoder the concepts act only by re-weighting sense attention, which keeps the Backpack decomposition exact.

## Switch 3: OrtSAE orthogonality penalty (`og`)

The OrtSAE penalty is the mean, over vectors, of the squared maximum cosine similarity to another vector of the same random chunk:

```
Omega(W) = mean_i ( max_{j != i} cos(w_i, w_j) )^2
```

Here it is applied to *every set of vectors the model uses as units*: the senses of a random set of tokens and the columns of the
concept-compression matrix, not only to SAE dictionaries. Weight `gamma = 0.1`.

## Post-hoc SAEs on every model

After training, a TopK SAE and an OrtSAE (`gamma = 0.25`), both with 1024 latents and `K = 16`, are fitted to the hidden states of every model.
This is the one interface all models share, so they can be compared on common ground.

## The eight main configurations

| Model | Senses | Concept head | Penalty | What it isolates |
|---|---|---|---|---|
| P | no | no | no | reference, post-hoc SAEs only |
| CC | no | label loss only | no | cost of supervision without routing |
| CM / CMo | no | fed back | no / on concept directions | cost of routing through concepts |
| BP / BPo | 4 | no | no / on senses | the exact readout |
| BC / BCo | 4 | fed back | no / on both | concepts steering exact senses |

Fifteen further variants (the remaining switch combinations plus two wider plain models used as capacity controls) are trained with one seed.
