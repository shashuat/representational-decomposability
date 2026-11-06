"""TinyStories -> 4k BPE -> uint16 .bin per split.

Splits are story-level and disjoint: train / sae (post-hoc SAEs and probes) / dev (monitoring only) are consecutive slices of the
HF train set; test is the official validation set and is only used for reporting."""
import os

import numpy as np
import torch
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

import paths

D = os.environ.get('DATA', paths.DATA)


def carve(train_texts, test_texts, n=(300_000, 20_000, 5_000)):
    """Consecutive slices of the train set -> train/sae/dev; official validation -> test."""
    cut = np.cumsum((0,) + n)
    splits = {k: train_texts[int(a):int(b)] for k, a, b in zip(('train', 'sae', 'dev'), cut[:-1], cut[1:])}
    return {**splits, 'test': test_texts}


def prep(V=4096, n=(300_000, 20_000, 5_000)):
    ds = load_dataset('roneneldan/TinyStories', cache_dir=paths.HF)
    S = carve(ds['train'][:sum(n)]['text'], ds['validation']['text'], n)
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer, tk.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    tk.train_from_iterator(S['train'][:100_000], trainers.BpeTrainer(vocab_size=V, special_tokens=['<eos>'],
                                                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    os.makedirs(D, exist_ok=True)
    tk.save(f'{D}/tok.json')
    for split, texts in S.items():  # one <eos> (id 0) after every story
        np.concatenate([np.array(e.ids + [0], dtype=np.uint16) for e in tk.encode_batch(texts)]).tofile(f'{D}/{split}.bin')


def windows(split, T=256):
    """Non-overlapping windows: x (n, T) and y = x shifted by one. The window index keys the cached concept labels."""
    a = torch.from_numpy(np.fromfile(f'{D}/{split}.bin', dtype=np.uint16).astype(np.int64))
    n = (len(a) - 1) // T
    return a[:n * T].view(n, T), a[1:n * T + 1].view(n, T)


if __name__ == '__main__':
    prep()
