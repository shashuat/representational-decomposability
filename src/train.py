"""Step 1 of 2: train one configuration, write runs/<name>/{ckpt.pt, log.jsonl}.
Evaluation is a separate process (evaluate.py) that re-loads the checkpoint.
The checkpoint holds the weights of the step with the lowest dev CE (dev is monitoring only; test is never used for selection)."""
import argparse
import json
import math
import os
import time

import torch

import paths
from data import windows
from model import LM

DEF = {'name': 'base', 'seed': 0, 'V': 4096, 'd': 128, 'L': 2, 'H': 2, 'T': 256, 'k': 0, 'C': 0, 'mix': 0, 'Kc': 4, 'lam': .1, 'og': 0., 'lab': '',
       'steps': 10000, 'bs': 128, 'lr': 2e-3, 'wd': .1, 'warm': 200, 'out': paths.RUNS, 'comp': 1, 'sln': 0,
       'es': 0, 'patience': 1000, 'min_delta': .002, 'es_start': 2000}  # optional early stopping on dev CE (off in the grid: a cosine lr makes it truncate stalls)


def lr_at(a, t):
    """Linear warmup then cosine decay to zero."""
    return a['lr'] * min(1, (t + 1) / a['warm']) * .5 * (1 + math.cos(math.pi * t / a['steps']))


def main(a):
    t_start = time.time()
    torch.manual_seed(a['seed'])
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    X, Y = (t.to(dev) for t in windows('train', a['T']))
    labels = torch.load(f"{a['lab']}/train.pt").to(dev) if a['lab'] else None  # (n, T, Ka) concept labels aligned with the windows
    Xd, Yd = (t[:512].to(dev) for t in windows('dev', a['T']))  # dev: monitoring only
    model = LM(**a).to(dev)
    fwd = torch.compile(model) if a['comp'] else model
    opt = torch.optim.AdamW(model.parameters(), a['lr'], betas=(.9, .95), weight_decay=a['wd'])
    run_dir = f"{a['out']}/{a['name']}"
    os.makedirs(run_dir, exist_ok=True)
    sync = torch.cuda.synchronize if dev == 'cuda' else (lambda: None)
    acc = torch.zeros(2, device=dev)  # running sums of (ce, lc) since the last log line: window means without a host sync per step
    sync()
    best, best_step, best_sd, ref, ref_step = float('inf'), 0, None, float('inf'), 0  # lowest dev CE (checkpoint), last gain of at least min_delta (patience)
    setup_s, t_train, t_last, t_dev, first_step_s, log_step = time.time() - t_start, time.time(), time.time(), 0., 0., 0
    with open(f'{run_dir}/log.jsonl', 'w') as log:
        for t in range(a['steps']):
            for g in opt.param_groups:
                g['lr'] = lr_at(a, t)
            i = torch.randint(len(X), (a['bs'],), device=dev)
            with torch.autocast(dev, torch.bfloat16):
                _, o = fwd(X[i], Y[i], lab=None if labels is None else labels[i].long())
            loss = o['ce'] + a['lam'] * o.get('lc', 0) + a['og'] * model.pen()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            acc += torch.stack([o['ce'].detach(), torch.as_tensor(o.get('lc', 0.), device=dev).detach()])
            if t == 0:
                sync()
                first_step_s = time.time() - t_train  # includes torch.compile
            if t % 100 == 0 or t == a['steps'] - 1:
                sync()
                now = time.time()
                n, elapsed = t - log_step + 1, now - t_train - t_dev  # steps since the last log line; training wall time excluding dev evals
                tr_ce, tr_lc = (acc / n).tolist()
                with torch.no_grad(), torch.autocast(dev, torch.bfloat16):
                    dev_ce = model(Xd, Yd)[1]['ce'].item()
                t_dev += time.time() - now
                if dev_ce < best:
                    best, best_step, best_sd = dev_ce, t, {k: v.detach().clone() for k, v in model.state_dict().items()}
                if dev_ce < ref - a['min_delta']:
                    ref, ref_step = dev_ce, t
                stop = bool(a['es']) and t >= a['es_start'] and t - ref_step >= a['patience']
                rec = {'step': t, 'ce': tr_ce, 'lc': tr_lc, 'dev_ce': dev_ce, 'best_dev_ce': best, 'elapsed_s': elapsed,
                       'tok_s': n * a['bs'] * a['T'] / max(now - t_last, 1e-9), 'lr': lr_at(a, t)}
                log.write(json.dumps(rec) + '\n')
                log.flush()
                print(a['name'], rec)
                acc.zero_()
                log_step, t_last = t + 1, time.time()
                if stop:
                    break
    sync()
    train_s = time.time() - t_train
    t_save = time.time()
    torch.save({'cfg': a, 'sd': best_sd}, f'{run_dir}/ckpt.pt')
    timing = {'setup_s': setup_s, 'first_step_s': first_step_s, 'train_s': train_s, 'dev_eval_s': t_dev, 'steps': a['steps'],
              'steps_run': t + 1, 'best_step': best_step, 'best_dev_ce': best, 'early_stopped': t + 1 < a['steps'],
              'ms_per_step': 1e3 * (train_s - first_step_s - t_dev) / max(t, 1), 'tok_s': a['bs'] * a['T'] * (t + 1) / max(train_s - t_dev, 1e-9),
              'save_s': time.time() - t_save, 'total_s': time.time() - t_start, 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else 0.}
    with open(f'{run_dir}/timing.json', 'w') as f:
        json.dump(timing, f, indent=1)
    print(a['name'], 'timing', json.dumps(timing))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key, val in DEF.items():
        parser.add_argument(f'--{key}', type=type(val), default=val)
    main(vars(parser.parse_args()))
