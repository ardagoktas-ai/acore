"""
Acore-WM training script (pure PyTorch, no custom CUDA packages).

Acore-WM is a small world model of Acore's micro-universes: given the history
of (grid observation, action) it predicts the next grid, the reward and the
done flag. Data is generated on the fly by envs.py.

Tokenisation (per time step t):
    MAX_H row tokens of o_t   (each row: cell embeddings -> linear -> d_model)
    1 action token a_t
followed by the MAX_H row tokens of the final observation. The sequence is
processed by a causal decoder transformer. Targets:
    at the action token a_t      -> row 0 of o_{t+1}, reward_t, done_t
    at row token r of o_{t+1}    -> row r+1 of o_{t+1}
so each next frame is predicted row by row, conditioned on everything before.

Usage:
    python train.py --smoke                      # tiny CPU run, <= ~3 minutes
    python train.py --price-per-hour <USD>       # full run on one GPU
    python train.py --resume                     # continue from out_dir/latest.pt
"""

import argparse
import csv
import math
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs import (MAX_H, MAX_W, N_ACTIONS, OBS_VOCAB, PAD, REWARD_VALUES,
                  TrajectoryStream, generate_batch)

EVAL_SEED_BASE = 10 ** 9            # held-out worlds never seen in training


# ---------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.n_head, self.dropout = n_head, dropout
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, n, 3, self.n_head, d // self.n_head) \
            .permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(self.ln2(x))


def seq_len(T):
    return T * (MAX_H + 1) + MAX_H


class WorldModel(nn.Module):
    def __init__(self, d_model, n_layer, n_head, d_cell, T, dropout=0.0):
        super().__init__()
        self.T = T
        self.cell = nn.Embedding(OBS_VOCAB, d_cell)
        self.row_in = nn.Linear(MAX_W * d_cell, d_model)
        self.row_pos = nn.Embedding(MAX_H, d_model)
        self.act = nn.Embedding(N_ACTIONS, d_model)
        self.pos = nn.Embedding(seq_len(T), d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head, dropout) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.frame_head = nn.Linear(d_model, MAX_W * OBS_VOCAB)
        self.rew_head = nn.Linear(d_model, len(REWARD_VALUES))
        self.done_head = nn.Linear(d_model, 2)
        step = MAX_H + 1
        idx_act = torch.tensor([t * step + MAX_H for t in range(T)])
        idx_frame = torch.tensor([[idx_act[t].item() if r == 0 else (t + 1) * step + r - 1
                                   for r in range(MAX_H)] for t in range(T)])
        self.register_buffer("idx_act", idx_act, persistent=False)
        self.register_buffer("idx_frame", idx_frame, persistent=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, obs, act):
        """obs: [B, T+1, H, W] long, act: [B, T] long."""
        b, T = act.shape
        rows = self.row_in(self.cell(obs).flatten(3)) + self.row_pos.weight      # [B,T+1,H,d]
        a = self.act(act).unsqueeze(2)                                            # [B,T,1,d]
        x = torch.cat([rows[:, :T], a], dim=2).reshape(b, T * (MAX_H + 1), -1)
        x = torch.cat([x, rows[:, T]], dim=1) + self.pos.weight
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)
        frame = self.frame_head(h[:, self.idx_frame]).view(b, T, MAX_H, MAX_W, OBS_VOCAB)
        ha = h[:, self.idx_act]
        return frame, self.rew_head(ha), self.done_head(ha)


def compute_loss(model, batch):
    frame, rew_logits, done_logits = model(batch["obs"], batch["act"])
    mask = batch["mask"]
    tgt = batch["obs"][:, 1:]
    cell_mask = (tgt != PAD).float() * mask[:, :, None, None]
    ce = F.cross_entropy(frame.float().reshape(-1, OBS_VOCAB), tgt.reshape(-1),
                         reduction="none").view_as(cell_mask)
    l_frame = (ce * cell_mask).sum() / cell_mask.sum().clamp_min(1.0)
    n = mask.sum().clamp_min(1.0)
    l_rew = (F.cross_entropy(rew_logits.float().reshape(-1, len(REWARD_VALUES)),
                             batch["rew"].reshape(-1), reduction="none").view_as(mask)
             * mask).sum() / n
    l_done = (F.cross_entropy(done_logits.float().reshape(-1, 2), batch["done"].reshape(-1),
                              reduction="none").view_as(mask) * mask).sum() / n
    total = l_frame + l_rew + l_done
    return total, {"frame": l_frame.item(), "reward": l_rew.item(), "done": l_done.item()}


def to_device(np_batch, device):
    out = {}
    for k, v in np_batch.items():
        t = torch.from_numpy(v.astype(np.float32 if k == "mask" else np.int64))
        if device.type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        out[k] = t.to(device)
    return out


# ---------------------------------------------------------------------- #
# Schedule and checkpoints
# ---------------------------------------------------------------------- #
def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1.0 + math.cos(math.pi * p))


def raw_model(model):
    return getattr(model, "_orig_mod", model)


def make_optimizer(model, args, device):
    """AdamW with two groups (decay for matrices, none for biases/norms).
    Used for training and for every reload, so the group layout always matches."""
    params = list(raw_model(model).parameters())
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=args.lr, betas=(0.9, 0.95), fused=device.type == "cuda")


def alt_path(path):
    root, ext = os.path.splitext(path)
    return root + "_alt" + ext


def _replace_with_retry(tmp, path, tries=10):
    """os.replace is atomic, but on Windows a file that was just written can be
    locked for a moment by sync clients or antivirus scanners. Retry briefly."""
    for i in range(tries):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.1 * (i + 1))
    return False


def save_checkpoint(path, model, opt, state, args):
    """Writes to a temporary file and renames it atomically, so an interruption
    never leaves half a file. Returns the path that was actually written."""
    tmp = path + ".tmp"
    torch.save({"model": raw_model(model).state_dict(), "opt": opt.state_dict(),
                "state": dict(state), "args": vars(args)}, tmp)
    if _replace_with_retry(tmp, path):
        return path
    alt = alt_path(path)
    if _replace_with_retry(tmp, alt):
        print(f"WARNING: {path} is locked by another process; wrote {alt} instead")
        return alt
    print(f"WARNING: could not rename {tmp}; the checkpoint stays in that file")
    return tmp


def newest_checkpoint(path):
    """Newest existing file among path and its fallback name (by modification time)."""
    cands = [p for p in (path, alt_path(path)) if os.path.exists(p)]
    return max(cands, key=os.path.getmtime) if cands else path


def load_checkpoint(path, model, opt, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    raw_model(model).load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    return ck["state"]


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Acore-WM")
    p.add_argument("--smoke", action="store_true", help="tiny CPU run, prints SMOKE_OK")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="resume from out_dir/latest.pt or from the given path")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=8)
    p.add_argument("--n-head", type=int, default=8)
    p.add_argument("--d-cell", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--T", type=int, default=64, help="time steps per sequence")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=200_000, help="length of the cosine schedule")
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--keep-every", type=int, default=10_000, help="also keep step_N.pt copies")
    p.add_argument("--max-hours", type=float, default=14.0, help="hard stop (training time)")
    p.add_argument("--price-per-hour", type=float, default=0.0, help="USD per GPU hour")
    p.add_argument("--max-cost", type=float, default=30.0, help="USD; stop before exceeding")
    p.add_argument("--no-compile", action="store_true")
    a = p.parse_args(argv)
    if a.smoke:
        a.d_model, a.n_layer, a.n_head, a.d_cell, a.T = 64, 2, 4, 8, 8
        a.batch_size, a.steps, a.warmup, a.lr, a.min_lr = 16, 80, 5, 3e-3, 3e-4
        a.workers, a.no_compile, a.log_every, a.eval_every = 0, True, 10, 40
        a.ckpt_every, a.keep_every = 40, 0
        a.max_hours = min(a.max_hours, 2.5 / 60)
        if a.out_dir == "checkpoints":
            a.out_dir = "checkpoints_smoke"
    return a


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = WorldModel(args.d_model, args.n_layer, args.n_head, args.d_cell, args.T,
                       args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = make_optimizer(model, args, device)

    state = {"step": 0, "consumed": 0, "elapsed_hours": 0.0}
    latest = os.path.join(args.out_dir, "latest.pt")
    if args.resume:
        path = newest_checkpoint(latest) if args.resume == "auto" else args.resume
        if os.path.exists(path):
            state = load_checkpoint(path, model, opt, device)
            print(f"resumed from {path} at step {state['step']}")
        else:
            print(f"no checkpoint at {path}; starting fresh")

    budget_h = args.max_hours
    if args.price_per_hour > 0:
        budget_h = min(budget_h, args.max_cost / args.price_per_hour)
    remaining_h = max(0.0, budget_h - state["elapsed_hours"])
    print(f"device={device} params={n_params:,} seq_len={seq_len(args.T)}")
    if args.price_per_hour > 0:
        print(f"COST_ESTIMATE: up to {budget_h:.2f} h x ${args.price_per_hour:.2f}/h = "
              f"${budget_h * args.price_per_hour:.2f} total (max-cost ${args.max_cost:.2f}, "
              f"max-hours {args.max_hours}); {remaining_h:.2f} h left in this budget")
    else:
        print(f"no --price-per-hour given; time limit {budget_h:.2f} h "
              f"({remaining_h:.2f} h left)")
    sys.stdout.flush()

    fwd_model = model
    if not args.no_compile and device.type == "cuda":
        fwd_model = torch.compile(model)

    stop = {"flag": False}

    def on_signal(signum, _frame):
        print(f"signal {signum}: checkpointing and stopping")
        stop["flag"] = True

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            signal.signal(sig, on_signal)

    stream = TrajectoryStream(args.batch_size, args.T, base_seed=args.seed,
                              workers=args.workers, consumed=state["consumed"])
    eval_batch = to_device(generate_batch(range(EVAL_SEED_BASE, EVAL_SEED_BASE + 32), args.T),
                           device)
    log_path = os.path.join(args.out_dir, "log.csv")
    if args.smoke and not args.resume and os.path.exists(log_path):
        os.remove(log_path)                      # each smoke run gets a clean log
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "loss", "frame", "reward", "done", "lr", "eval_loss",
                      "hours", "cost_usd"])

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                              enabled=device.type == "cuda")

    def evaluate():
        fwd_model.eval()
        with torch.no_grad(), autocast():
            loss, _ = compute_loss(fwd_model, eval_batch)
        fwd_model.train()
        return loss.item()

    saved = {"step": -1, "path": latest}

    def checkpoint():
        if saved["step"] == state["step"]:
            return                               # this step is already on disk
        state["consumed"] = stream.consumed
        saved["path"] = save_checkpoint(latest, model, opt, state, args)
        saved["step"] = state["step"]
        if args.keep_every and state["step"] % args.keep_every == 0:
            save_checkpoint(os.path.join(args.out_dir, f"step_{state['step']}.pt"),
                            model, opt, state, args)

    losses, t_start, t_last = [], time.time(), time.time()
    safety_h = 0.0 if args.smoke else 3.0 / 60       # reserve time for the final checkpoint
    step_hours = 0.0
    reason = "schedule finished"
    try:
        while state["step"] < args.steps:
            if stop["flag"]:
                reason = "signal"
                break
            if state["elapsed_hours"] + 2 * step_hours + safety_h > budget_h:
                reason = "time/cost budget reached"
                break
            lr = lr_at(state["step"], args)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = to_device(stream.next(), device)
            with autocast():
                loss, parts = compute_loss(fwd_model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            now = time.time()
            step_hours = (now - t_last) / 3600
            state["elapsed_hours"] += step_hours
            t_last = now
            state["step"] += 1
            losses.append(loss.item())
            if state["step"] % args.log_every == 0 or state["step"] == 1:
                ev = evaluate() if state["step"] % args.eval_every == 0 else ""
                cost = state["elapsed_hours"] * args.price_per_hour
                print(f"step {state['step']} loss {losses[-1]:.4f} frame {parts['frame']:.4f} "
                      f"rew {parts['reward']:.4f} done {parts['done']:.4f} lr {lr:.2e} "
                      f"eval {ev if ev == '' else f'{ev:.4f}'} "
                      f"h {state['elapsed_hours']:.3f} ${cost:.2f}", flush=True)
                log.writerow([state["step"], losses[-1], parts["frame"], parts["reward"],
                              parts["done"], lr, ev, state["elapsed_hours"], cost])
                log_f.flush()
            if state["step"] % args.ckpt_every == 0:
                checkpoint()
            if args.smoke and now - t_start > 150:
                reason = "smoke time cap"
                break
    finally:
        checkpoint()
        stream.close()
        log_f.close()
    print(f"stopped at step {state['step']}: {reason}; checkpoint {saved['path']}")

    if args.smoke:
        fresh = WorldModel(args.d_model, args.n_layer, args.n_head, args.d_cell, args.T)
        fresh_opt = make_optimizer(fresh, args, torch.device("cpu"))
        restored = load_checkpoint(saved["path"], fresh, fresh_opt, torch.device("cpu"))
        assert restored["step"] == state["step"], "checkpoint round trip failed"
        k = min(5, max(1, len(losses) // 4))
        start, end = sum(losses[:k]) / k, sum(losses[-k:]) / k
        print(f"LOSS_START {start:.4f}")
        print(f"LOSS_END {end:.4f}")
        if end < start:
            print("SMOKE_OK")
        else:
            print("SMOKE_FAIL: loss did not fall")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
