# Acore-WM training on one rented H100 (Vast.ai)

Acore is symbolic and has no weights. Acore-WM is the small neural world model I
train next to it: from the history of (grid observation, action) it predicts the
next grid, the reward and the done flag of Acore's micro-universes. Data is
generated on the fly by `envs.py`; nothing has to be downloaded.

Results of the real run do not exist yet: [TODO: final loss, steps, hours and cost after the H100 run].

## 0. Before renting: local checks (CPU)

```
pip install -r requirements.txt
python envs.py            # physics self-check against the test-suite sequences
python train.py --smoke   # tiny CPU run, prints LOSS_START / LOSS_END / SMOKE_OK
```

## 1. Choose the instance

1. On vast.ai, open "Search", filter GPU = H100 (SXM or PCIe), 1 GPU.
2. Prefer hosts with high reliability, enough CPU cores (data generation runs in
   worker processes; `--workers` defaults to cores minus 2) and at least
   [TODO: disk GB] of disk for checkpoints.
3. Image/template: an official PyTorch image with CUDA (for example the
   `pytorch/pytorch` "-cuda-...-devel" tags). PyTorch >= 2.3 is required.
4. Launch mode: SSH. Note the hourly price shown: [TODO: USD per hour of the chosen offer].

## 2. Upload the code

From my machine (port and host come from the instance's "Connect" dialog):

```
scp -P <PORT> envs.py train.py requirements.txt root@<HOST>:/workspace/acore-wm/
scp -P <PORT> -r bench root@<HOST>:/workspace/acore-wm/
```

Then on the instance:

```
cd /workspace/acore-wm
pip install -r requirements.txt
python envs.py
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 3. Start in tmux

```
tmux new -s wm
python train.py --price-per-hour <USD_PER_HOUR> --max-cost 30 --max-hours 14 2>&1 | tee -a train.log
```

Detach with `Ctrl-b d`, reattach with `tmux attach -t wm`. The script prints a
`COST_ESTIMATE` line before training: hours x price, capped by `--max-cost`
and `--max-hours`. It stops on its own before exceeding either limit and writes
a final checkpoint.

Optional GPU benchmark of the Bellman backup (same kernel as my chip), run once:

```
python bench/bench_gpu.py --out bench_results.csv
```

## 4. Resume after an interruption

Checkpoints are written atomically to `checkpoints/latest.pt` every
`--ckpt-every` steps (plus `step_N.pt` copies every `--keep-every` steps), and
on Ctrl-C / SIGTERM. The data stream position, step and elapsed training hours
are stored in the checkpoint, so the time/cost budget continues where it left off:

```
python train.py --resume --price-per-hour <USD_PER_HOUR> 2>&1 | tee -a train.log
```

If `latest.pt` is ever locked by another process (seen with sync clients on
Windows), the script writes `latest_alt.pt` instead and `--resume` picks the
newer of the two automatically.

`--resume path/to/step_N.pt` resumes from a specific file. If the instance itself
was lost, upload the last downloaded `latest.pt` into `checkpoints/` first.

## 5. Download checkpoints and logs

```
scp -P <PORT> root@<HOST>:/workspace/acore-wm/checkpoints/latest.pt .
scp -P <PORT> root@<HOST>:/workspace/acore-wm/checkpoints/log.csv .
scp -P <PORT> root@<HOST>:/workspace/acore-wm/train.log .
scp -P <PORT> root@<HOST>:/workspace/acore-wm/bench_results.csv .
```

I download `latest.pt` periodically during the run, not only at the end.

## 6. Cost control

- `--max-cost` (default 30 USD) and `--max-hours` (default 14) bound training
  time; the script reserves a few minutes for the final checkpoint.
- These limits count training time only. Setup, uploads and idle time are
  billed too, so I keep them short.
- When finished: download everything, then DESTROY the instance. A stopped
  instance still bills for storage.
- Set a credit limit / low-balance alert in the Vast.ai billing settings.
