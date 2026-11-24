---
license: "[TODO: choose license]"
tags:
  - world-model
  - reinforcement-learning
  - model-based-rl
  - synthetic-data
  - grid-world
  - acore
library_name: pytorch
---

# Acore-WM

**Status: weights not trained yet.** This repository is a placeholder for the model card. The weights will be uploaded after the real GPU training run. [TODO: final loss, steps, hours and cost]

## What it is

Acore-WM is a small neural world model that I train next to Acore, my own symbolic reasoning core. From a history of (grid observation, action) it predicts:

- the next grid observation,
- the reward,
- the done flag

of Acore's micro-universes.

Acore itself is symbolic and has no weights. Acore-WM is part of my vertically integrated stack (Acore, my own NPU and ISA, my own CUDA-style driver). It is not AGI; it is one step on my path toward it.

## Intended use

- Research on world models for small, fully specified environments.
- Planning experiments together with Acore.

Out of scope: any real-world, safety-critical or decision-making use. The model only knows synthetic grid worlds.

## Training data

Only synthetic micro-universes, generated on the fly by `envs.py` in the training code. No external, scraped or personal data is used, and nothing is downloaded.

## Training procedure

- Code: `training/train.py` in my GitHub repository [TODO: link].
- Hardware plan: one rented H100. [TODO: actual hours and cost]
- Verification so far: a CPU smoke run only, loss 7.0234 -> 6.2045. This is a code check, not a result.

## Evaluation

[TODO: no evaluation yet]

## Limitations

- Not trained yet.
- Only synthetic grid environments.

## Author

[TODO: my full name], solo founder, Aksaray, Turkey.
