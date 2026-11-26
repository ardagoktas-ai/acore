# Acore: Predictive Cognitive Core

[![Live Demo & API](https://shields.io🤗%20Hugging%20Face-Live%20Demo%20%26%20API-yellow)](https://huggingface.co)
[![Model Weights](https://shields.io🤗%20Model-acore--v1-blue)](https://huggingface.co)

### 🚀 Try Live Interactive Demo & API (No Installation)
👉 **[Launch Acore Live Demo & API on Hugging Face Spaces](https://huggingface.co)**

**Acore** is a 27.58M-parameter predictive cognitive core built from scratch by **Arda Göktaş** (`ardagoktas-ai`) to learn abstract state transitions, reward dynamics, and terminal conditions across diverse environments for zero-shot rollout planning.

- **Model Weights & H100 Benchmarks:** [ardagoktas-ai/acore-v1 on Hugging Face](https://huggingface.coardagoktas-ai/acore-v1)
- **Key Metrics:** `27,581,846` parameters | `54,050` H100 steps | `0.0783` final prediction loss | `0.4611` unseen-environment generalization loss | `2,623.113 GFLOP/s` H100 throughput

## Repository Structure
- `acore_v1_9.py` & `test_v1.py`: Core predictive engine, custom CUDA/kernel benchmarks, and verification suite
- `yonetici.py`: Autonomous experiment orchestration and self-directed run management
- `night_run/`: Multi-stage training pipeline, generalization evaluation, and rollout planning
- `npu_driver/`: Low-level C++ / Verilog / Python hardware accelerator driver stack
