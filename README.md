# TinyLLM From Scratch

A 373-million parameter decoder-only transformer language model trained from scratch on the [SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) dataset. The goal was to build and train every component — data pipeline, architecture, training loop — without relying on any existing model library.

---

## What We Built

A GPT-style language model capable of generating coherent text, trained entirely from scratch. No pretrained weights, no high-level wrappers like HuggingFace Transformers — just raw PyTorch.

---

## Dataset

**SlimPajama-6B** — a cleaned, deduplicated subset of the RedPajama corpus containing roughly 6 billion tokens drawn from web pages, books, code, Wikipedia, and academic papers. It provides broad general language coverage while being small enough to train on a single GPU.

- Tokenizer: `cl100k_base` (used by GPT-4), vocabulary size 100,277
- Sequence length: 2,048 tokens per chunk
- Sequences are packed end-to-end with no padding to maximise GPU utilisation

---

## Model Architecture

A decoder-only transformer (LLaMA-style) with the following design:

| Component           | Choice      | Detail                              |
| ------------------- | ----------- | ----------------------------------- |
| Parameters          | 373M        |                                     |
| Layers              | 24          |                                     |
| Model dimension     | 1,024       |                                     |
| Attention heads     | 16 Q / 4 KV | Grouped Query Attention (GQA)       |
| Feed-forward        | SwiGLU      | d_ff = 2,816                        |
| Positional encoding | RoPE        | θ = 500,000                         |
| Normalisation       | RMSNorm     | pre-norm on every block             |
| Weight tying        | Yes         | embedding and LM head share weights |

**Key architectural choices:**

- **Grouped Query Attention (GQA)** — 4 KV heads shared across 16 Q heads, reducing KV cache memory without hurting quality
- **SwiGLU** — outperforms ReLU/GELU feed-forwards at this scale
- **RoPE** — relative positional encoding that generalises to longer sequences
- **RMSNorm** — faster than LayerNorm with equivalent stability

---

## Training

| Setting                | Value                                    |
| ---------------------- | ---------------------------------------- |
| Hardware               | NVIDIA A100 80GB (Google Colab Pro)      |
| Duration               | ~5 days                                  |
| Steps completed        | 15,000 of 25,000                         |
| Final train loss       | < 3.13                                   |
| Batch size (effective) | 144 (48 micro × 3 grad accum)            |
| Tokens per step        | ~295,000                                 |
| Total tokens seen      | ~4.4 billion                             |
| Precision              | bfloat16 (AMP)                           |
| Optimizer              | AdamW (lr=3e-4, weight decay=0.1)        |
| LR schedule            | Linear warmup (500 steps) → cosine decay |
| Gradient clipping      | 1.0                                      |

**Training techniques used to fit within 80GB VRAM:**

- **Gradient checkpointing** — recomputes activations during backward pass instead of storing them; reduced peak VRAM from ~79GB to ~50GB
- **Chunked cross-entropy** — projects hidden states to logits 1,024 tokens at a time, avoiding a ~36GB allocation spike from the full [batch × seq × vocab] tensor
- **`PYTORCH_ALLOC_CONF=expandable_segments:True`** — prevents allocator fragmentation

Training loss dropped from ~10.7 at initialisation to below 3.13 by step 15,000, giving a perplexity below 23 — lower than GPT-2 (117M parameters) despite being trained from scratch.

---

## Pipeline

1. **Data loading** — downloads and caches SlimPajama locally for fast repeated access
2. **Cleaning** — filters empty, short, and malformed samples
3. **Tokenization** — encodes text with cl100k_base tokenizer
4. **Packing** — concatenates and splits token sequences into 2048-token chunks
5. **Training** — next-token prediction with cross-entropy loss
6. **Evaluation** — validation loss tracked every 500 steps
7. **Generation** — autoregressive sampling with temperature, top-k, and top-p

---

## Project Structure

```
├── configs/          # YAML configs for model, training, data
├── data/             # Data loading, cleaning, tokenizing, packing
├── model/            # Transformer architecture
├── training/         # Trainer, optimizer, scheduler
├── evaluation/       # Validation metrics
├── generation/       # Text generation utilities
├── scripts/          # CLI entry points
└── tests/            # Unit tests
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Add HF_TOKEN in .env file
HF_TOKEN=hf_your_token_here

# Train
python scripts/train.py --train-config configs/train_config.yaml

# Resume from checkpoint
python scripts/train.py --train-config configs/train_config.yaml --resume checkpoints/step_10000.pt

# Generate
python scripts/generate.py --checkpoint checkpoints/final.pt --prompt "Once upon a time"
```

---

## Google Colab

### Cell 1 — Mount Drive and clone repo

```python
from google.colab import drive
drive.mount('/content/drive')

import os
if os.path.exists("TinyLLM_From_Scrach"):
    %cd TinyLLM_From_Scrach
    !git pull
else:
    !git clone https://github.com/Israk-Arafat/TinyLLM_From_Scrach.git
    %cd TinyLLM_From_Scrach

!pip install -r requirements.txt -q
```

### Cell 2 — Set your HuggingFace token

```python
%%writefile .env
HF_TOKEN=hf_your_token_here
```

### Cell 3a — Start fresh training

```python
!python scripts/train.py \
    --train-config configs/train_config.yaml \
    --checkpoint-dir /content/drive/MyDrive/tinyllm_checkpoints
```

### Cell 3b — Resume after a session disconnect

```python
import glob, os

ckpt_dir = "/content/drive/MyDrive/tinyllm_checkpoints"

# Auto-pick the latest step checkpoint
step_ckpts = sorted(glob.glob(f"{ckpt_dir}/step_*.pt"),
                    key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
latest = step_ckpts[-1]
print(f"Resuming from: {latest}")

!python scripts/train.py \
    --train-config configs/train_config.yaml \
    --checkpoint-dir /content/drive/MyDrive/tinyllm_checkpoints \
    --resume {latest}
```
