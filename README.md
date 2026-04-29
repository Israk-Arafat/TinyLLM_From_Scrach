# TinyLLM From Scratch

Training a ~400M-parameter transformer language model from scratch on the [SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) dataset.

## Pipeline

1. **Data loading** — stream SlimPajama without downloading the full dataset
2. **Cleaning** — filter empty/short/junk samples
3. **Tokenization** — convert text to token IDs with a pretrained tokenizer
4. **Packing** — pack tokens into fixed 2048-token chunks
5. **Training** — next-token prediction with a decoder-only transformer
6. **Evaluation** — track validation loss during training
7. **Generation** — prompt the trained model to continue text

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

### Cell 4 — Plot training curves (from saved CSV)
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/content/drive/MyDrive/tinyllm_checkpoints/../tinyllm_logs/training_log.csv")
# or: df = pd.read_csv("logs/training_log.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(df["opt_step"], df["train_loss"], label="train")
axes[0].plot(df.dropna(subset=["val_loss"])["opt_step"],
             df.dropna(subset=["val_loss"])["val_loss"], label="val")
axes[0].set(title="Loss", xlabel="step", ylabel="cross-entropy")
axes[0].legend()

axes[1].plot(df["opt_step"], df["lr"])
axes[1].set(title="Learning Rate Schedule", xlabel="step", ylabel="lr")

axes[2].plot(df["opt_step"], df["grad_norm"])
axes[2].set(title="Gradient Norm", xlabel="step", ylabel="‖g‖")

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
```