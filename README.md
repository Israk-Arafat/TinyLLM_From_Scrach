# TinyLLM From Scratch

Training a ~300M-parameter transformer language model from scratch on the [SlimPajama-6B](https://huggingface.co/datasets/DKYoon/SlimPajama-6B) dataset.

## Pipeline

1. **Data loading** — stream SlimPajama without downloading the full dataset
2. **Cleaning** — filter empty/short/junk samples
3. **Tokenization** — convert text to token IDs with a pretrained tokenizer
4. **Packing** — pack tokens into fixed 512-token chunks
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
python scripts/train.py --config configs/train_config.yaml

# Generate
python scripts/generate.py --prompt "Once upon a time"
```

# For Google Colab
import os
if os.path.exists("TinyLLM_From_Scrach"):
    %cd TinyLLM_From_Scrach
    !git pull
else:
    !git clone https://github.com/Israk-Arafat/TinyLLM_From_Scrach.git
    %cd TinyLLM_From_Scrach

!pip install -r requirements.txt -q

# Add HF_TOKEN in .env file
%%writefile .env
HF_TOKEN=hf_your_token_here

# train, pointing checkpoints at Drive
!python scripts/train.py --checkpoint-dir /content/drive/MyDrive/tinyllm_checkpoints