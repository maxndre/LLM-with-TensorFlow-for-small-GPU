# 🧠 MM1: a GPT trained from scratch on a 2015 laptop GPU

> **The final iteration of a 3-month R&D project: building, training and optimizing a decoder-only GPT in TensorFlow/Keras with no pretrained weights, on hardware that had no business running it.**

## 📌 Project Overview

MM1 is a decoder-only transformer assembled from Keras primitives — no pretrained weights, no LLM framework, no fine-tuning. The transformer blocks, the causal masking, the data pipeline and the training loop are written here; attention and layer normalization come from `tf.keras.layers`.

**The interesting part of this project is not the architecture, it is the budget.** Every design choice below is downstream of what fits in the VRAM of a ten-year-old mobile GPU:

* **GPU:** NVIDIA Quadro M1000M (GM107GLM)
* **CPU:** Intel® Core™ i7-6820HQ @ 2.70GHz × 8
* **Environment:** CUDA 11.8 / TensorFlow 2.12

## ⚖️ Trade-offs & Engineering Choices

The VRAM ceiling forces a single budget to be split four ways — vocabulary size, embedding width, depth, and context length. Spend it in one place and you pay for it in another.

* **Context window.** `maxlen` is capped at **128 tokens**. Attention cost grows with the square of the sequence length, so the context window is the most expensive axis to buy and the first one cut.
* **Vocabulary arbitrage.** The vocabulary is capped at **~3000 tokens** to keep the embedding and output projection small enough to fit. French is a heavily inflected language, so a small vocabulary produces a high rate of unknown (`[UNK]`) tokens — the cost of that choice, paid in data quality.
* **Countermeasure.** A **dynamic filtering algorithm** in the data pipeline rejects training sequences containing an excessive proportion of `[UNK]`, so the model still learns from readable text rather than from noise.
* **Dataset:** French Wikipedia.

## 🏗️ Model Architecture

| | |
|---|---|
| Vocabulary size | ~3000 tokens |
| Context window (`maxlen`) | 128 tokens |
| Embedding dimension | 256 |
| Attention heads | 4 |
| Feed-forward dimension | 1024 |
| Transformer layers | 8 |

## ✨ What is actually implemented here

* **Transformer block wiring** (`model.py`) — residual connections around both sub-layers, post-norm placement, dropout before the add. Built on `layers.MultiHeadAttention` and `layers.LayerNormalization`; the block structure around them is written by hand.
* **Strict causal masking** — a dynamic lower-triangular mask via `tf.linalg.band_part`, reshaped and passed into every block, so the model cannot attend to future tokens.
* **Data pipeline** — sequence cleaning, the `[UNK]` filtering described above, shuffling and `prefetch(AUTOTUNE)`.
* **Resilient training loop** — CSV logging and checkpointing; training resumes at the exact epoch and learning rate it stopped at, which matters when a run takes days on this hardware.
* **Memory management** — `set_memory_growth` so TensorFlow does not pre-allocate the whole card.
* **Inference** (`generate.py`) — temperature sampling, with logic to suppress `[UNK]` generation for readable output.

## 🚀 Getting Started

### 1. Installation

*Note: TF 2.12 is required for compatibility with the CUDA 11.8 setup.*

```bash
git clone https://github.com/maxndre/LLM-with-TensorFlow-for-small-GPU.git
cd LLM-with-TensorFlow-for-small-GPU
pip install -r requirements.txt
```

### 2. Project structure

`data_txt/`: raw `.txt` Wikipedia chunks.
`VOCAB/`: the compiled `vocab.txt`.

### 3. Training

```bash
python3 train.py --epochs 10 --batch_size 16 --learning_rate 1e-4
```

Append `--creer_model` to force a fresh model instead of loading the latest checkpoint.

### 4. Text generation

```bash
python3 generate.py --prompt "La capitale de la France est" --length 50 --temperature 0.8
```

---

Built with TensorFlow 2.12, Keras-NLP, and a laptop that got very warm.
