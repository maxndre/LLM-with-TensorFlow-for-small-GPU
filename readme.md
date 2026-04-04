# 🧠 MM1: Custom GPT Architecture from Scratch

> **The final iteration of a 3-month R&D project focused on building, training, and optimizing a Generative Pre-trained Transformer (GPT) entirely from scratch using TensorFlow and Keras, specifically engineered for highly constrained hardware.**

## 📌 Project Overview

MM1 is a custom-built, decoder-only Transformer model. Instead of relying on pre-packaged high-level NLP libraries, this project implements the core mechanics of modern Large Language Models (LLMs) from the ground up. 

**The primary constraint and challenge of this project was hardware limitation.** The entire architecture, training loop, and data pipeline were aggressively optimized to run on an older, low-end GPU setup:
* **GPU:** NVIDIA Quadro M1000M (GM107GLM)
* **CPU:** Intel® Core™ i7-6820HQ @ 2.70GHz × 8
* **Environment:** CUDA 11.8 / TensorFlow 2.12

## ⚖️ Trade-offs & Engineering Choices

Because of the strict VRAM limitations, several architectural trade-offs had to be made:
* **Dataset:** The model was trained on the **French Wikipedia**.
* **Vocabulary Arbitrage:** To prevent Out-Of-Memory (OOM) errors, the vocabulary size was heavily capped. Training on a rich, highly inflected language like French with a small vocabulary inevitably leads to a high frequency of unknown (`[UNK]`) tokens. 
* **Context Window:** The sequence length (`maxlen`) is limited to 128 tokens to keep the attention matrices small enough to fit in VRAM.

To counter the `[UNK]` issue, a **dynamic filtering algorithm** was built into the data pipeline to automatically reject training sequences containing an excessive number of unreadable tokens, ensuring the model still learns high-quality semantic relationships.

## 🏗️ Model Architecture

* **Vocabulary Size:** Configurable (~3000 tokens)
* **Context Window (Maxlen):** 128 tokens
* **Embedding Dimension:** 256
* **Attention Heads:** 4
* **Feed-Forward Dimension:** 1024
* **Transformer Layers:** 8

## ✨ Key Technical Features

* **Custom Transformer Blocks:** Fully implemented Multi-Head Attention and Feed-Forward networks with Layer Normalization and Dropout for robust learning.
* **Strict Causal Masking:** Engineered dynamic causal masks (`tf.linalg.band_part`) to strictly prevent the model from attending to future tokens during training.
* **Smart Dataset Filtering:** Custom data pipeline that cleans sequences and manages the high `[UNK]` token rate.
* **Resilient Training Loop:** Built-in CSV-based logging and checkpointing. The model automatically resumes training at the exact epoch and learning rate where it left off.
* **Modular Engineering:** Codebase cleanly separated into model architecture, data processing, training, and inference scripts for high maintainability.

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the required dependencies. *Note: Strict adherence to TF 2.12 is required for compatibility with the CUDA 11.8 setup.*

```bash
git clone [https://github.com/YourUsername/MM1.git](https://github.com/YourUsername/MM1.git)
cd MM1
pip install -r requirements.txt
```

### 2. Project Structure
Ensure you have your raw text files and vocabulary set up correctly before training:

`data_txt/`: Place your raw `.txt` Wikipedia chunks here.

`VOCAB/`: Place your compiled `vocab.txt` here.

### 3. Training the Model
Launch the training script. You can customize hyperparameters directly via command-line arguments:

``` Bash
python3 train.py --epochs 10 --batch_size 16 --learning_rate 1e-4
```

To force the creation of a brand new model instead of loading the latest checkpoint, append the ```--creer_model``` flag.

### 4. Text Generation (Inference)
Once the model has trained and saved its weights to the CHECKPOINTS_MM1 directory, you can generate text. The inference script includes logic to artificially suppress the generation of `[UNK]` tokens for readable output.

```Bash
python3 generate.py --prompt "La capitale de la France est" --length 50 --temperature 0.8 
```

Built with passion, TensorFlow 2.12, and Keras-NLP.


