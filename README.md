# N‑Gram Next‑Word Predictor

This project implements a **next‑word prediction system** using a classical **N‑gram language model** with **Maximum Likelihood Estimation (MLE)** and **simple backoff**. It processes raw text files (such as Project Gutenberg novels), learns statistical word‑sequence patterns, and predicts the most likely next words given user input through an interactive command‑line interface.

The project demonstrates the complete NLP workflow, including text normalization, N‑gram model construction, probability estimation, and interactive inference. The focus is on correctness, modular design, and clarity rather than production‑level optimization.

---

## Requirements

- **Python**: 3.9 or later (recommended via Anaconda / Conda)
- **Dependencies**: Install all required libraries using `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ngram-predictor.git
cd ngram-predictor
```

### 2. Create and Activate an Anaconda Environment

```bash
conda create -n ngram-env python=3.9
conda activate ngram-env
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create the file:

```
config/.env
```

Populate it with the following variables:

```env
TRAIN_RAW_DIR=data/raw/train
TRAIN_TOKENS=data/processed/train_tokens.txt
MODEL_P=data/model/model.json
VOCAB=data/model/vocab.json
NGRAM_ORDER=4
UNK_THRESHOLD=2
TOP_K=5
```

> **Important:** The `.env` file should not be committed to GitHub and must be listed in `.gitignore`.

### 5. Add Raw Training Data

Place raw `.txt` files into:

```
data/raw/train/
```

### 6. Download Required NLTK Resources

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

---

## Usage

### Data Preparation

```bash
python main.py --step dataprep
```

### Model Training

```bash
python main.py --step model
```

### Inference (CLI)

```bash
python main.py --step inference
```

### Full Pipeline

```bash
python main.py --step all
```

---

## Project Structure

```
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/train/
│   ├── processed/train_tokens.txt
│   └── model/model.json, vocab.json
├── src/
│   ├── data_prep/normalizer.py
│   ├── model/ngram_model.py
│   └── inference/predictor.py
├── main.py
├── requirements.txt
└── README.md
```

---

## Notes

- Normalization is centralized in `Normalizer.normalize()`.
- Backoff logic lives exclusively in `NGramModel.lookup()`.
- No advanced smoothing techniques are implemented.
