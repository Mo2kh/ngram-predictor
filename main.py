import os
import argparse
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel


# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env"))


# ---------------------------------------------------------
# Module 1 — Data Preparation
# ---------------------------------------------------------
def dataprep():
    """
    Run the data preparation pipeline:
    - Load raw text
    - Sentence tokenize
    - Normalize each sentence
    - Word tokenize
    - Save train_tokens.txt
    """
    normalizer = Normalizer()

    train_raw_dir = os.getenv("TRAIN_RAW_DIR")
    train_tokens = os.getenv("TRAIN_TOKENS")

    print("=== Running Data Preparation ===")

    # 1. Load raw text (Gutenberg stripped per file)
    raw_text = normalizer.load(train_raw_dir)

    # 2. Sentence tokenize FIRST
    sentences = normalizer.sentence_tokenize(raw_text)

    # 3. Normalize each sentence and tokenize words
    tokenized_sentences = []
    for s in sentences:
        normalized = normalizer.normalize(s)
        tokens = normalizer.word_tokenize(normalized)
        if tokens:
            tokenized_sentences.append(tokens)

    # 4. Save output
    normalizer.save(tokenized_sentences, train_tokens)

    print(f" train_tokens.txt saved to {train_tokens}")


# ---------------------------------------------------------
# Module 2 — Model Training
# ---------------------------------------------------------
def model():
    """
    Run the model training pipeline:
    - Build vocabulary
    - Build n-gram counts & probabilities
    - Save model.json and vocab.json
    """
    print("=== Running Model Training ===")

    model = NGramModel(
        ngram_order=int(os.getenv("NGRAM_ORDER")),
        unk_threshold=int(os.getenv("UNK_THRESHOLD"))
    )

    train_tokens = os.getenv("TRAIN_TOKENS")
    model_path = os.getenv("MODEL_P")
    vocab_path = os.getenv("VOCAB")

    # 1. Build vocabulary
    model.build_vocab(train_tokens)

    # 2. Build counts and probabilities
    model.build_counts_and_probabilities(train_tokens)

    # 3. Save model and vocabulary
    model.save_model(model_path)
    model.save_vocab(vocab_path)

    print(f" model.json saved to {model_path}")
    print(f" vocab.json saved to {vocab_path}")


# ---------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step")
    args = parser.parse_args()

    if args.step == "dataprep":
        dataprep()
    elif args.step == "model":
        model()
    else:
        print("Usage: python main.py --step {dataprep|model}")