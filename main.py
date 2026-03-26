import argparse
import os
import sys
from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NGramModel
from src.inference.predictor import Predictor


# ---------------------------------------------------------
# Utility: read all config from .env
# ---------------------------------------------------------
def load_config():
    load_dotenv("config/.env")

    config = {
        "TRAIN_RAW_DIR": os.getenv("TRAIN_RAW_DIR"),
        "TRAIN_TOKENS": os.getenv("TRAIN_TOKENS"),
        "MODEL_PATH": os.getenv("MODEL_P"),
        "VOCAB_PATH": os.getenv("VOCAB"),
        "NGRAM_ORDER": int(os.getenv("NGRAM_ORDER", 4)),
        "UNK_THRESHOLD": int(os.getenv("UNK_THRESHOLD", 1)),
        "TOP_K": int(os.getenv("TOP_K", 5)),
    }

    # Basic sanity check
    for key, value in config.items():
        if value is None:
            print(f"[ERROR] Missing config for {key} in .env")
            sys.exit(1)

    return config


# ---------------------------------------------------------
# Module 1: Data Preparation
# ---------------------------------------------------------


def run_dataprep(config):
    print("=== Running Data Preparation ===")
    normalizer = Normalizer()

    # -----------------------------------------------------------
    # 01 Load raw text from all training .txt files
    #     (strip_gutenberg applied per file inside load())
    # -----------------------------------------------------------
    raw_text = normalizer.load(os.path.join(os.getcwd(), config["TRAIN_RAW_DIR"])
 )

    # -----------------------------------------------------------
    # 02 Basic cleanup (punctuation is intentionally preserved!)
    # -----------------------------------------------------------
    stage1 = normalizer.lowercase(raw_text)
    stage1 = normalizer.remove_numbers(stage1)
    stage1 = normalizer.remove_whitespace(stage1)

    # -----------------------------------------------------------
    # 03 Sentence tokenization (requires punctuation!)
    # -----------------------------------------------------------
    sentences = normalizer.sentence_tokenize(stage1)

    # -----------------------------------------------------------
    # 04 Per-sentence normalization
    #      remove punctuation AFTER sentence boundaries are known
    #      then tokenize into words
    # -----------------------------------------------------------
    tokenized_sentences = []
    for s in sentences:
        s_clean = normalizer.remove_punctuation(s)
        s_clean = normalizer.remove_whitespace(s_clean)  # clean again after punctuation removal
        tokens = normalizer.word_tokenize(s_clean)
        if tokens:
            tokenized_sentences.append(tokens)

    # -----------------------------------------------------------
    # 05 Save tokenized sentences to train_tokens.txt
    # -----------------------------------------------------------
    normalizer.save(tokenized_sentences,os.path.join(os.getcwd(),config["TRAIN_TOKENS"]))

    print(f"[OK] Saved tokenized training data → {os.path.join(os.getcwd(),config['TRAIN_TOKENS'])}")



    
# ---------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":
    config = load_config()
    print(os.path.join(os.getcwd(), config["TRAIN_RAW_DIR"]))
    run_dataprep(config)
