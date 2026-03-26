"""
NGramModel Module

This module implements an n-gram language model supporting:
- Vocabulary construction with <UNK> replacement
- Counting all n-grams from order 1 up to NGRAM_ORDER (read from config)
- Computing MLE probabilities at all orders
- Stupid backoff inference: highest-order match first, then fall back
- Saving and loading model.json and vocab.json

This class is the ONLY component that builds probability tables AND
performs backoff lookup. Predictor calls lookup() directly.

No smoothing, discounting, or Katz backoff redistribution is used.
This is a simplified “stupid backoff” as required by the project.
"""

import json
from collections import defaultdict, Counter
import os


class NGramModel:
    """
    NGramModel builds and queries an n-gram language model.

    Responsibilities:
    - Build vocabulary with <UNK> based on UNK_THRESHOLD
    - Count n-grams at all orders (1..NGRAM_ORDER)
    - Compute MLE probabilities
    - Provide lookup() with stupid backoff
    - Save/load model files
    """



    # ---------------------------------------------------------------------
    # 01 — Build Vocabulary
    # ---------------------------------------------------------------------
    def build_vocab(self, token_file):
        """
        Build the vocabulary from the tokenized training corpus.

        Parameters:
            token_file (str): Path to train_tokens.txt.

        Returns:
            None
        """


    # ---------------------------------------------------------------------
    # 02 & 03 — Build Counts + Compute MLE Probabilities
    # ---------------------------------------------------------------------
    def build_counts_and_probabilities(self, token_file):
        """
        Count all n-grams (1..NGRAM_ORDER) and compute MLE probabilities.

        Parameters:
            token_file (str): Path to train_tokens.txt.

        Returns:
            None
        """


    # ---------------------------------------------------------------------
    # 04 — Lookup (Stupid Backoff)
    # ---------------------------------------------------------------------
    def lookup(self, context_tokens):
        """
        Perform backoff lookup and return highest-order matching distribution.

        Parameters:
            context_tokens (list[str]):
                The context words from predictor (already normalized).

        Returns:
            dict: {word: probability} for the highest-order match found,
                  or {} if nothing matches.
        """
       
    # ---------------------------------------------------------------------
    # 05 — Save Model & Vocab
    # ---------------------------------------------------------------------
    def save_model(self, model_path):
        """
        Save all probability tables to model.json.

        Parameters:
            model_path (str): Path to output model.json.

        Returns:
            None
        """


    def save_vocab(self, vocab_path):
        """
        Save vocabulary list to vocab.json.

        Parameters:
            vocab_path (str): Path to vocab.json.

        Returns:
            None
        """
 

    # ---------------------------------------------------------------------
    # Load Model
    # ---------------------------------------------------------------------
    def load(self, model_path, vocab_path):
        """
        Load model.json and vocab.json into memory.

        Parameters:
            model_path (str): Path to saved model.json.
            vocab_path (str): Path to saved vocab.json.

        Returns:
            None
        """
