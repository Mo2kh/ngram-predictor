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

        word_counts = Counter()

        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                word_counts.update(tokens)

        # Keep words meeting UNK threshold
        self.vocab = [
            word for word, count in word_counts.items()
            if count >= self.unk_threshold
        ]

        # Add <UNK>
        self.vocab.append("<UNK>")
        self.vocab_set = set(self.vocab)


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

        # Reset counts
        self.counts = [defaultdict(Counter) for _ in range(self.ngram_order)]

        def map_unk(word):
            return word if word in self.vocab_set else "<UNK>"

        # -------- Count n-grams --------
        with open(token_file, "r", encoding="utf-8") as f:
            for line in f:
                tokens = [map_unk(tok) for tok in line.strip().split()]
                length = len(tokens)

                for i in range(length):
                    for n in range(1, self.ngram_order + 1):
                        if i + n <= length:
                            ngram = tokens[i:i + n]
                            context = tuple(ngram[:-1])
                            target = ngram[-1]
                            self.counts[n - 1][context][target] += 1

        # -------- Compute MLE probabilities --------
        self.probs = [defaultdict(dict) for _ in range(self.ngram_order)]

        total_unigrams = sum(self.counts[0][()].values())

        for order in range(1, self.ngram_order + 1):
            for context, targets in self.counts[order - 1].items():
                denominator = (
                    total_unigrams if order == 1 else sum(targets.values())
                )
                for word, count in targets.items():
                    self.probs[order - 1][context][word] = count / denominator


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
  
        # Replace OOV context tokens with <UNK>
        context_tokens = [
            word if word in self.vocab_set else "<UNK>"
            for word in context_tokens
        ]

        for order in range(self.ngram_order, 0, -1):
            needed = order - 1
            if len(context_tokens) < needed:
                continue

            context = tuple(context_tokens[-needed:]) if needed > 0 else ()

            if context in self.probs[order - 1]:
                return self.probs[order - 1][context]

        return {}

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

        output = {}

        for order in range(1, self.ngram_order + 1):
            key = f"{order}gram"
            output[key] = {}

            for context, targets in self.probs[order - 1].items():
                context_str = " ".join(context)
                output[key][context_str] = targets

        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


    def save_vocab(self, vocab_path):
        """
        Save vocabulary list to vocab.json.

        Parameters:
            vocab_path (str): Path to vocab.json.

        Returns:
            None
        """

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2)


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

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.vocab_set = set(self.vocab)

        with open(model_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)

        self.probs = [defaultdict(dict) for _ in range(self.ngram_order)]

        for order in range(1, self.ngram_order + 1):
            key = f"{order}gram"
            for context_str, targets in model_data.get(key, {}).items():
                context = tuple(context_str.split()) if context_str else ()
                self.probs[order - 1][context] = targets
