"""
Predictor Module

This module defines the Predictor class responsible for performing inference
on a trained NGramModel. It accepts a pre-loaded NGramModel instance and a
Normalizer instance, normalizes user input, maps out-of-vocabulary context
words to <UNK>, performs backoff lookup through the model, and returns the
top-k next-word predictions sorted by probability.

This module does NOT build or modify the model — it uses the pre-trained
model passed in via the constructor.
"""

import os
from dotenv import load_dotenv



class Predictor:
    """
    Predictor orchestrates the inference pipeline:

    1. Normalize raw input text using the provided Normalizer.
    2. Extract context of length NGRAM_ORDER - 1.
    3. Map OOV context words to <UNK>.
    4. Call NGramModel.lookup() to perform backoff probability lookup.
    5. Sort candidate words by probability and return the top-k.

    This class does not perform any training or file loading.
    """

    def __init__(self, model, normalizer):
        """
        Initialize Predictor with a pre-loaded model and Normalizer.

        Parameters:
            model (NGramModel): A trained and loaded n-gram model.
            normalizer (Normalizer): A Normalizer instance for text cleaning.

        Returns:
            None
        """

        self.model = model
        self.normalizer = normalizer

        # Load environment configuration
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env"))

        self.ngram_order = int(os.getenv("NGRAM_ORDER"))
        self.top_k = int(os.getenv("TOP_K"))


    # ----------------------------------------------------------
    # Step 01 — Normalize input & extract context
    # ----------------------------------------------------------
    def normalize(self, text):
        """
        Normalize input text using Normalizer.normalize() and extract
        the last NGRAM_ORDER − 1 tokens as context.

        Parameters:
            text (str): Raw text entered by the user.

        Returns:
            list[str]: The extracted context tokens.
        """

        normalized = self.normalizer.normalize(text)
        tokens = normalized.split()

        context_size = self.ngram_order - 1
        if context_size <= 0:
            return []

        return tokens[-context_size:]


    # ----------------------------------------------------------
    # Step 02 — Map OOV words to <UNK>
    # ----------------------------------------------------------
    def map_oov(self, context):
        """
        Replace out-of-vocabulary words with <UNK>.

        Parameters:
            context (list[str]): Context tokens extracted from input.

        Returns:
            list[str]: Context tokens with OOV mapped to <UNK>.
        """

        return [
            token if token in self.model.vocab_set else "<UNK>"
            for token in context
        ]


    # ----------------------------------------------------------
    # Step 03 & 04 — Predict next words
    # ----------------------------------------------------------
    def predict_next(self, text, k=None):
        """
        Predict the top-k next-word candidates using backoff lookup.

        Parameters:
            text (str): Raw user input text.
            k (int, optional): Override TOP_K value from config.

        Returns:
            list[str]: Top-k predicted next words sorted by probability.
        """

        if k is None:
            k = self.top_k

        # 1. Normalize input & extract context
        context = self.normalize(text)

        # 2. Map OOV tokens
        context = self.map_oov(context)

        # 3. Backoff lookup (handled by the model)
        candidates = self.model.lookup(context)

        if not candidates:
            return []

        # 4. Rank candidates by probability (descending)
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        return [word for word, _ in ranked[:k]]

def main():
    """
    Standalone test runner for Predictor.

    This function:
    - Loads config from config/.env
    - Loads trained model and vocabulary
    - Instantiates Normalizer and Predictor
    - Runs an interactive prediction loop

    Run from project root:
        python src/inference/predictor.py
    """

 
    import os
    from dotenv import load_dotenv
    import sys 
    sys.path.append(os.getcwd())
    from src.model.ngram_model import NGramModel
    from src.data_prep.normalizer import Normalizer

    # -------------------------------------------------
    # Load environment variables
    # -------------------------------------------------
    load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env"))

    print("=== Predictor Standalone CLI ===")
    print("Type text and press Enter. Type 'quit' to exit.\n")

    # -------------------------------------------------
    # Load model & normalizer
    # -------------------------------------------------
    model = NGramModel(
        ngram_order=int(os.getenv("NGRAM_ORDER")),
        unk_threshold=int(os.getenv("UNK_THRESHOLD"))
    )
    model.load(os.getenv("MODEL_P"), os.getenv("VOCAB"))

    normalizer = Normalizer()
    predictor = Predictor(model, normalizer)

    # -------------------------------------------------
    # Interactive loop
    # -------------------------------------------------
    try:
        while True:
            text = input("> ").strip()

            if text.lower() in {"quit", "exit"}:
                print("Goodbye.")
                break

            predictions = predictor.predict_next(text)
            print("Predictions:", predictions)

    except KeyboardInterrupt:
        print("\nGoodbye.")


if __name__ == "__main__":
    main()