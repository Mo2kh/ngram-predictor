"""
PredictorUI Module (Streamlit)

This module implements a browser-based UI for the N-gram next-word
prediction system using Streamlit. It runs alongside the CLI and
delegates all prediction logic to Predictor, NGramModel, and Normalizer.

Extra Credit (+5 points)
"""

import os
import streamlit as st
from dotenv import load_dotenv
import sys 
sys.path.append(os.getcwd())
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor


class PredictorUI:
    """
    Streamlit-based UI wrapper for the Predictor.

    Responsibilities:
    - Load environment configuration
    - Load trained model and vocabulary
    - Accept user input via the browser
    - Display top-k next-word predictions
    """

    def __init__(self):
        """
        Initialize UI, load configuration, model, and predictor.
        """
        load_dotenv(dotenv_path=os.path.join(os.getcwd(), "config/.env"))

        self.model = NGramModel(
            ngram_order=int(os.getenv("NGRAM_ORDER")),
            unk_threshold=int(os.getenv("UNK_THRESHOLD"))
        )
        self.model.load(os.getenv("MODEL"), os.getenv("VOCAB"))

        self.normalizer = Normalizer()
        self.predictor = Predictor(self.model, self.normalizer)

        self.top_k = int(os.getenv("TOP_K"))

    def render(self):
        """
        Render the Streamlit web interface.
        """
        st.set_page_config(
            page_title="N-Gram Next-Word Predictor",
            layout="centered"
        )

        st.title("📝 N‑Gram Next‑Word Predictor")
        st.write(
            "Enter a sequence of words below and the model will "
            "predict the most likely next words using an N‑gram language model."
        )

        user_input = st.text_input("Enter text:", "")

        if st.button("Predict"):
            if not user_input.strip():
                st.warning("Please enter some text.")
            else:
                predictions = self.predictor.predict_next(user_input, self.top_k)

                if predictions:
                    st.subheader("Predictions")
                    for i, word in enumerate(predictions, start=1):
                        st.write(f"{i}. **{word}**")
                else:
                    st.info("No predictions found for this input.")


# ---------------------------------------------------------
# Streamlit entry point
# ---------------------------------------------------------
def main():
    ui = PredictorUI()
    ui.render()


if __name__ == "__main__":
    main()