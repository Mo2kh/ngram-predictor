"""
Normalizer Module

This module provides the Normalizer class, responsible for:
- Loading raw text files from disk
- Cleaning and normalizing their content
- Tokenizing text into sentences and words
- Saving the processed corpus into tokenized output files

The same Normalizer class is used in two contexts:
1. Module 1 (Data Prep): Process complete raw books into normalized, tokenized text.
2. Module 3 (Inference): Normalize a single text input consistently for lookup.

All normalization behavior is centralized in normalize(text), ensuring
consistent preprocessing across the entire project.
"""

import os
import re
import string


class Normalizer:
    """
    A text normalization utility that prepares corpus data for N-gram modeling.

    Responsibilities include:
    - Loading raw Gutenberg text files
    - Removing boilerplate headers/footers
    - Normalizing text (lowercase, remove punctuation, numbers, whitespace)
    - Sentence tokenization
    - Word tokenization
    - Saving processed sentences to disk

    This class is used both during:
    - Data preparation on full book files
    - Inference-time normalization of individual text inputs
    """

    # ------------ LOADING ------------ #

    def load(self,folder_path):
        """
        Load all .txt files from a given directory.

        Parameters:
            folder_path (str): Path to a folder containing .txt files.

        Returns:
            str: Concatenated text content of all .txt files in the folder.
        """
        
        text_all = []

        
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, "r", encoding="utf-8") as f:
                    raw = f.read()

                # Clean EACH file before concatenation
                cleaned =self.strip_gutenberg(raw)
                text_all.append(cleaned)


        return "\n".join(text_all)
 

    # ------------ CLEANING ------------ #

    def strip_gutenberg(self, text):
        """
        Remove Project Gutenberg header and footer lines.

        Parameters:
            text (str): Full raw book text.

        Returns:
            str: Text with header & footer removed.
        """
        
        start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"
        end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*"
        if start_pattern not in text or end_pattern not in text: 
            return text
        # Remove before START
        text = re.sub(f"(?s).*?{start_pattern}", "", text)

        # Remove from END onward
        text = re.sub(f"{end_pattern}(?s).*", "", text)

        return text



    def lowercase(self,text):
        """
        Convert all text to lowercase.

        Parameters:
            text (str): Input text.

        Returns:
            str: Lowercased text.
        """
        return text.lower()
        

    def remove_punctuation(self,text):
        """
        Remove punctuation characters.

        Parameters:
            text (str): Input text.

        Returns:
            str: Text without punctuation.
        """
        return text.translate(str.maketrans("", "", string.punctuation + "“”‘’"))

    def remove_numbers(self,text):
        """
        Remove all digits.

        Parameters:
            text (str): Input text.

        Returns:
            str: Text without numbers.
        """
        return re.sub(r"\d+", "", text)

    def remove_whitespace(self,text):
        """
        Normalize whitespace: remove extra spaces and blank lines.

        Parameters:
            text (str): Input text.

        Returns:
            str: Cleaned text with single spacing.
        """
        
        text = re.sub(r"\s+", " ", text)
        return text.strip()


    # ------------ NORMALIZATION PIPELINE ------------ #

    def normalize(self,text):
        """
        Apply all normalization steps in the correct order:

        lowercase → remove punctuation → remove numbers → remove whitespace

        Parameters:
            text (str): Raw or partially cleaned text.

        Returns:
            str: Fully normalized text.
        """
       
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.remove_numbers(text)
        text = self.remove_whitespace(text)
        return text

        

    # ------------ TOKENIZATION ------------ #

    def sentence_tokenize(self,text):
        """
        Split text into sentences.

        Parameters:
            text (str): Normalized text.

        Returns:
            list[str]: List of sentence strings.
        """
       
        sentences = re.split(r"[.!?]+\s", text)
        return [s.strip() for s in sentences if s.strip()]


    def word_tokenize(self,sentence):
        """
        Tokenize a sentence into individual tokens.

        Parameters:
            sentence (str): A single sentence.

        Returns:
            list[str]: List of tokens (words).
        """
        return sentence.split()

    # ------------ SAVE ------------ #

    def save(self,sentences, filepath):
        """
        Save tokenized sentences to a file, one sentence per line.

        Parameters:
            sentences (list[list[str]]):
                A list where each element is a list of word tokens.
            filepath (str):
                Path where the output file will be written.

        Returns:
            None
        """
        
        with open(filepath, "w", encoding="utf-8") as f:
            for tokens in sentences:
                f.write(" ".join(tokens) + "\n")
