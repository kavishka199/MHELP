"""
component: cognitive welbeing
file: cognitive_welbeing.py
author: kavishka
"""

import pandas as pd
import spacy
from transformers import pipeline
from fuzzywuzzy import fuzz

# Read the dataset
df = pd.read_csv('datasets/cognitive_welbeing.csv')

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="distilbert-base-cased")

# Define the get_answer function
def get_answer(question):
    # Check if the question is in the dataset
    match = None
    max_score = -1
    for i, q in enumerate(df['Question']):
        score = fuzz.token_sort_ratio(question, q)
        if score > max_score:
            max_score = score
            match = df.iloc[i]

    # If the score is high enough, get the answer
    if max_score > 70:
        context = match['Answer']
        answer = qa_pipeline(question=question, context=context)['answer']
    else:
        answer = "I'm sorry, I am an AI model and I'm not trained to answer this question."

    return answer