"""
component: emotional welbeing
file: emotional_welbeing.py
author: kalana madhawa
"""

# impoprt libs
import pandas as pd
import torch
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer

# Read the dataset
df = pd.read_csv('datasets/emotional_welbeing.csv')

# Set up the model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# Define the get_answer function
def get_answer(question):
    context = df['Answer'][0]  # Use the first answer from the dataset as the context
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end]))
    if '[SEP]' in answer:
        answer = "I'm sorry, I am an AI model my data are not trained to answer this question."
    return answer