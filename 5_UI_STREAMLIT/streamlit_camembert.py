import pandas as pd
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification, AdamW, CamembertConfig, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import streamlit as st
import os
import sentencepiece
import requests
from pyngrok import ngrok
from safetensors import safe_open
from safetensors.torch import load_model

# URLs for the config and model file on SwissTransfer
file_url = 'https://www.swisstransfer.com/d/d13af6ae-1f27-4f73-ad87-99488bb4b79b'
config_url = 'https://www.swisstransfer.com/d/6cc9ba75-fcd6-4718-a4af-9f81216ae7f4'
model_url = 'https://www.swisstransfer.com/d/4dd3b1f2-34fa-497e-a792-1903760e8ff4'

def load_model():
    model_path = r'./geneva_model'
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model

# Fonction de prédiction
def predict_difficulty(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=64, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

#GIF image
gif_url = "http://www.animated-gifs.fr/category_countries/france/39437216.gif"
gif_width = 300
st.markdown(
    f"""
    <div style="display: flex; justify-content: center;">
        <img src="{gif_url}" width="{gif_width}">
    </div>
    """,
    unsafe_allow_html=True,
)

#Description
st.markdown(
  """
    <div style="text-align: center;">
        <h1> Language Level Evaluator </h1>
        <p>This application uses a model to evaluate the French language levl of a text.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Text Area
user_input = st.text_area("Enter the text to evaluate :")

# Prediction button
if st.button("Evaluate"):
    if user_input:
        model = load_model()
        tokenizer = CamembertTokenizer.from_pretrained('camembert/camembert-large')
        predicted_class = predict_difficulty(user_input)
        difficulty_levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']
        difficulty = difficulty_levels[predicted_class]

        st.success(f"The predicted languag level is : {difficulty}")
    else:
        st.warning("Trying to evaluate empty air? We're good, but not that good. Type in some text, and we'll work our language magic.")

# Advice for users
st.info("Tip: Copy & Paste a paragraph and / or sentence to get a better evaluation of the language level")
