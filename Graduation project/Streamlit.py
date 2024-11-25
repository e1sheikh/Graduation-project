import streamlit as st
import pandas as pd
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# Function to load models
@st.cache_resource
def load_models():
    svc_model = joblib.load('svc_model.pkl')
    naive_model = joblib.load('naive_model.pkl')
    logistic_model = joblib.load('logistic_model.pkl')
    lstm_model = tf.keras.models.load_model('lstm_model.h5')
    return svc_model, naive_model, logistic_model, lstm_model

# Load models
svc_model, naive_model, logistic_model, lstm_model = load_models()

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Define stop words and stemmer
stop_words = set(stopwords.words('english'))
stemming = PorterStemmer()

# Text cleaning function
def clean_text(text):
    # Convert to lowercase
    tokens = word_tokenize(text.lower())

    # Remove punctuation, stopwords, numbers, and apply stemming
    tokens = [stemming.stem(word) for word in tokens if word not in string.punctuation 
              and word not in stop_words and not word.isdigit()]

    return ' '.join(tokens)

# Text input for real-time feedback
st.title("Customer Product Reviews Sentiment Analysis App")
user_input = st.text_area("Enter text for sentiment analysis:", placeholder="Type here...")

# Model selection
choose_model = st.radio(
    "Choose your model:",
    ('SVM', 'Naive Bayes', 'Logistic Regression', 'LSTM'))

if st.button("Analyze Sentiment"):
    if user_input:
        # Preprocess the text
        cleaned_input = clean_text(user_input)
        
        # Prediction based on selected model
        if choose_model == 'SVM':
            prediction = svc_model.predict([cleaned_input])
        elif choose_model == 'Naive Bayes':
            prediction = naive_model.predict([cleaned_input])
        elif choose_model == 'Logistic Regression':
            prediction = logistic_model.predict([cleaned_input])
        elif choose_model == 'LSTM':
            # Tokenize and pad the sequence for LSTM
            tokenizer = tf.keras.preprocessing.text.Tokenizer()
            tokenizer.fit_on_texts([cleaned_input])  # Fit on the input text
            sequence = tokenizer.texts_to_sequences([cleaned_input])
            padded_sequence = pad_sequences(sequence, maxlen=150)  # Assuming maxlen=150
            prediction = lstm_model.predict(padded_sequence)
            prediction = (prediction > 0.5).astype("int32")  # Convert probabilities to class labels

        # Convert prediction to sentiment label
        if choose_model != 'LSTM':
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
        else:
            sentiment = "Positive" if prediction[0][0] == 1 else "Negative"

        # Display sentiment with visualization
        if sentiment == "Positive":
            st.success(f"Prediction: {sentiment} ")
        else:
            st.error(f"Prediction: {sentiment} ")
