import streamlit as st
import pandas as pd
from transformers import pipeline
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Suppress warnings
import warnings
warnings.simplefilter('ignore')

# Initialize the image captioning pipeline
caption = pipeline('image-to-text')

# Load vectorizer and models
vectorizer = joblib.load('vectorizer.pkl')
model_likes = joblib.load('model_likes.pkl')
model_shares = joblib.load('model_shares.pkl')
model_saves = joblib.load('model_saves.pkl')

# Define function for making predictions
def make_predictions(caption, vectorizer, models):
    caption_transformed = vectorizer.transform([caption])
    predictions = {}
    for metric, model in models.items():
        predictions[metric] = abs(model.predict(caption_transformed)[0])
    return predictions

# Streamlit app
st.title("Instagram Engagement Predictor")

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Generate caption for the uploaded image
    cap = caption(uploaded_file)
    generated_caption = cap[0]['generated_text']

    # Make predictions for the generated caption
    models = {'Likes': model_likes, 'Shares': model_shares, 'Saves': model_saves}
    predictions = make_predictions(generated_caption, vectorizer, models)

    # Display predictions
    st.subheader("Generated Caption:")
    st.write(generated_caption)
    st.subheader("Predictions:")
    for metric, value in predictions.items():
        st.write(f"Predicted {metric}:", int(value.round()))
