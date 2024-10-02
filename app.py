import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image  # Updated import for consistency
import numpy as np
from gtts import gTTS
import io

# Load the pre-trained model
model = tf.keras.models.load_model('saved_model/currency_denomination_model7.keras')  # Ensure the correct path to your saved model

# Function to predict the denomination and generate voice output
def predict_currency(img):
    # Load and preprocess the image
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    
    # Resize the image to the input shape of the model
    img_array = tf.image.resize(img_array, (128, 128))  # Resize to 128x128
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Model prediction
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)[0]
    
    # Replace this with actual class labels (ensure these match your training classes)
    class_labels = ['10', '20', '50', '100', '200', '500', '2000']
    denomination = class_labels[class_idx]
    
    # Convert text to speech and generate audio in memory
    tts = gTTS(text=f'This is a {denomination} rupees note', lang='en')
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)  # Move to the start of the BytesIO object
    
    return denomination, audio_bytes

# Streamlit app layout
st.set_page_config(page_title="Currency Denomination Identifier", layout="centered")
st.title("Currency Denomination Identifier")
st.markdown("Upload an image of a currency note, and the system will predict its denomination and provide an audio output.")

# File upload section
uploaded_file = st.file_uploader("Choose a currency note image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image with higher resolution
    img = image.load_img(uploaded_file, target_size=(512, 512))  # Display higher resolution
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Predict denomination
    denomination, audio_bytes = predict_currency(img)
    
    st.success(f"Predicted Denomination: {denomination}")
    
    # Audio output
    st.audio(audio_bytes, format="audio/mp3")

# Add footer
st.markdown("""
    <style>
        footer {visibility: hidden;}
        footer:after {
            content: 'Powered by TensorFlow & Streamlit';
            visibility: visible;
            display: block;
            position: relative;
            padding: 10px;
            color: #4CAF50;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)
