import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model

model = load_model("tomato_disease_detector-99acc.keras")

classes = ['Bacterial spot', 'Early blight', 'Late blight', 'Leaf Mold',
           'Septoria leaf spot', 'Spider mites', 'Target Spot',
           'Yellow Leaf Curl Virus', 'Mosaic virus', 'Healthy']

st.header("🍅 Tomato Leaf Disease Detector!")
st.subheader("Model Accuracy: 97.76%")


image = st.file_uploader("Upload your picture here!")

if image is not None:
    image = Image.open(image).convert("RGB")
    st.image(image, caption='The photo you uploaded', width=250)
    img = image.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

button = st.button("Detect The Disease!")

if button and image is not None:
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    st.success(f"The Detected Disease is: {predicted_class}")