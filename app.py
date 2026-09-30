import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.title("Flower Classification Model")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")


model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    # আপনার ক্লাসের নামগুলো দিয়ে দিন
    class_names = ["Class 1", "Class 2", "Class 3"]

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100

    st.write(f"### Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")