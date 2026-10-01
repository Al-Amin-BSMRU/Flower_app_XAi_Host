import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.title("Flower Classification Model")

# labels.txt থাকলে সেটাই ব্যবহার হবে (repo-তে model.h5-এর পাশে রাখুন)।
# না থাকলে নিচের বর্ণানুক্রমিক তালিকা (১৯টি নাম) ব্যবহার হবে।
FALLBACK_CLASS_NAMES = sorted([
    'astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy',
    'carnation', 'common_daisy', 'coreopsis', 'daffodil', 'dandelion',
    'iris', 'lavender', 'lotus', 'magnolia', 'orchid',
    'rose', 'sunflower', 'tulip', 'water_lily'
])


@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("model.h5", compile=False)
    if os.path.exists("labels.txt"):
        with open("labels.txt") as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = FALLBACK_CLASS_NAMES
    return model, names


model, class_names = load_assets()

# মডেলের আউটপুট সংখ্যা আর নামের সংখ্যা না মিললে থামিয়ে দেখানো
if model.output_shape[-1] != len(class_names):
    st.error(f"ক্লাস সংখ্যা মেলেনি: মডেলে {model.output_shape[-1]}টি, নামের তালিকায় {len(class_names)}টি।")
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")   # PNG alpha / grayscale / ঘোরানো ছবির সমস্যা ঠিক করে
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess: ট্রেনিংয়ের মতোই 224x224, পিক্সেল ÷ 255 (মান 0..1)
    img = image.resize((224, 224))
    img_array = np.asarray(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)[0]

    idx = int(np.argmax(predictions))
    predicted_class = class_names[idx].replace("_", " ").title()
    confidence = float(predictions[idx]) * 100

    st.write(f"### Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.write("Top 3:")
    for i in np.argsort(predictions)[::-1][:3]:
        st.write(f"- {class_names[i].replace('_', ' ').title()}: {predictions[i] * 100:.1f}%")
