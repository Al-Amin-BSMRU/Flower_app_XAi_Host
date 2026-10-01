import gc  # RAM মেমরি পরিষ্কার করার জন্য
import os
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from lime import lime_image
from skimage.segmentation import mark_boundaries

st.set_page_config(page_title="Flower Classification with XAI", page_icon="🌸", layout="wide")
st.title("🌸 Flower Classification with Explainable AI (Grad-CAM & LIME)")
st.write("Upload a flower image to get the prediction and see which parts of the image the model used.")

IMAGE_SIZE = (224, 224)
SEED = 42

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
            class_names = [line.strip() for line in f if line.strip()]
    else:
        class_names = FALLBACK_CLASS_NAMES

    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    head_layers = model.layers[model.layers.index(base) + 1:]
    feature_extractor = tf.keras.Model(base.inputs, base.get_layer("out_relu").output)
    return model, class_names, feature_extractor, head_layers


model, CLASS_NAMES, feature_extractor, head_layers = load_assets()

if model.output_shape[-1] != len(CLASS_NAMES):
    st.error(f"ক্লাস সংখ্যা মেলেনি: মডেলে {model.output_shape[-1]}টি, নামের তালিকায় {len(CLASS_NAMES)}টি।")
    st.stop()


def pretty(name):
    return name.replace("_", " ").title()


def preprocess(pil_image):
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    resized = img.resize(IMAGE_SIZE)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return resized, arr


# ---------------- Grad-CAM ----------------
def gradcam_heatmap(img_array, class_idx):
    x = tf.convert_to_tensor(img_array[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out = feature_extractor(x, training=False)
        tape.watch(conv_out)
        h = conv_out
        for layer in head_layers:
            h = layer(h)
        score = h[:, class_idx]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(conv_out[0] * weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def gradcam_images(img_array, heatmap, alpha=0.4):
    big = tf.image.resize(heatmap[..., None], IMAGE_SIZE, method="bilinear").numpy()[..., 0]
    color = plt.get_cmap("jet")(np.clip(big, 0, 1))[..., :3]
    overlay = (1 - alpha) * img_array + alpha * color
    return np.clip(color, 0, 1), np.clip(overlay, 0, 1)


# ---------------- LIME (Optimized for RAM) ----------------
def lime_predict_fn(images):
    images = np.asarray(images, dtype=np.float32)
    outs = []
    # Batch size কমিয়ে ৮ করা হয়েছে RAM প্রসেস হালকা রাখতে
    for i in range(0, len(images), 8):
        outs.append(model(images[i:i + 8], training=False).numpy())
    return np.concatenate(outs, axis=0)


def compute_lime(img_array, class_idx, num_samples, num_regions):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array.astype("double"),
        lime_predict_fn,
        labels=(class_idx,),
        top_labels=None,
        hide_color=0,
        num_samples=num_samples,
        batch_size=8,  # ৩২ থেকে কমিয়ে ৮ করা হয়েছে
        random_seed=SEED,
    )
    temp, mask = explanation.get_image_and_mask(
        class_idx, positive_only=True, num_features=num_regions, hide_rest=False)
    boundaries = np.clip(mark_boundaries(temp, mask), 0, 1)

    temp_only, _ = explanation.get_image_and_mask(
        class_idx, positive_only=True, num_features=num_regions, hide_rest=True)
    
    # প্রসেসিং শেষে মেমরি পরিষ্কার করা
    del explainer, explanation
    gc.collect()

    return boundaries, np.clip(temp_only, 0, 1)


# ---- UI ----
st.sidebar.header("LIME settings")
# স্যাম্পল ১০০ থেকে ১০০০ এর বদলে ৩০ থেকে ১৫০ করা হয়েছে যাতে সার্ভার ক্র্যাশ না করে
lime_samples = st.sidebar.slider("Samples (fewer = faster & prevents crash)", 30, 150, 60, step=10)
lime_regions = st.sidebar.slider("Number of important regions", 3, 10, 5)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    resized_img, img_array = preprocess(Image.open(uploaded_file))

    probs = model.predict(img_array[None, ...], verbose=0)[0]
    pred_index = int(np.argmax(probs))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Image")
        st.image(resized_img, use_container_width=True)
    with col2:
        st.subheader("Prediction")
        st.success(f"**{pretty(CLASS_NAMES[pred_index])}**")
        st.info(f"Confidence: {100 * probs[pred_index]:.2f}%")
        st.markdown("**Top 3**")
        for i in np.argsort(probs)[::-1][:3]:
            st.write(f"- {pretty(CLASS_NAMES[i])}: {100 * probs[i]:.1f}%")

    st.divider()
    st.subheader("Explainable AI")
    tab_gradcam, tab_lime = st.tabs(["Grad-CAM", "LIME"])

    with tab_gradcam:
        try:
            heatmap = gradcam_heatmap(img_array, pred_index)
            color, overlay = gradcam_images(img_array, heatmap)
            c1, c2 = st.columns(2)
            c1.image(color, caption="Grad-CAM Heatmap", use_container_width=True)
            c2.image(overlay, caption="Grad-CAM Overlay", use_container_width=True)
            st.caption("লাল/হলুদ অঞ্চলে মডেল সবচেয়ে বেশি গুরুত্ব দিয়েছে, নীল অঞ্চলে কম।")
        except Exception as e:
            st.warning(f"Grad-CAM তৈরি করতে সমস্যা হয়েছে: {e}")

    with tab_lime:
        st.write("LIME ছবিকে ছোট ছোট অংশে ভেঙে দেখে কোন অংশগুলো পূর্বাভাসের পক্ষে সবচেয়ে বেশি কাজ করেছে। "
                 "এতে কিছু সময় লাগে (সার্ভারে সাধারণত ১০–৪০ সেকেন্ড)।")
        key = (uploaded_file.name, uploaded_file.size, lime_samples, lime_regions)

        if st.button("Generate LIME explanation"):
            try:
                with st.spinner("LIME explanation তৈরি হচ্ছে..."):
                    result = compute_lime(img_array, pred_index, lime_samples, lime_regions)
                st.session_state["lime_result"] = (key, result)
            except Exception as e:
                st.warning(f"LIME তৈরি করতে সমস্যা হয়েছে: {e}")

        saved = st.session_state.get("lime_result")
        if saved is not None and saved[0] == key:
            boundaries, only_regions = saved[1]
            c1, c2 = st.columns(2)
            c1.image(boundaries, caption=f"LIME: top {lime_regions} regions (boundaries)", use_container_width=True)
            c2.image(only_regions, caption="LIME: important regions only", use_container_width=True)
            st.caption("চিহ্নিত অংশগুলো সেই অংশ, যেগুলো এই ফুলের পূর্বাভাসকে সবচেয়ে বেশি সমর্থন করেছে।")
