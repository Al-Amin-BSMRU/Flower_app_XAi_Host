import base64
import gc
import io
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from lime import lime_image
from skimage.segmentation import mark_boundaries
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(title="Flower XAI API")

IMAGE_SIZE = (224, 224)
SEED = 42

FALLBACK_CLASS_NAMES = sorted([
    'astilbe', 'bellflower', 'black_eyed_susan', 'calendula', 'california_poppy',
    'carnation', 'common_daisy', 'coreopsis', 'daffodil', 'dandelion',
    'iris', 'lavender', 'lotus', 'magnolia', 'orchid',
    'rose', 'sunflower', 'tulip', 'water_lily'
])

# ১. মডেল এবং অ্যাসেট লোড করা
model = tf.keras.models.load_model("model.h5", compile=False)

if os.path.exists("labels.txt"):
    with open("labels.txt") as f:
        CLASS_NAMES = [line.strip() for line in f if line.strip()]
else:
    CLASS_NAMES = FALLBACK_CLASS_NAMES

base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
head_layers = model.layers[model.layers.index(base) + 1:]
feature_extractor = tf.keras.Model(base.inputs, base.get_layer("out_relu").output)

def preprocess(pil_image):
    img = ImageOps.exif_transpose(pil_image).convert("RGB")
    resized = img.resize(IMAGE_SIZE)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return resized, arr

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
    return np.clip(overlay, 0, 1)

def lime_predict_fn(images):
    images = np.asarray(images, dtype=np.float32)
    outs = []
    for i in range(0, len(images), 8):
        outs.append(model(images[i:i + 8], training=False).numpy())
    return np.concatenate(outs, axis=0)

def compute_lime(img_array, class_idx, num_samples=60, num_regions=5):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array.astype("double"),
        lime_predict_fn,
        labels=(class_idx,),
        top_labels=None,
        hide_color=0,
        num_samples=num_samples,
        batch_size=8,
        random_seed=SEED,
    )
    temp, mask = explanation.get_image_and_mask(class_idx, positive_only=True, num_features=num_regions, hide_rest=False)
    boundaries = np.clip(mark_boundaries(temp, mask), 0, 1)
    
    del explainer, explanation
    gc.collect()
    return boundaries

def numpy_to_base64(img_np):
    """Numpy অ্যারে ছবিকে Base64 String-এ রূপান্তর করে যা এন্ড্রয়েডে পাঠানো সহজ"""
    img_uint8 = (img_np * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)
    buff = io.BytesIO()
    pil_img.save(buff, format="JPEG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

# ---- API ENDPOINT ----
@app.post("/explain_xai")
async def process_xai(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
        resized_img, img_array = preprocess(pil_img)

        # ১. প্রেডিকশন
        probs = model.predict(img_array[None, ...], verbose=0)[0]
        pred_index = int(np.argmax(probs))
        class_name = CLASS_NAMES[pred_index].replace("_", " ").title()
        confidence = float(100 * probs[pred_index])

        # ২. Grad-CAM জেনারেট
        heatmap = gradcam_heatmap(img_array, pred_index)
        overlay = gradcam_images(img_array, heatmap)
        gradcam_b64 = numpy_to_base64(overlay)

        # ৩. LIME জেনারেট
        lime_boundaries = compute_lime(img_array, pred_index, num_samples=60, num_regions=5)
        lime_b64 = numpy_to_base64(lime_boundaries)

        return JSONResponse(content={
            "status": "success",
            "prediction": class_name,
            "confidence": f"{confidence:.2f}%",
            "gradcam_image": gradcam_b64,
            "lime_image": lime_b64
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})
