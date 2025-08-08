import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
from PIL import Image, UnidentifiedImageError

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="😷 Mask Detector", page_icon="😷", layout="centered")

# ---------- LOAD MODEL FROM GOOGLE DRIVE ----------
@st.cache_resource
def load_model():
    model_path = "mask_detector_model.keras"
    
    if not os.path.exists(model_path):
        file_id = "1o3BWIkWtjxPbcCXFNJOCuEuPdGsvCfCu"  # 🔁 Replace this with your actual file ID
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    
    return tf.keras.models.load_model(model_path)

model = load_model()
class_names = ['without_mask', 'with_mask']

# ---------- HEADER ----------
st.markdown("""
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4CAF50;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #888;
            margin-bottom: 20px;
        }
        .result {
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
            margin-top: 20px;
        }
        .mask {background-color: #4CAF50; color: white;}
        .no-mask {background-color: #F44336; color: white;}
        footer {text-align:center; font-size:14px; color:#aaa; margin-top:40px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">😷 Mask Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image to check if a mask is worn</div>', unsafe_allow_html=True)

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("📤 Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    try:
        # Open image
        pil_img = Image.open(uploaded_file).convert("RGB")
        img = np.array(pil_img)

        # Convert RGB -> BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize for model
        img_resized = cv2.resize(img_bgr, (180, 180))

        # Show uploaded image
        st.image(pil_img, caption="🖼️ Uploaded Image", width=400)

        # Prepare for prediction
        img_array = np.expand_dims(img_resized, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        pred = (prediction > 0.5).astype(int).flatten()

        if pred[0] == 0:
            st.markdown('<div class="result mask">✅ With Mask</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result no-mask">❌ Without Mask</div>', unsafe_allow_html=True)

    except UnidentifiedImageError:
        st.error("❌ Unsupported or corrupted image file. Please use PNG or JPG.")

# ---------- FOOTER ----------
st.markdown("<footer>🚀 Created by <b>Pavankumar</b> | 💻 Powered by TensorFlow & Streamlit</footer>", unsafe_allow_html=True)
