import streamlit as st
import os
from PIL import Image
from datetime import datetime
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis


st.set_page_config(page_title="Face Registration", layout="centered")

st.markdown("""
    <style>
        .stApp {background-color: #f4f4f4;}
        .title {text-align: center; font-size: 36px; font-weight: bold; color: #4CAF50;}
        .subtitle {text-align: center; font-size: 18px; margin-bottom: 20px;}
        .progress-text {font-weight: bold;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>📸 Face Registration</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Register your face by capturing 10 images</div>", unsafe_allow_html=True)


if "photo_count" not in st.session_state:
    st.session_state.photo_count = 0
if "photo_saved" not in st.session_state:
    st.session_state.photo_saved = False

name = st.text_input("Enter your name to begin:")

if name:
    save_dir = os.path.join("data/images", name)
    os.makedirs(save_dir, exist_ok=True)

    st.markdown("### 📷 Capture Face")
    st.caption("Take a clear photo of your face. Capture all 10 photos for better accuracy.")

    progress = st.progress(st.session_state.photo_count / 10)
    st.markdown(f"<p class='progress-text'>Captured {st.session_state.photo_count}/10 photos</p>", unsafe_allow_html=True)

    photo = st.camera_input("Take a new photo")

    if photo and st.session_state.photo_count < 10:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(save_dir, f"{name}_{timestamp}.jpg")
        img = Image.open(photo)
        img.save(img_path)

        st.session_state.photo_count += 1
        st.success(f"✅ Photo saved ({st.session_state.photo_count}/10)")
        progress.progress(st.session_state.photo_count / 10)

        if st.session_state.photo_count == 10:
            st.session_state.photo_saved = True
            st.success("🎉 All 10 photos captured! You can now encode them.")

    if st.session_state.photo_saved:
        if st.button("🧠 Encode and Return to Main Page"):
            yolo_model = YOLO("yolov11s-face.pt")
            app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
            app.prepare(ctx_id=0)

            db_path = "data/images"
            feature_path = "data/features"
            feature_file = os.path.join(feature_path, "face_embeddings.pkl")

            if os.path.exists(feature_file):
                with open(feature_file, "rb") as f:
                    embeddings = pickle.load(f)
            else:
                embeddings = {}

            for person_name in os.listdir(db_path):
                if person_name in embeddings:
                    continue

                person_path = os.path.join(db_path, person_name)
                embeddings[person_name] = []

                for img_name in os.listdir(person_path):
                    img_path = os.path.join(person_path, img_name)
                    img = cv2.imread(img_path)
                    results = yolo_model(img)[0]
                    for box in results.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box[:4])
                        margin_x = int((x2 - x1) * 0.2)
                        margin_y = int((y2 - y1) * 0.2)
                        x1 = max(0, x1 - margin_x)
                        y1 = max(0, y1 - margin_y)
                        x2 = min(img.shape[1], x2 + margin_x)
                        y2 = min(img.shape[0], y2 + margin_y)
                        face_crop = img[y1:y2, x1:x2]
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        emb = app.get(face_rgb)
                        if emb:
                            embeddings[person_name].append(emb[0].embedding.tolist())

            with open(feature_file, "wb") as f:
                pickle.dump(embeddings, f)

            st.success("✅ Face embeddings saved successfully!")
            st.switch_page("main.py")

else:
    st.info("Please enter your name to start.")

