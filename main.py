import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import pickle
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_models():
    yolo_model = YOLO("yolov11s-face.pt")
    face_app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
    face_app.prepare(ctx_id=0)
    return yolo_model, face_app

def load_embeddings():
    with open("data/features/face_embeddings.pkl", "rb") as f:
        stored_embeddings = pickle.load(f)
    # avg_emb = {
    #     person: np.mean(vectors, axis=0)
    #     for person, vectors in stored.items()
    # }
    return stored_embeddings

yolo_model, face_app = load_models()
# avg_embeddings = load_embeddings()
stored_embeddings = load_embeddings()

class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.prev_time = None

    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        image = frame.to_ndarray(format="bgr24")
        results = yolo_model(image)[0]

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box[:4])
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(image.shape[1], x2 + margin_x)
            y2 = min(image.shape[0], y2 + margin_y)

            face_crop = image[y1:y2, x1:x2]
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            faces = face_app.get(face_rgb)

            label = "Unknown"
            if faces:
                query_emb = faces[0].embedding.reshape(1, -1)
                # similarities = {
                #     person: cosine_similarity(query_emb, np.array(emb).reshape(1, -1))[0][0]
                #     for person, emb in avg_embeddings.items()
                # }
                similarities = {}
                for person, vectors in stored_embeddings.items():
                    vectors = np.array(vectors)  
                    sims = cosine_similarity(query_emb, vectors)[0] 
                    top_k = min(3, len(sims))
                    top_k_sims = np.sort(sims)[-top_k:] 
                    similarities[person] = np.mean(top_k_sims)

                best_match = max(similarities, key=similarities.get)
                if similarities[best_match] > 0.4:
                    label = f"{best_match} ({similarities[best_match]:.2f})"

            x1 = max(0, x1 + margin_x)
            y1 = max(0, y1 + margin_y)
            x2 = min(image.shape[1], x2 - margin_x)
            y2 = min(image.shape[0], y2 - margin_y)

            color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if self.prev_time:
            fps = 1 / (cv2.getTickCount() / cv2.getTickFrequency() - self.prev_time)
            cv2.putText(image, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        self.prev_time = cv2.getTickCount() / cv2.getTickFrequency()
        return image

st.title("Real-Time Face Recognition")
st.caption("Using YOLOv11 + InsightFace")
if st.button("Register new face"):
    st.switch_page("pages/register_face.py")

webrtc_streamer(
    key="face-detect",
    video_transformer_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
