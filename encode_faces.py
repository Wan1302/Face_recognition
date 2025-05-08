import cv2
import numpy as np
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import os
import pickle

# Load YOLOv11s face detector
yolo_model = YOLO("yolov11s-face.pt")

# Load ArcFace model
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
        print(f"Precomputed facial feature embeddings are available for {person_name}.")
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
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_crop_rgb = face_crop_rgb.astype(np.uint8)
            
            emb = app.get(face_crop_rgb)
            if emb:
                feature = emb[0].embedding
                embeddings[person_name].append(feature.tolist())

with open(feature_file, "wb") as f:
    pickle.dump(embeddings, f)
