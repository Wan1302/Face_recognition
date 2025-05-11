import cv2
import pickle
import numpy as np
import time
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

yolo_model = YOLO("yolov11s-face.pt")

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

with open("data/features/face_embeddings.pkl", "rb") as f:
    stored_embeddings = pickle.load(f)

# Avg
# avg_embeddings = {
#     person: np.mean(vectors, axis=0)
#     for person, vectors in stored_embeddings.items()
# }

cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to exit...")

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)[0]

    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(frame.shape[1], x2 + margin_x)
        y2 = min(frame.shape[0], y2 + margin_y)

        face_crop = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        faces = app.get(face_rgb)

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
        x2 = min(frame.shape[1], x2 - margin_x)
        y2 = min(frame.shape[0], y2 - margin_y)

        if label != "Unknown":
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
