# Face recognition
This project implements a two-stage pipeline for face recognition, combining real-time face detection and indentification:<br>
1. **Face Detection** using a custom-trained YOLO model to accurately locate faces in images or webcam streams.<br>
2. **Feature Extraction & Matching** with ArcFace embeddings, enabling robust comparison of faces using cosine similarity.<br>

# Pipeline
![Pipeline](pipeline_face_recognition.png)

# Models
- **Face Detection**: [Download model YOLOv11s-face](https://github.com/akanametov/yolo-face)
- **Feature Extraction**: [Download model buffalo_l](https://github.com/deepinsight/insightface/tree/master/model_zoo)<br>

# Usage
**Clone my repository**:
```
git clone https://github.com/Wan1302/Face_recognition.git
```

**Collect new faces for dataset**: enter your name and press `Enter` to take 10 pictures.
```
py collect_new_faces.py
```

**Encode faces**: 
```
py encode_faces.py
```

**Realtime face recognition**:
```
py realtime_face_recognition.py
```

# Inferences