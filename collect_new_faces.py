import cv2
import os

name = input("Enter your name: ")

save_dir = f"data/images/{name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Can not open webcam.")
    exit()

num_photos = 20
captured = 0
image_paths = []

while captured < num_photos:
    ret, frame = cap.read()
    cv2.imshow("Press 'Enter' to take pictures...", frame)

    if cv2.waitKey(1) & 0xFF == 13:
        save_path = os.path.join(save_dir, f"{name}_{captured+1}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Saved {save_path}")
        image_paths.append(save_path)
        captured += 1

cap.release()
cv2.destroyAllWindows()