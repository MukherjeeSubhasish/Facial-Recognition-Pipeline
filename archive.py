"""
"""

import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ------------------------------
# Step 1: Load ORL Faces Dataset
# ------------------------------
def load_orl_faces(dataset_path="orl_faces"):
    images, labels = [], []
    label_map = {}  # mapping ID -> folder name (person)

    label_id = 0
    for person_dir in sorted(os.listdir(dataset_path)):     # list of subdirs and files in 'orl_faces' path
        if person_dir.startswith("s"):  # s1, s2, ...
            label_map[label_id] = person_dir
            person_path = os.path.join(dataset_path, person_dir)    # 'orl_faces/s1'

            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                images.append(img)
                labels.append(label_id)

            label_id += 1

    return np.array(images), np.array(labels), label_map


# ------------------------------
# Step 2: Face Detection & Preprocessing
# ------------------------------
def detect_and_crop(img, size=(100, 100)):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        # fallback: just resize whole image
        return cv2.resize(img, size)

    (x, y, w, h) = faces[0]  # take first detected face
    face = img[y:y+h, x:x+w]
    return cv2.resize(face, size)


# ------------------------------
# Step 3: Training + Testing
# ------------------------------
def main():
    # Load dataset
    images, labels, label_map = load_orl_faces("orl_faces")
    print(f"Loaded {len(images)} images from {len(label_map)} people.")

    # Preprocess (detect faces & resize)
    processed = [detect_and_crop(img) for img in images]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        processed, labels, test_size=0.1, stratify=labels, random_state=42
    )

    # Train LBPH recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(X_train, np.array(y_train))
    print("Training complete.")

    # Evaluate
    correct, total = 0, 0
    for img, label in zip(X_test, y_test):
        pred, conf = recognizer.predict(img)
        total += 1
        if pred == label:
            correct += 1

    acc = correct / total * 100
    print(f"Recognition Accuracy: {acc:.2f}%")

    # ------------------------------
    # Visual Test
    # ------------------------------
    test_img = X_test[0]
    true_label = y_test[0]
    pred_label, conf = recognizer.predict(test_img)

    out_img = cv2.resize(test_img, (200, 200))
    cv2.imwrite("test_prediction.png", out_img)
    print(
        f"True: {label_map[true_label]}, "
        f"Predicted: {label_map[pred_label]}, "
        f"Confidence: {conf:.2f}"
    )
    print("Saved test image as test_prediction.png")



if __name__ == "__main__":
    main()

