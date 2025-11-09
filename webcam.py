# =========================
# Imports
# =========================
import cv2
import mediapipe as mp
from base64 import b64decode
from google.colab import output
from IPython.display import Javascript
from google.colab.patches import cv2_imshow

# =========================
# 1. JavaScript webcam snapshot
# =========================
def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
      async function takePhoto(quality) {
        const div = document.createElement('div');
        const video = document.createElement('video');
        const stream = await navigator.mediaDevices.getUserMedia({video: true});
        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();

        // Wait for user snapshot
        const btn = document.createElement('button');
        btn.textContent = 'Take Snapshot';
        div.appendChild(btn);

        await new Promise((resolve) => btn.onclick = resolve);
        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getTracks().forEach(track => track.stop());
        const data = canvas.toDataURL('image/jpeg', quality);
        div.remove();
        return data;
      }
      takePhoto(%f);
    ''' % quality)
    display(js)
    data = output.eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

# =========================
# 2. Initialize MediaPipe Face Detector
# =========================
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
)

# =========================
# 3. Capture snapshot from webcam
# =========================
print("Starting webcam — click 'Take Snapshot' below to capture an image")
image_path = take_photo("captured_face.jpg")

frame = cv2.imread(image_path)
print("Snapshot captured")

# =========================
# 4. Detect face(s)
# =========================
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_detector.process(rgb)

if results.detections:
    print(f"Detected {len(results.detections)} face(s)")
    for detection in results.detections:
        bbox = detection.location_data.relative_bounding_box
        ih, iw, _ = frame.shape
        x, y, w, h = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
        x, y = max(0, x), max(0, y)
        w, h = min(iw - x, w), min(ih - y, h)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
else:
    print("No faces detected.")

# =========================
# 5. Display annotated frame
# =========================
cv2_imshow(frame)
