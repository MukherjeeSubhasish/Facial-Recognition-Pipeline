Conceptual Design

I am using the following dataset:

load_orl_faces("orl_faces")


Why I am not using BioID?
The BioID dataset has a solid collection of face images. However, it does not provide a direct mapping between each image and a specific user name or user ID.
For face recognition, such mapping is essential, because each image needs to be associated with the corresponding user. Doing this manually for BioID would take significant time.
Therefore, I chose the ORL Faces dataset.
	• It contains 400 face images of 40 people.
	• Each person has 10 images.
	• The dataset is already organized in a hierarchical directory structure, making user-to-image mapping straightforward.

================================.       Steps.       ================================

Step 1:
Load the dataset:
import cv2

Step 2:
Map all images to their respective user IDs.
This is already handled by the directory structure of the dataset.

Step 3:
Load a pretrained Haar Cascade model from OpenCV:
haarcascade_frontalface_default.xml
Initialize the detectMultiScale method from this classifier.

Step 4:
Use the detectMultiScale object to detect faces in each image.
If detection is successful, crop and resize the face region to 100 × 100 pixels.
	• Coordinates saved: (x, y, w, h)
		○ (x, y) → top-left corner of the face
		○ (w, h) → width and height from that corner

Step 5:
Split the 400 images randomly into training and test sets (90% / 10%).

Initialize the recognizer:
cv2.face.LBPHFaceRecognizer_create()
Train (fine-tune) the recognizer with the training dataset.

Step 6:
Inference: Evaluate the model on the test dataset.
Report the recognition accuracy.
