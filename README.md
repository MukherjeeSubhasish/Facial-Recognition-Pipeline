Conceptual Design

I am using the following dataset:

load_orl_faces("orl_faces")


Why am I not using BioID?

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




================================       Part2      ================================

sources:
https://www.kaggle.com/datasets/tavarez/the-orl-database-for-training-and-testing
https://git-disl.github.io/GTDLBench/datasets/

citation:

@INPROCEEDINGS{GTDLBenchICDCS, 
    author={{Liu}, Ling and {Wu}, Yanzhao and {Wei}, Wenqi and {Cao}, Wenqi and {Sahin}, Semih and {Zhang}, Qi}, 
    booktitle={2018 IEEE 38th International Conference on Distributed Computing Systems (ICDCS)}, 
    title="{Benchmarking Deep Learning Frameworks: Design Considerations, Metrics and Beyond}",
    year={2018},
    pages={1258-1269}, 
    doi={10.1109/ICDCS.2018.00125}, 
    ISSN={2575-8411}, 
    month={July},
}

@ARTICLE{GTDLBencharTSC,
    author={Y. {Wu} and L. {Liu} and C. {Pu} and W. {Cao} and S. {Sahin} and W. {Wei} and Q. {Zhang}}, 
    journal={IEEE Transactions on Services Computing}, 
    title={A Comparative Measurement Study of Deep Learning as a Service Framework}, 
    year={2019}, 
    volume={}, 
    number={}, 
    pages={1-1}, 
    keywords={Libraries;Parallel processing;Hardware;Training;Runtime;Deep learning;Task analysis;Deep Learning as a Service;Big Data;Deep Neural Networks;Accuracy}, 
    doi={10.1109/TSC.2019.2928551}, 
    ISSN={1939-1374}, 
    month={},
}

@INPROCEEDINGS{GTDLBenchBigData, 
    author={{Wu}, Yanzhao and and {Cao}, Wenqi and {Sahin}, Semih and {Liu}, Ling}, 
    booktitle={2018 IEEE 38th International Conference on Big Data}, 
    title="{Experimental Characterizations and Analysis of Deep Learning Frameworks}", 
    year={2018},
    month={December},
}

Total image count = 400

Individuals count = 40

Each person has 10 face samples, all of the 10 face images are kept in a separate directory. Like S1, S2, ..., S40

For training dataset, I would take 6 images

For validation dataset, I would take 2 images

For test dataset, I would take 2 images

I need to make sure, all of the 40 human samples are in the 'train'/'validation'/ and 'test' dataset in exact 6:2:2 ratio. I need to randomize the dataset (seed=42 for reproducibility) before using for training/validation/testing. The model will not see the 'test' dataset during training. Therefore, the accuracy on the test dataset in inference should be unbiased.

sample characteristics:

resolution: 92 x 112

sensors used: The ORL dataset consists of standard, visible-spectrum grayscale images of faces that were captured with a simple camera. No special or advanced sensors were used to create this classic dataset. 

Slight variations in lighting. Images of the same person has some head rotation in different directions i.e. they are not facing the camera directly. 

IN some of the images the person is wearing a glass. However, not everyone is wearing a glass.

Mount open and mouth closed. I can sense different emotion. Anger-Smile etc.

I have uploaded a baseline code. Your feedback will be appreciated.

