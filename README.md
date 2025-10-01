Conceptual Design

Dataset Choice

Originally, I considered using the ORL Faces dataset because it provides a simple hierarchical mapping of users to images. However, for deep learning–based approaches like CNNs and Vision Transformers, the dataset size is too small to effectively train and evaluate complex models.

To address this, I will:

	• Use a larger, publicly available face dataset (e.g., LFW, VGGFace2, or MS-Celeb-1M). I will finalize this within 7 days [download and data processing and some sanity checks]
	• Retain the ORL Faces dataset for small-scale experimentation purpose.
	
This hybrid approach ensures that:

	1. A large dataset provides robust pre-trained weights.
	2. My smaller dataset provides controlled testing and domain-specific validation.

Step 1: CNN-Based Face Recognition Pipeline

	1. Load Dataset
		○ Use a large-scale dataset (e.g., VGGFace2).
		○ Preprocess images: face detection, alignment, resizing (e.g., 224×224 for CNNs).
		
	2. CNN Backbone (Feature Extractor)
		○ Use a pre-trained CNN such as AlexNet, ResNet-50, or FaceNet.
		○ Modify architecture:
			§ Remove the final classification layer.
			§ Keep the embedding (feature vector) layer before classification.
		○ This embedding serves as a compact representation of each face.
		
	3. Embedding-Based Recognition
		○ For each known person, store the embedding vector(s).
		○ For a new face, compute its embedding and compare against stored embeddings using distance metrics (e.g., cosine similarity, Euclidean distance).
		○ Classification is then based on nearest-neighbor matching in embedding space, rather than a fixed classifier.

Step 2: Vision Transformers (ViT)

	1. Use a Vision Transformer pretrained on a large dataset (e.g., ImageNet or face-specific datasets).
	2. Without fine-tuning:
		○ Extract embeddings from the transformer’s final layer.
		○ Perform recognition via embedding similarity (same as CNN pipeline).
	3. With fine-tuning:
		○ Fine-tune the transformer using my smaller dataset (ORL) for adaptation to domain-specific conditions.

Step 3: Real-Time Face Recognition 
(I am not sure how much time will the webcam hookup take, but I will definite give it a shot)

	1. Webcam Integration
		○ Connect OpenCV to capture frames in real time.
		○ Apply the CNN/ViT pipeline to extract embeddings for detected faces.
		
	2. Database of Embeddings
		○ Maintain embeddings for myself and a set of known friends.
		○ Perform real-time matching of incoming frames to stored embeddings.
		
	3. Robustness Evaluation
		○ Test recognition accuracy under different conditions:
			§ Frontal vs. non-frontal images
			§ Varying lighting conditions
			§ Partial occlusion (glasses, masks, etc.)

Step 4: Evaluation

	1. Training/Testing Split
		○ For large dataset: standard 80/20 split.
		○ For ORL Faces: 90/10 split (to replicate classical pipeline).
		
	2. Metrics
		○ Recognition accuracy
		○ Embedding similarity thresholds (ROC, FAR/FRR)
		○ Robustness tests with controlled variations




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

