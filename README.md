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


==============       DATA ACQUISITION AND PREPARATION      ==================

'filename': 

	'/home/smukher5/.cache/huggingface/datasets/chronopt-research___cropped-vggface2-224/default/0.0.0/dc48caf49ea0de02988f83e2130e7fc52bc1bff8/cropped-vggface2-224-train-00000-of-00040.arrow'


citation:

	@misc{cao2018vggface2datasetrecognisingfaces,
	      title={VGGFace2: A dataset for recognising faces across pose and age}, 
	      author={Qiong Cao and Li Shen and Weidi Xie and Omkar M. Parkhi and Andrew Zisserman},
	      year={2018},
	      eprint={1710.08092},
	      archivePrefix={arXiv},
	      primaryClass={cs.CV},
	      url={https://arxiv.org/abs/1710.08092}, 
	}


download website: 

	https://www.kaggle.com/datasets/hearfool/vggface2

	https://huggingface.co/datasets/chronopt-research/cropped-vggface2-224


dataset details---->>>>

DatasetDict({

    train: Dataset({
        features: ['image', 'label'],
        num_rows: 3138862
    })
    validation: Dataset({
        features: ['image', 'label'],
        num_rows: 169178
    })
})

dataset["train"][0] ---->>>>

	{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=224x224 at 0x7F8AD8342170>, 'label': 1}

	dataset_size = 20GB
	
	8631 different 'labels' are there in the training set.
	
	500 different 'labels' are there in the validation set.

Observation:

	The "train" and the "test" dataset do not have overlapping 'labels'. First, it felt like an error. 
	But, then I realized, if the 'train' and 'test' dataset overlaps, the model can simply remember 
	some face instead of recognizing them through learned feature detection.


Characterization of the samples-

	Image Size-
	
	224×224 pixels
	
	
	Color Channels-
	
	RGB (3-channel, visible light)
	
	
	Source-
	
	Web photographs (Google Image Search)
	
	
	Sensor Type-
	
	Varies (consumer-grade digital cameras, smartphones)
	
	
	Illumination-
	
	Mixed (daylight, indoor, flash, shadow)
	
	
	Ambient Conditions-
	
	Uncontrolled, highly diverse
	
	
	Labels-
	
	Unique identity IDs (non-overlapping across splits)

============================================================

========================FIRST UPDATE========================

============================================================

Data preprocessing and justification——


	1. transforms.Grayscale(num_output_channels=1)
		• Purpose: Ensures that every image is converted to grayscale (single channel).
		• Why:
			○ Some datasets or models might contain RGB (3 channels), while CNN expects single-channel input.
			○ Converting to grayscale simplifies computation and reduces parameters if color information is unnecessary.
		• Effect: Each pixel now has one intensity value instead of three (R, G, B).

	2. transforms.Resize((128, 128))
		• Purpose: Resizes all images to a uniform 128×128 pixels.
		• Why:
			○ Neural networks require a consistent input size.
			○ Makes batching possible and reduces computational cost if the original images are large.
		• Effect: Every image now has the same spatial dimensions, suitable for feeding into CNN.

	3. transforms.ToTensor()
		• Purpose: Converts the PIL image or NumPy array into a PyTorch tensor.
		• Why:
			○ Neural networks in PyTorch work with tensors, not images.
			○ Automatically scales pixel values from [0, 255] → [0.0, 1.0].
		• Effect:
			○ Data type changes to torch.FloatTensor.

	4. transforms.Normalize(mean=[0.5], std=[0.5])
		• Purpose: Normalizes the tensor so that pixel values are centered around 0.
		• Effect:
			○ Since input values are in [0, 1], this transforms them to approximately [-1, 1].
		• Why:
			○ Normalization helps the model converge faster and maintain stable gradients during training.

	Summary:
	We can convert every image into standardized 128×128 grayscale tensors with values scaled to [-1, 1], which is optimal for CNN input and training stability.

PCA illustrations——

	 1. What Each Point Represents?
		• Each dot in the plot corresponds to one face image from the ORL dataset.
		• The position of the dot is determined by the two most important principal components (PC1 and PC2), which are new axes capturing most of the variance (differences) in the images.
	
	 2. What the Colors Mean?
		• Each color represents one class label (a different person’s face).
		• In the original dataset, we have 40 people × 10 images each, so 40 unique classes.
		• The colorbar on the right shows which color corresponds to which class index (0–39).
	We can see eight distinct colors, it means only a subset of the classes (e.g., 8 persons × 5 images each) was visualized.
	
	 3. Analysis—
		• Principal Component 1 (x-axis): captures the direction of greatest variation among all faces.
			○ Images far apart along this axis differ the most in terms of key facial features.
		• Principal Component 2 (y-axis): captures the next most significant variation (independent of PC1).
	So the scatter plot shows how faces are distributed according to the two strongest factors of visual difference.
	
	 4. What Clusters Mean—
		• If points of the same color are close together, it means those images (same person) are visually similar — the PCA has captured consistent facial identity patterns.
		• If colors overlap heavily, that means some faces share similar features (maybe— pose, expression, lighting), making them harder to distinguish linearly.
	So:
		• Tight, well-separated clusters → dataset is easily separable by class (good for classification).
		• Overlapping clusters → more difficult dataset, possibly needs deeper nonlinear methods (CNN, transformers etc.) to separate classes.
	


