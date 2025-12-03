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

PCA illustrations/ data visualizations——

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

Run command——

	python 07_orl_pca.py

	outputs are stored in "pca_visualizations" directory and "visualized_samples" directory

	env requirements are stored in requirements.txt file

	I haven't uploaded the VGGFace2 dataset because the size is very big. All the scripts with "orl" dataset should run properly. If you get any error, please let me know. I would show the results from my laptop then.




# ------------------------------------------------------------------
Full Technical Report (Task 4)
# ------------------------------------------------------------------

Contrastive Learning for Face Embeddings Using ResNet50 and VGGFace2 (Cropped)
This project focuses on learning discriminative facial embeddings using a contrastive-learning–based approach on the large-scale cropped_vggface2_224 dataset available from HuggingFace. The primary objective is not to perform conventional supervised classification but rather to learn a metric space in which images of the same individual are positioned close together while images of different individuals are pushed farther apart. This form of learning—driven by CosineEmbeddingLoss—allows the model to generalize to unseen identities, making it suitable for identity discrimination and similarity-based tasks. The dataset itself is substantial, approximately 20 GB, containing over 3 million training images, which introduces practical challenges associated with memory usage, storage, and training duration. In this work, the entire pipeline—from data loading to embedding generation, clustering evaluation, and visualization—is executed on an NVIDIA H100 GPU, leveraging its high computational throughput and memory bandwidth.

The design of the classifier or more precisely, the design of the embedding model—is grounded in the nature of the task. Face recognition often relies more on embedding quality rather than explicit classification. Models such as FaceNet, ArcFace, and CosFace demonstrate that learning a high-quality embedding space leads to better generalization than learning to predict labels using a softmax classifier. Because our goal is to evaluate the identity-separating capability of embeddings on completely unseen validation identities, a traditional classifier would be inadequate: a softmax classifier cannot predict unknown classes. 

The contrastive-learning paradigm, however, naturally supports such open-set scenarios. CosineEmbeddingLoss produces embeddings that encode identity similarity through geometric proximity. If two embeddings have a high cosine similarity (close to +1), the model interprets them as belonging to the same identity; if the similarity is low (or negative), they represent different identities. This makes contrastive learning a suitable foundation for unsupervised clustering-based evaluation.

To train a contrastive model, the input pipeline constructs pairs of images, each labeled either positive (same identity) or negative (different identities). The code includes a custom dataset class, PairDataset, that performs this pairing on the fly. For a given anchor image, the dataset selects either a positive pair by sampling another image with the same label or a negative pair by sampling an image with a different label. The sampling probability is governed by the hyperparameter pos_prob, which is set to 0.5. This means half of the training pairs are positives and the other half negatives. Interestingly, experiments show that reducing this probability to 0.1—meaning only 10% positive pairs—causes accuracy to drop by about 7%. The explanation for this effect lies in the balance of training signals: when positive pairs become too scarce, the model receives excessively strong signals to push embeddings apart but insufficient signals to pull same-identity embeddings together. The imbalance destabilizes the learned representation, causing identity clusters to become less cohesive. Maintaining a more balanced ratio between positive and negative examples ensures the embedding space develops both attractive and repulsive forces, thereby yielding clearer cluster boundaries.

Before feeding images to the model, the preprocessing pipeline applies a sequence of transformations. These include Resize(256) followed by CenterCrop(224), which align images with the training requirements of ResNet50, ensuring consistent framing and input dimensions. Converting images from PIL format to tensors standardizes pixel representation, while applying ImageNet normalization centers and scales pixel distributions, making training more stable. Without such normalization, pretrained weights (which expect normalized inputs) would not perform as intended. This preprocessing pipeline ensures that training proceeds efficiently while respecting the assumptions embedded in the pretrained model.

The experiments also investigate the effect of embedding dimension. Initially, embeddings were restricted to 128 dimensions, but increasing this to 1024 dimensions resulted in approximately 7% improvement in clustering accuracy. A larger embedding dimension allows the model to represent more nuanced identity-related variations, especially important in a dataset as diverse and wide-ranging as VGGFace2. High-dimensional embeddings create a richer geometric space for separating identities. However, increasing embedding size also increases memory consumption and computational overhead. The NVIDIA H100 permits this scaling thanks to its large memory capacity, but the effect must still be balanced against downstream application needs and storage constraints.

Computational constraints significantly shape how the system is configured. The batch size of 384 was chosen specifically to maximize the GPU memory utilization on the H100. Attempts to exceed this batch size consistently resulted in out-of-memory errors, indicating that 384 was the upper practical limit for this model architecture, embedding dimension, and training transforms. Larger batch sizes are favored when training contrastive-learning models because they allow more diverse and informative positive/negative pairings within each batch. However, their feasibility is directly tied to GPU memory availability, especially when embedding layers are large and backbone architectures like ResNet50 have high intermediate feature map memory footprints.

Because the VGGFace2 dataset is extremely large—over 3 million images—training on the full dataset is highly time-consuming. Even running on just 10% of the dataset requires several hours of GPU time. Extrapolating from these experiments, training the full 3 million images over five epochs is expected to take nearly an entire day of continuous computation, even on modern accelerators like the H100. For this reason, early experiments are conducted on subsets of the dataset, and only five epochs of training are used. While increasing the number of epochs would likely improve clustering accuracy further by allowing the embedding space to converge more thoroughly, doing so incurs substantial time and computational costs. Thus, training depth must be carefully balanced against available resources.

Evaluation of the learned embedding space is performed not through classification accuracy but via an unsupervised clustering approach using KMeans. Because the validation images come from identities the model has never seen before, KMeans serves as a natural mechanism for grouping embeddings based on similarity. The number of clusters is set equal to the number of identity labels in the validation split. After KMeans assigns a cluster index to each embedding, the next challenge is to measure accuracy. Since clustering labels are inherently arbitrary (cluster 0 does not necessarily correspond to identity 0), direct comparison is impossible. Instead, a contingency matrix—a cross-tabulation of true labels versus predicted clusters—is constructed. The Hungarian algorithm is applied to this matrix to determine the optimal one-to-one matching between clusters and true identities, maximizing correct assignments. The resulting matched sum represents the best achievable accuracy under label permutation, and dividing it by the total number of samples produces the final clustering accuracy. This metric is well-justified because it evaluates whether embeddings form identity-coherent clusters without penalizing label permutation, which is irrelevant in unsupervised settings.

The overall clustering accuracy on the validation set is moderate but reasonable given the difficulty of the task. The validation identities are completely unseen during training, and the model has no direct supervised guidance for those labels. This is inherently more challenging than supervised classification, where labels are known and fixed. Indeed, when the same ResNet50 model is trained in a fully supervised mode using softmax classification—taking a subset of the training dataset, splitting it into train and validation sets, and evaluating classification accuracy—the results exceed 95% accuracy. This striking difference underscores the difficulty of learning identity-specific features solely through pairwise contrastive signals without explicit class supervision. Logs and full code for the supervised experiment are included separately for reference and comparison.
One small but effective improvement planned before the final examination is integrating RandomResizedCrop and RandomHorizontalFlip into the augmentation pipeline. These augmentations simulate variations in pose, zoom, and orientation, forcing the embedding model to focus on identity-consistent features rather than superficial aspects such as exact cropping or image alignment. They are easy to implement, computationally inexpensive, and widely used in contrastive learning frameworks such as SimCLR. This enhancement should meaningfully improve generalization and yield clearer identity clusters.

How to run the file?
To run this project, you will need Python 3.8 or later along with PyTorch, torchvision, HuggingFace datasets, scikit-learn, matplotlib, and SciPy. The dataset used for this project—chronopt-research/cropped-vggface2-224—is approximately 20 GB, so downloading it may take some time depending on your download speed. Once installed, please ensure that you have sufficient GPU memory; the experiments here were conducted on an NVIDIA H100 GPU using a batch size of 384, which is the maximum batch size that fits without causing out-of-memory errors. Running the script will automatically load the dataset, preprocess images using resizing, centercropping, normalization, construct positive and negative training pairs for contrastive learning, fine-tune a ResNet50 model for 5 epochs, extract embeddings from the validation set, run KMeans clustering to estimate identity-separation accuracy, and generate t-SNE plots demonstrating the structure of the embedding space. The script also saves training logs and the fine-tuned model checkpoint for future use. To execute the pipeline, simply run the Python script with below command. Log files containing training progress, validation accuracy, and timing information are provided for convenience and reproducibility in the "FINAL" directory.

------------------------
python FINAL/main.py   |
------------------------

How to improve accuracy further for the test dataset? 
Since validation class = 500, therefore random chance leads us to 0.2% hit rate. I have observed as the number of classes increases as I go from smaller dataset to larger dataset the accuracy drops from 71% to 15%. This can be improved with higher epochs, higher embedding size, finetuning the "Positive_prob" variable from (0,1) range, using complex models, using different loss functions like 'triplet loss' etc. Due, to the massive size of the entire dataset (3M) I could only do one full dataset simulation with 5epochs only since it took almost a day to complete.
For sanity check I did one supervised learning experiment where I split the train dataset into two parts-training and validation and used the same model, to make sure that my training pipeline is NOT broken. It gave me 96% accuracy on that experiment showing that the model is learning properly. 
The code and log file are uploaded in the "FINAL" directory— "sanity_supervised_learning.py" and "sanity_supervised_learning.log"

# ----------------------------------------------------------------------------------
TASK3 dataset PCA/t-SNE analysis work(pending) can be found in the FINAL/task3 directory.
# ----------------------------------------------------------------------------------


