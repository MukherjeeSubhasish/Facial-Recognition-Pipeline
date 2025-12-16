# Part 4

## A justification of the choice of classifier

This project does not use a conventional classifier by design.

Face recognition is an open-set problem, where identities in the validation/test set are completely unseen during training. A softmax-based classifier assumes a closed set of classes and therefore cannot generalize to new identities.

Instead, the model learns a facial embedding space using contrastive learning (CosineEmbeddingLoss) and ArcFace. In this setup:

* Images of the same identity are pulled closer in embedding space.

* Images of different identities are pushed farther apart.

* Recognition is performed via similarity search, not label prediction.

This approach is consistent with state-of-the-art systems such as FaceNet, ArcFace, and CosFace, and is fundamentally better suited for identity generalization than a fixed classifier head.

A supervised softmax classifier was used only as a sanity check, not as the final solution.

## A classification accuracy achieved on the training and validation subsets.

Setup: Train/validation split within the training identities

Model: Same ResNet50 backbone

Result: ~96% validation accuracy

This confirms that the training pipeline, preprocessing, and model implementation are correct. It's important to note that the model knows the output labels for the 'validation' set, because, it has already seen these identities in the 'train' dataset. This makes the validaiton task a lot easier.

## Ideas for improvements

I saw a high 'validation' accuracy of 96% when I split the train dataset into 'train' and 'validation' such that, the problem statement is converted into a supervised learning. Although, our main goal is to test this model on unseen labels, this exercise was done as a sanity check. If I would like to improve the accuracy further, I would experiment with the following-

* Increase the dimension of the embeddings. Currenly, I have tried  with 1024 as the maximum embedding dimension and it gave me the best response so far.
* Train for more epochs for better learning. I did not see a saturation of the accuracy metrics while training, but since the dataset is quite big, I eventually had to stop at a point.
* Increase batch size if possible. Currrently, I am limited by the GPU memory capacity. This would help me with faster training.

## How to run the code for train-validation split? This is the classification task on the seen labels. Unseen label experiment will be explained in the TASK5

To run this project, you will need Python 3.8 or later along with PyTorch, torchvision, HuggingFace datasets, scikit-learn, matplotlib, and SciPy. The dataset used for this project—chronopt-research/cropped-vggface2-224—is approximately 20 GB, so downloading it may take some time depending on your download speed. Once installed, please ensure that you have sufficient GPU memory; the experiments here were conducted on an NVIDIA H100 GPU using a batch size of 384, which is the maximum batch size that fits without causing out-of-memory errors. Running the script will automatically load the dataset, preprocess images using resizing, centercropping, normalization, construct positive and negative training pairs for contrastive learning, fine-tune a ResNet50 model for 5 epochs, extract embeddings from the validation set, run KMeans clustering to estimate identity-separation accuracy, and generate t-SNE plots demonstrating the structure of the embedding space. The script also saves training logs and the fine-tuned model checkpoint for future use. 

To execute the pipeline, simply run the Python script with below command. Log files containing training progress, validation accuracy, and timing information are provided for convenience and reproducibility in the "FINAL" directory.

```bash
python FINAL/sanity_supervised_learning.py
```

# Part5

## Description of the test dataset

The test database used in this project is the validation split of the VGGFace2 (cropped-224) dataset, downloaded from HuggingFace.

Size of the test database:

* Approximately 169,178 images

* 500 unique identities

Image properties:

* Resolution: 224 × 224

* Color: RGB

* Source: Web images (uncontrolled conditions)

## Why the 'test dataset' is significantly different?

* Test identities were never seen during training.

* Faces include different poses, ages, lighting conditions, and image quality distributions.

* The model has no label-level supervision for test identities.

## Ideas for improvements

I got 79% accuracy for the complete 'test' dataset with 169k images and 500 unseen labels.

As the number of identities increases, the embedding space must separate more classes.

Random chance accuracy drops sharply (≈0.2% for 500 classes).

Contrastive learning without explicit class supervision is inherently harder than supervised classification.

Thus, lower accuracy at scale is expected and not a failure.

Key Observations

* Increasing embedding dimension from 128 → 1024 improved accuracy by ~7%.

* Positive/negative pair balance is critical:

* Reducing pos_prob from 0.5 to 0.1 caused ~7% accuracy drop.

* ArcFace significantly improves inter-class separation compared to plain contrastive loss.

How accuracy might be improved?

* Train for more epochs (currently limited by dataset size and compute time).

* Add stronger data augmentations (RandomResizedCrop, HorizontalFlip).

* Explore harder losses (Triplet Loss, harder negative mining).

* Further tuning of embedding dimension and sampling strategy.

Overall, the achieved performance is reasonable given the open-set setting, and results clearly improve with stronger metric-learning objectives.
