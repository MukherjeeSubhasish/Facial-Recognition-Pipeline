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


## A short commentary related to the observed accuracy and ideas for improvements.


## How to run the code for train-validation split? This is the classification task on the seen labels. Unseen label experiment will be explained in the TASK5.

To run this project, you will need Python 3.8 or later along with PyTorch, torchvision, HuggingFace datasets, scikit-learn, matplotlib, and SciPy. The dataset used for this project—chronopt-research/cropped-vggface2-224—is approximately 20 GB, so downloading it may take some time depending on your download speed. Once installed, please ensure that you have sufficient GPU memory; the experiments here were conducted on an NVIDIA H100 GPU using a batch size of 384, which is the maximum batch size that fits without causing out-of-memory errors. Running the script will automatically load the dataset, preprocess images using resizing, centercropping, normalization, construct positive and negative training pairs for contrastive learning, fine-tune a ResNet50 model for 5 epochs, extract embeddings from the validation set, run KMeans clustering to estimate identity-separation accuracy, and generate t-SNE plots demonstrating the structure of the embedding space. The script also saves training logs and the fine-tuned model checkpoint for future use. 

To execute the pipeline, simply run the Python script with below command. Log files containing training progress, validation accuracy, and timing information are provided for convenience and reproducibility in the "FINAL" directory.

----------------------------------------------|
python FINAL/sanity_supervised_learning.py    |
----------------------------------------------|
