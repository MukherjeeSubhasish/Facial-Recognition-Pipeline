# Part 4

## A justification of the choice of classifier

This project does not use a conventional classifier by design.

Face recognition is an open-set problem, where identities in the validation/test set are completely unseen during training. A softmax-based classifier assumes a closed set of classes and therefore cannot generalize to new identities.

Instead, the model learns a facial embedding space using contrastive learning (CosineEmbeddingLoss) and ArcFace. In this setup:

Images of the same identity are pulled closer in embedding space.

Images of different identities are pushed farther apart.

Recognition is performed via similarity search, not label prediction.

This approach is consistent with state-of-the-art systems such as FaceNet, ArcFace, and CosFace, and is fundamentally better suited for identity generalization than a fixed classifier head.

A supervised softmax classifier was used only as a sanity check, not as the final solution.

## A classification accuracy achieved on the training and validation subsets.
## A short commentary related to the observed accuracy and ideas for improvements.
## How to run the code?
