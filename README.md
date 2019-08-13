# Siamese Sentence Similarity Classifier for pyTorch

## Overview
This repository contains a re-implementation of Mueller's et al., ["Siamese Recurrent Architectures for Learning Sentence Similarity."](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewFile/12195/12023) (AAAI, 2019). For the technical details, please refer to the publication.

## Training
To train the classifier, execute `similarity_estimator/training.py` after modifying the hard-coded values (such as the training corpus filename) to your own specifications.

## Evaluation
To evaluate the performance of a trained model, run the `similarity_estimator/testing.py` script. Again, adjust user-specific values as needed within the script itself.

## Note
This re-implementation was completed with personal use in mind and is, as such, not actively maintained. You are, however, very welcome to extend or adjust it according to your own needs, should you find it useful. Happy coding :) .
