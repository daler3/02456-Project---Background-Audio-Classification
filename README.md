# 02456-Project-Background-Audio-Classification
This project treats the topic of sound event classification using Convolutional Neural Networks, analyzing and trying to improve the architecture proposed by K. J. Piczak. The network consists of 2 convolutional layers with max-pooling followed by two fully connected layers, and it is trained using log powered mel-spectrograms and their delta features. Moreover, the architecture is modified to perform multilabel sound classification, such as the classification of two different simultaneous sound events.

The dataset used to evaluate the architecture is UrbanSound8K. This dataset has also been used to create two new synthetic datasets (by overlaying two sound events) to test the performances of the multilabel classifier.

The implementation part has been done with Keras using a Tensorflow backend. The obtained architecture performs similar to the original one with respect to the single-labelled sounds. In the multilabel case, the classifier is able to classify both sounds in 14% of the cases, while one of the two sounds is recognized in most of the cases. We also show that pretraining the multilabel classifier with single-labelled sounds from the original dataset (and successively training it with overlaying sounds) helps in achieving a better accuracy, especially when the sound segment to be classified is composed by a single event.  
