# Task 1: MNIST Classification

This project is a simple implementation of the MNIST dataset classification.
The project is divided into 3 classes, each of which represents a different model for classification:

1) Feed-Forward Neural Network;
2) Convolutional Neural Network;
3) Random Forest.

The project is implemented using the TensorFlow library for neural networks and the scikit-learn library for the Random
Forest model.

###  Task 1: Feed-Forward Neural Network
The Feed-Forward Neural Network is implemented using the TensorFlow library. The model consists of 3 layers: an input
layer, a hidden layer, and an output layer. The input layer has 784 neurons, which correspond to the 28x28 pixels of the
input images. The hidden layer has 128 neurons, and the output layer has 10 neurons, which correspond to the 10 classes
of the MNIST dataset. The model uses the ReLU activation function for the hidden layer and the softmax activation
function for the output layer. The model is trained using the Adam optimizer and the categorical cross-entropy loss
function.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*SfRJNb5dOOPZYEFY5jDRqA.png" width=500 height=250/>
Credit: <a href="https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06">image source</a>

### Task 2: Convolutional Neural Network
The Convolutional Neural Network is implemented using the TensorFlow library. The model consists of 2 convolutional
layers, 2 max-pooling layers, flattennig layer and 2 dense layers. Like the Feed-Forward Neural Network, the model uses 
the ReLU activation function for the convolutional and dense layers and the softmax activation function for the output
layer, and is trained using the Adam optimizer and the categorical cross-entropy loss function.

<img src="https://goodboychan.github.io/images/CNN_MNIST.png" width=500 height=150/>
Credit: <a href="https://goodboychan.github.io/python/deep_learning/tensorflow-keras/2020/10/10/01-CNN-with-MNIST.html">image source</a>

### Task 3: Random Forest
The Random Forest model is implemented using the scikit-learn library. The Random forest classifier creates a set of decision trees from a randomly selected subset of the training set. It is a set of decision trees from a randomly selected subset of the training set and then It collects the votes from different decision trees to decide the final prediction.

<img src="https://www.ris-ai.com/static/images/models/random-forest-algorithm.jpg" width=500 height=350/>
Credit: <a href="https://www.ris-ai.com/random-forest-algorithm">image source</a>

## Setup
To set up the project, follow these steps:

1. Clone the repository:
   

2. Install the required libraries: 
    ```bash
    pip install -r requirements.txt
    ```
3. Run the demo notebook:
   ```bash
   jupyter notebook MNIST_Classification.ipynb
   ```
