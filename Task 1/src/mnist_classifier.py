from .model_interface import MnistClassifierInterface
from .cnn import ConvolutionalNeuralNetwork
from .random_forest import RandomForestClassifier
from .feed_forward_nn import FeedForwardNeuralNetwork


class MnistClassifier(MnistClassifierInterface):
    def __init__(self, algorithm):
        if algorithm == 'cnn':
            self.algorithm = ConvolutionalNeuralNetwork()
        elif algorithm == 'rf':
            self.algorithm = RandomForestClassifier()
        elif algorithm == 'nn':
            self.algorithm = FeedForwardNeuralNetwork()
        else:
            raise ValueError("Invalid algorithm name. Please use 'cnn', 'rf' or 'nn'.")

    def train(self, x_train, y_train):
        self.algorithm.train(x_train, y_train)

    def predict(self, x_test):
        return self.algorithm.predict(x_test)
