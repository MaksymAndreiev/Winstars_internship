from .model_interface import MnistClassifierInterface
from sklearn.ensemble import RandomForestClassifier as RandomForestModel


class RandomForestClassifier(MnistClassifierInterface):
    def __init__(self, manual=False):
        self.model = RandomForestModel()

    def train(self, x_train, y_train):
        x_train = x_train.reshape(x_train.shape[0], -1)
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        x_test = x_test.reshape(x_test.shape[0], -1)
        return self.model.predict(x_test)
