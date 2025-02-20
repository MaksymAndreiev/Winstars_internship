from .model_interface import MnistClassifierInterface
import tensorflow as tf
import numpy as np


class ConvolutionalNeuralNetwork(MnistClassifierInterface):
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    def train(self, x_train, y_train):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy']
                           )
        self.model.fit(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

    def predict(self, x_test):
        return np.argmax(self.model.predict(x_test), axis=-1)
