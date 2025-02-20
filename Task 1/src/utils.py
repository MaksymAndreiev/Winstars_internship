import tensorflow as tf


def load_mnist():
    mnist_dataset = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

    # Normalize the data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Add a channel dimension
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return x_train, y_train, x_test, y_test