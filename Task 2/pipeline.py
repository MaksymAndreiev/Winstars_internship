import tensorflow as tf
from matplotlib import pyplot as plt

from classification import get_random_image, predict
from ner import get_entities


def process_user_input(text, prediction):
    """

    :param text:
    :param prediction:
    :return:
    """
    entities = get_entities(text)
    for entity in entities:
        if entity["label"] == "ANIMAL":
            animal = entity["entity"]
            if animal in prediction.split():
                return True
    return False


if __name__ == "__main__":
    image_path = get_random_image()
    prediction = predict(image_path)
    # Plot the image
    plt.imshow(tf.keras.preprocessing.image.load_img(image_path))
    plt.show()
    # Get user input
    text = input("Enter a guess: ")
    result = process_user_input(text, prediction)
    if result:
        print("Correct! It's a {}!".format(prediction))
    else:
        print("Incorrect! This is {}.".format(prediction))
