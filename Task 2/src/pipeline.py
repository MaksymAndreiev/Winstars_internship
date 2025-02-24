import spacy
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

ner_model = spacy.load("../models/ner_model")
classification_model = tf.keras.models.load_model("../models/wildcat_cnn_model.h5")
dataset = pd.read_csv("../data/WILDCATS.CSV")


def get_random_image():
    random_image = dataset.sample(1)
    image_path = random_image.iloc[0]["filepaths"]
    image_path = f"../data/{image_path}"
    return image_path


def predict(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.expand_dims(input_arr, 0)
    input_arr = input_arr / 255
    predictions = classification_model.predict(input_arr)
    # Get classes ad convert to lowercase
    class_labels = list(dataset["labels"].unique())
    class_labels = [label.lower() for label in class_labels]
    pred_index = tf.argmax(predictions, axis=1).numpy()[0]
    prediction = class_labels[pred_index]
    if prediction == "lions":
        prediction = "lion"
    return prediction


def get_entities(text):
    doc = ner_model(text)
    entities = []
    for ent in doc.ents:
        entities.append({"entity": ent.text, "label": ent.label_})
    return entities


def process_user_input(text, prediction):
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
    print("-" * 50 + "DEBUG" + "-" * 50)
    print("Entities in the text: ", get_entities(text))
    print("-" * 50 + "DEBUG" + "-" * 50)
    result = process_user_input(text, prediction)
    if result:
        print("Correct! It's a {}!".format(prediction))
    else:
        print("Incorrect! This is {}.".format(prediction))
