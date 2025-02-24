import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

model = tf.keras.models.load_model("../models/wildcat_cnn_model.h5")
dataset = pd.read_csv("../data/WILDCATS.CSV")


# Load the image
def get_random_image():
    """
    Get a random image from the dataset.
    :return: Path to the image.
    """
    random_image = dataset.sample(1)
    image_path = random_image.iloc[0]["filepaths"]
    image_path = f"../data/{image_path}"
    return image_path


def predict(image_path):
    """
    Predict the class of the given image.
    :param image_path: Path to the image.
    :return: Predicted class name.
    """
    # Load the image and preprocess it
    image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = tf.expand_dims(input_arr, 0)
    input_arr = input_arr / 255
    # Get the predictions
    predictions = model.predict(input_arr)
    # Get classes and convert to lowercase
    class_labels = list(dataset["labels"].unique())
    class_labels = [label.lower() for label in class_labels]
    pred_index = tf.argmax(predictions, axis=1).numpy()[0] # Get the index of the prediction (highest probability)
    prediction = class_labels[pred_index] # Get the class name
    if prediction == "lions":  # As the label is "lions" in the dataset, we need to convert it to "lion"
        prediction = "lion"
    return prediction


if __name__ == "__main__":
    image_path = get_random_image()  # Get a random image
    prediction = predict(image_path) # Predict the class of the image
    plt.imshow(tf.keras.preprocessing.image.load_img(image_path)) # Plot the image
    plt.title(f"Predicted: {prediction}, actual: {image_path.split('/')[-2].lower()}") # Set the title
    plt.show()
