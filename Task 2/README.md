# Task 2. NER with Image Classification

## Problem Statement
1. Get an image.
2. User enters a text query with guess of the animal in the image.
3. The classifier should predict the animal in the image.
4. NER should extract the animal name from the text query.
5. Compare the results of the classifier and NER.

## Approach
### Dataset
For the classification task, I found a dataset on Kaggle which contains images of 10 different
wildcats. The dataset can be found [here](https://www.kaggle.com/datasets/gpiosenka/cats-in-the-wild-image-classification/data).
The dataset contains images of the following wildcats:
1. African leopard
2. Caracal
3. Cheetah
4. Clouded leopard
5. Jaguar
6. Lion
7. Ocelot
8. Puma
9. Snow leopard
10. Tiger

### Image Classification
I used a pre-trained MobileNetV2 model to classify the images. I froze the first layers of the model and added pooling and dense layers on top of it. 
Also I added dropout layer to prevent overfitting. The model was trained for 25 epochs. 

### Named Entity Recognition
Honestly, I didn't have experience with NER before. I was trying to find any dataset for NER, but I couldn't find any.
So, following the tutorial on [this](https://www.geeksforgeeks.org/python-named-entity-recognition-ner-using-spacy/) outdated
tutorial, I used the `spacy` library to perform NER and I created a small dataset by myself. The dataset contains
some sentence examples with different animals. The dataset is really small, so the NER model may not work well.
I labeled the dataset manually and trained the NER model using the `spacy` library.

## Setup
1. Clone the repository.
2. Install the required libraries using the following command:
```bash 
pip install -r requirements.txt
```
3. Run either `pipeline.py` or `demo.ipynb` to see the results.

