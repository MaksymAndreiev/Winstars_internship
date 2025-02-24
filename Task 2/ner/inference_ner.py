import spacy

# Load the trained NER model
model = spacy.load("../models/ner_model")

def get_entities(text):
    """
    Get entities from the given text.
    :param text: input text in form of string
    :return: dictionary of entities with their labels
    """
    doc = model(text)
    entities = []
    for ent in doc.ents:
        entities.append({"entity": ent.text, "label": ent.label_})
    return entities

# Example sentences
example_sentences = [
    "A dog was resting near the lake.",
    "Have you ever seen a tiger in the wild?",
    "The elephant at the zoo was very active.",
    "My friend has a pet cat at home.",
    "The baby seal stayed close to its mother.",
    "Scientists are studying the behavior of the dolphin.",
    "The bear made a loud noise at night.",
    "In some cultures, owl is considered sacred.",
    "A lion was spotted near the river.",
    "Children love to draw pictures of horse.",
    "The rabbit was searching for food in the grass.",
    "A squirrel approached cautiously, sensing danger.",
    "I guess it's a fox.",
    "I see a wolf in the image.",
    "Gepard",
    "This is a flamingo.",
]

# Process each sentence and extract entities
for number, sentence in enumerate(example_sentences, 1):
    entities = get_entities(sentence)
    print(f"{number}. {sentence}: {entities}")
