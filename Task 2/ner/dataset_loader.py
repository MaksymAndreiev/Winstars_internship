import random


def generate_animal_sentences(num_sentences=500):
    """
    Generate a dataset of sentences with animal names. Each sentence contains a single animal name. The animal name is
    annotated with the NER label "ANIMAL".
    :param num_sentences: Number of sentences to generate
    :return: List of tuples with sentence and NER annotations
    """
    sentence_templates = [
        "I saw a {animal} in the forest.",
        "The {animal} is running across the field.",
        "A {animal} was resting near the lake.",
        "Farmers take care of their {animal}s every day.",
        "Have you ever seen a {animal} in the wild?",
        "The {animal} at the zoo was very active.",
        "My friend has a pet {animal} at home.",
        "A group of {animal}s was moving together.",
        "I love watching documentaries about {animal}s.",
        "The baby {animal} stayed close to its mother.",
        "Scientists are studying the behavior of {animal}s.",
        "The {animal} jumped over the fence.",
        "A {animal} can be found in many different climates.",
        "People used to ride {animal}s for transportation.",
        "The {animal} made a loud noise at night.",
        "In some cultures, {animal}s are considered sacred.",
        "A {animal} was spotted near the river.",
        "Children love to draw pictures of {animal}s.",
        "The {animal} was searching for food in the grass.",
        "A {animal} approached cautiously, sensing danger.",
        "I guess it's a {animal}.",
        "I see a {animal} in the image.",
        "{animal}",
        "{animal}.",
        "This is a {animal}.",
        "A {animal}.",
        "Idk, maybe a {animal}.",
        "I saw an {animal} in the forest.",
    ]

    animal_list = [
        "dog", "cat", "elephant", "lion", "tiger", "zebra", "giraffe", "panda", "kangaroo", "rhinoceros",
        "horse", "cow", "sheep", "goat", "deer", "monkey", "fox", "wolf", "bear", "rabbit",
        "squirrel", "ocelot", "badger", "beaver", "chipmunk", "cheetah", "leopard", "jaguar", "hyena", "coyote",
        "bison", "moose", "buffalo", "porcupine", "armadillo", "hedgehog", "raccoon", "opossum", "sloth", "antelope",
        "camel", "donkey", "mule", "boar", "ferret", "weasel", "skunk", "lynx", "caribou", "wolverine",
        "pangolin", "tapir", "meerkat", "gazelle", "caracal", "platypus", "narwhal", "dolphin", "whale", "seal",
        "walrus", "manatee", "puma", "albatross", "flamingo", "owl", "eagle", "hawk", "falcon", "vulture",
        "parrot", "toucan", "woodpecker", "peacock", "crane", "heron", "stork", "robin", "sparrow", "swallow",
        "bat", "python", "cobra", "viper", "alligator", "crocodile", "gecko", "chameleon", "iguana", "monitor lizard",
        "tortoise", "turtle", "frog", "toad", "salamander", "newt", "axolotl", "starfish", "jellyfish", "seahorse",
        "otter", "gepard", "penguin", "koala", "kookaburra", "platypus", "kangaroo", "wallaby", "tasmanian devil",
        "wombat"
    ]

    dataset = []
    if num_sentences > len(sentence_templates):
        limit = num_sentences // len(sentence_templates)
    else:
        limit = 1
    sentence_count = {}
    for _ in range(num_sentences):
        sentence = random.choice(sentence_templates)
        if sentence_count.get(sentence, 0) >= limit and sentence not in ["{animal}", "{animal}."]:
            sentence_templates.remove(sentence)
            if not sentence_templates:
                break
            sentence = random.choice(sentence_templates)
        sentence_count[sentence] = sentence_count.get(sentence, 0) + 1
        animal_start = sentence.find("{animal}")
        animal = random.choice(animal_list)
        animal_end = animal_start + len(animal)
        sentence = sentence.format(animal=animal)
        ner_annotation = {"entities": [(animal_start, animal_end, "ANIMAL")]}
        dataset.append((sentence, ner_annotation))

    return dataset
