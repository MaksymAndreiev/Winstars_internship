from spacy.training import Example
from dataset_loader import generate_animal_sentences
from ner_model import model
import warnings

warnings.filterwarnings("ignore")

# Prepare training data
train_data = generate_animal_sentences(1000)
# Convert training data into spaCy Example objects
examples = []
for text, annotations in train_data:
    doc = model.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)

# Initialize the model with the training data
model.initialize(lambda: examples)

best_loss = float("inf")
patience = 5
counter = 0

# Train the model for a few iterations
for epoch in range(30):  # Train for 10 epochs
    losses = {}
    for example in examples:
        model.update([example], losses=losses, drop=0.5)
    print(f"Epoch {epoch + 1}, Loss: {losses['ner']}")
    if losses['ner'] < best_loss:
        best_loss = losses['ner']
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Training stopped after {patience} non-improving epochs.")
            break

# Save the trained model
model.to_disk("../../models/ner_model")
