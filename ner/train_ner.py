from spacy.training import Example
from .dataset_loader import generate_animal_sentences
from .ner_model import model
import warnings

warnings.filterwarnings("ignore")

# Prepare training data
train_data = generate_animal_sentences(1000)

examples = []
for text, annotations in train_data:
    doc = model.make_doc(text)
    example = Example.from_dict(doc, annotations)
    examples.append(example)

model.initialize(lambda: examples)

best_loss = float("inf")
patience = 5
counter = 0
learning_rate = 0.001  # Start with a smaller LR

for epoch in range(30):
    losses = {}
    for example in examples:
        dropout = min(0.1 + epoch * 0.005, 0.3)  # Starts at 0.1, maxes at 0.3
        model.update([example], losses=losses, drop=dropout)

    print(f"Epoch {epoch + 1}, Loss: {losses['ner']}")

    # Learning rate scheduling: If loss doesn't improve, reduce LR
    if losses['ner'] < best_loss:
        best_loss = losses['ner']
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            learning_rate *= 0.5  # Reduce LR if loss plateaus
            counter = 0  # Reset counter
            print(f"Reducing learning rate to {learning_rate}")

    if counter >= patience:
        print(f"Training stopped after {patience} non-improving epochs.")
        break

# Save the trained model
model.to_disk("../models/ner_model")
