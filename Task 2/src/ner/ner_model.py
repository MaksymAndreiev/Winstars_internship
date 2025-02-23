import spacy
# Load a blank English model
model = spacy.blank('en')

# Create a new entity recognizer and add it to the pipeline
if 'ner' not in model.pipe_names:
    ner = model.add_pipe('ner')

# Add the label 'ANIMAL' to the entity recognizer
ner.add_label('ANIMAL')
