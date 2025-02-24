from class_model import model
from dataset_loader import train_generator, val_generator, test_generator


# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    verbose=1
)

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('../../models/wildcat_cnn_model.h5')
