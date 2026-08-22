import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
 
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 25
NUM_CLASSES = 4  # glioma, meningioma, pituitary, no_tumor
MODEL_PATH = "brain_tumor_model.h5"
 
 
# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------
def get_data_generators(train_dir, val_dir=None, val_split=0.2):
    """
    Creates training and validation data generators with augmentation.
    If val_dir is None, splits train_dir using val_split.
    """
    if val_dir:
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
        )
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
 
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=True,
        )
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False,
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=val_split,
        )
        train_gen = datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )
        val_gen = datagen.flow_from_directory(
            train_dir,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )
 
    return train_gen, val_gen
 
 
# ---------------------------------------------------------------------------
# 2. Model architecture
# ---------------------------------------------------------------------------
def build_model(num_classes=NUM_CLASSES, input_shape=(150, 150, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
 
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
 
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
 
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
 
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
 
 
# ---------------------------------------------------------------------------
# 3. Training
# ---------------------------------------------------------------------------
def train_model(data_dir, val_dir=None, epochs=EPOCHS, model_path=MODEL_PATH):
    train_gen, val_gen = get_data_generators(data_dir, val_dir)
    num_classes = train_gen.num_classes
 
    model = build_model(num_classes=num_classes)
    model.summary()
 
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7
        ),
        tf.keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_accuracy", save_best_only=True
        ),
    ]
 
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
    )
 
    # Save class label mapping alongside the model
    class_indices = train_gen.class_indices
    labels = {v: k for k, v in class_indices.items()}
    with open("class_labels.txt", "w") as f:
        for idx in sorted(labels):
            f.write(f"{idx},{labels[idx]}\n")
 
    plot_training_history(history)
    print(f"\nModel saved to {model_path}")
    print("Class labels saved to class_labels.txt")
    return model, history
 
 
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
 
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
 
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Training curves saved to training_history.png")
 
 
# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model_path, test_dir):
    model = tf.keras.models.load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False,
    )
    loss, acc = model.evaluate(test_gen)
    print(f"Test accuracy: {acc:.4f}, Test loss: {loss:.4f}")
 
    from sklearn.metrics import classification_report, confusion_matrix
    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes
    labels = list(test_gen.class_indices.keys())
 
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
 
 
# ---------------------------------------------------------------------------
# 5. Single-image prediction
# ---------------------------------------------------------------------------
def load_class_labels(path="class_labels.txt"):
    labels = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                idx, name = line.strip().split(",", 1)
                labels[int(idx)] = name
    return labels
 
 
def predict_image(model_path, image_path, labels_path="class_labels.txt"):
    model = tf.keras.models.load_model(model_path)
    labels = load_class_labels(labels_path)
 
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
 
    preds = model.predict(img_array)[0]
    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    class_name = labels.get(class_idx, f"class_{class_idx}")
 
    print(f"\nPrediction: {class_name}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print("\nAll class probabilities:")
    for i, prob in enumerate(preds):
        name = labels.get(i, f"class_{i}")
        print(f"  {name}: {prob * 100:.2f}%")
 
    return class_name, confidence
 
 
# ---------------------------------------------------------------------------
# 6. CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Prediction CNN")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on test set")
    parser.add_argument("--predict", action="store_true", help="Predict a single image")
    parser.add_argument("--data_dir", type=str, help="Path to training data directory")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to validation/testing directory")
    parser.add_argument("--test_dir", type=str, help="Path to test directory (for --evaluate)")
    parser.add_argument("--image", type=str, help="Path to image for prediction")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to saved model (.h5)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
 
    args = parser.parse_args()
 
    if args.train:
        if not args.data_dir:
            raise ValueError("--data_dir is required for training")
        train_model(args.data_dir, args.val_dir, epochs=args.epochs, model_path=args.model)
 
    elif args.evaluate:
        if not args.test_dir:
            raise ValueError("--test_dir is required for evaluation")
        evaluate_model(args.model, args.test_dir)
 
    elif args.predict:
        if not args.image:
            raise ValueError("--image is required for prediction")
        predict_image(args.model, args.image)
 
    else:
        parser.print_help()
