import os
import io
import numpy as np
from PIL import Image
import tensorflow as tf
 
IMG_SIZE = (150, 150)
MODEL_PATH = os.environ.get("BT_MODEL_PATH", "brain_tumor_model.h5")
LABELS_PATH = os.environ.get("BT_LABELS_PATH", "class_labels.txt")
 
_model = None
_labels = None
 
 
def _load_labels(path=LABELS_PATH):
    labels = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                idx, name = line.strip().split(",", 1)
                labels[int(idx)] = name
    else:
        # Fallback default ordering if class_labels.txt is missing
        labels = {0: "glioma", 1: "meningioma", 2: "no_tumor", 3: "pituitary"}
    return labels
 
 
def load_resources(model_path=MODEL_PATH, labels_path=LABELS_PATH):
    """
    Loads the model and label map once, caching them in module-level
    globals so repeated predictions don't reload from disk.
    """
    global _model, _labels
    if _model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. Train a model first "
                f"(see brain_tumor_prediction.py --train) or set BT_MODEL_PATH."
            )
        _model = tf.keras.models.load_model(model_path)
    if _labels is None:
        _labels = _load_labels(labels_path)
    return _model, _labels
 
 
def preprocess_image(image_bytes):
    """
    Converts raw uploaded image bytes into a normalized numpy array
    ready for the model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # add batch dimension
    return arr
 
 
def predict(image_bytes):
    """
    Runs a prediction on a single image (given as raw bytes, e.g. from
    an uploaded file). Returns a dict:
 
    {
        "predicted_class": "glioma",
        "confidence": 0.9421,
        "probabilities": {
            "glioma": 0.9421,
            "meningioma": 0.031,
            "no_tumor": 0.019,
            "pituitary": 0.0079
        }
    }
    """
    model, labels = load_resources()
    arr = preprocess_image(image_bytes)
 
    preds = model.predict(arr, verbose=0)[0]
    class_idx = int(np.argmax(preds))
 
    probabilities = {
        labels.get(i, f"class_{i}"): float(round(p, 4))
        for i, p in enumerate(preds)
    }
 
    return {
        "predicted_class": labels.get(class_idx, f"class_{class_idx}"),
        "confidence": float(round(preds[class_idx], 4)),
        "probabilities": probabilities,
    }
 
 
if __name__ == "__main__":
    # Quick manual test: python prediction.py path/to/image.jpg
    import sys
    if len(sys.argv) != 2:
        print("Usage: python prediction.py <image_path>")
        sys.exit(1)
 
    with open(sys.argv[1], "rb") as f:
        result = predict(f.read())
 
    print(f"Predicted class: {result['predicted_class']}")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("Probabilities:")
    for name, prob in result["probabilities"].items():
        print(f"  {name}: {prob * 100:.2f}%")
