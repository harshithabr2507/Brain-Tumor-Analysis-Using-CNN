# Brain-Tumor-Analysis-Using-CNN
Brain Tumor Analysis Using CNN with Flask Web Application
import os
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
 
import prediction  # our inference module (prediction.py)
 
app = Flask(__name__)
CORS(app)  # allow the frontend (served separately) to call this API
 
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB upload limit
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
 
 
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
 
@app.route("/")
def home():
    return render_template("INDEX.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
 
 
@app.route("/api/predict", methods=["POST"])
def predict_route():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file part in the request"}), 400
 
    file = request.files["image"]
 
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
 
    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400
 
    try:
        image_bytes = file.read()
        result = prediction.predict(image_bytes)
        return jsonify(result), 200
 
    except FileNotFoundError as e:
        # Model file missing on disk
        return jsonify({"error": str(e)}), 503
 
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500
 
 
@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File too large. Max upload size is 10 MB."}), 413
 
 
if __name__ == "__main__":
    # Warm up: load the model once at startup instead of on first request
    try:
        prediction.load_resources()
        print("Model loaded successfully. Starting server...")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("Server will start, but /api/predict will fail until a model is available.")
 
    app.run(host="0.0.0.0", port=5000, debug=True)
