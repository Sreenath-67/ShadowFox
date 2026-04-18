"""
Flask + Socket.IO Backend for CIFAR-10 Image Classifier
Supports multiple simultaneous users on the same local network.

Run:  python app.py
Then open:  http://<your-ip>:5000  on any device on the same WiFi
"""

import os, io, json, base64, time, traceback
import numpy as np
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from PIL import Image
import tensorflow as tf

# ─────────────────────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "cifar10-secret-key-2024"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─────────────────────────────────────────────────────────────
#  Model loading
# ─────────────────────────────────────────────────────────────
MODEL_PATH = Path("model/cifar10_best.keras")
META_PATH  = Path("model/metadata.json")

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_EMOJI = {
    "airplane": "✈️", "automobile": "🚗", "bird": "🐦", "cat": "🐱",
    "deer": "🦌", "dog": "🐶", "frog": "🐸", "horse": "🐴",
    "ship": "🚢", "truck": "🚚"
}
CLASS_CATEGORY = {
    "airplane": "vehicle", "automobile": "vehicle", "ship": "vehicle", "truck": "vehicle",
    "bird": "animal", "cat": "animal", "deer": "animal",
    "dog": "animal", "frog": "animal", "horse": "animal",
}

model      = None
model_meta = {}

def load_model():
    global model, model_meta
    if not MODEL_PATH.exists():
        print("⚠️  No trained model found. Run  python train_model.py  first.")
        return False
    print("🔄 Loading model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    if META_PATH.exists():
        with open(META_PATH) as f:
            model_meta = json.load(f)
    print(f"✅ Model loaded  |  Test accuracy: {model_meta.get('test_accuracy', '?'):.2%}")
    return True


def preprocess_image(image_bytes):
    """Resize to 32×32, normalise to [0,1]."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)          # (1, 32, 32, 3)


# ─────────────────────────────────────────────────────────────
#  Shared activity feed (last 50 predictions across all users)
# ─────────────────────────────────────────────────────────────
activity_feed = []

def add_to_feed(entry):
    activity_feed.append(entry)
    if len(activity_feed) > 50:
        activity_feed.pop(0)

# ─────────────────────────────────────────────────────────────
#  HTTP routes
# ─────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           model_loaded=model is not None,
                           test_accuracy=model_meta.get("test_accuracy", 0))


@app.route("/api/status")
def status():
    return jsonify({
        "model_loaded":   model is not None,
        "test_accuracy":  model_meta.get("test_accuracy", 0),
        "epochs_trained": model_meta.get("epochs_trained", 0),
        "class_names":    CLASS_NAMES,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        img_bytes = file.read()
        img_array = preprocess_image(img_bytes)

        start   = time.time()
        preds   = model.predict(img_array, verbose=0)[0]
        elapsed = (time.time() - start) * 1000        # ms

        top_idx    = int(np.argmax(preds))
        top_label  = CLASS_NAMES[top_idx]
        confidence = float(preds[top_idx])

        # All class probabilities sorted descending
        all_preds = [
            {
                "label":      CLASS_NAMES[i],
                "emoji":      CLASS_EMOJI[CLASS_NAMES[i]],
                "confidence": float(preds[i]),
                "category":   CLASS_CATEGORY[CLASS_NAMES[i]],
            }
            for i in np.argsort(preds)[::-1]
        ]

        result = {
            "prediction":   top_label,
            "emoji":        CLASS_EMOJI[top_label],
            "category":     CLASS_CATEGORY[top_label],
            "confidence":   confidence,
            "inference_ms": round(elapsed, 1),
            "all_classes":  all_preds,
        }

        # Build a thumbnail for the activity feed
        thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()

        feed_entry = {
            "label":      top_label,
            "emoji":      CLASS_EMOJI[top_label],
            "confidence": round(confidence * 100, 1),
            "thumbnail":  thumb_b64,
            "ts":         time.strftime("%H:%M:%S"),
        }
        add_to_feed(feed_entry)

        # Broadcast to all connected clients
        socketio.emit("new_prediction", feed_entry)

        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/feed")
def get_feed():
    return jsonify({"feed": list(reversed(activity_feed))})


# ─────────────────────────────────────────────────────────────
#  Socket.IO events
# ─────────────────────────────────────────────────────────────
connected_users = {}

@socketio.on("connect")
def on_connect():
    sid = request.sid
    connected_users[sid] = {"connected_at": time.time()}
    emit("server_info", {
        "user_count":    len(connected_users),
        "model_loaded":  model is not None,
        "recent_feed":   list(reversed(activity_feed[-10:])),
    })
    socketio.emit("user_count", {"count": len(connected_users)})
    print(f"[+] Client {sid[:8]}  |  Total: {len(connected_users)}")


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    connected_users.pop(sid, None)
    socketio.emit("user_count", {"count": len(connected_users)})
    print(f"[-] Client {sid[:8]}  |  Total: {len(connected_users)}")


# ─────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()

    import socket
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    print("\n" + "=" * 60)
    print("  🌐 CIFAR-10 Classifier Server")
    print("=" * 60)
    print(f"  Local :  http://localhost:5000")
    print(f"  Network: http://{local_ip}:5000")
    print("  Share the Network URL with others on the same WiFi")
    print("=" * 60 + "\n")

    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
