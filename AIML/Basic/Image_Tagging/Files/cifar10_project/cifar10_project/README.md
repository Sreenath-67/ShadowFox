# 🔬 CIFAR-10 Vision Lab

A full-stack image classification web app powered by TensorFlow and Flask.
Multiple users can connect simultaneously over the same local network.

---

## 📁 Project Structure

```
cifar10_project/
├── train_model.py          ← Step 1: Train & save the CNN model
├── app.py                  ← Step 2: Run the Flask + Socket.IO server
├── requirements.txt
├── model/                  ← Auto-created after training
│   ├── cifar10_best.keras
│   └── metadata.json
├── templates/
│   └── index.html          ← Frontend HTML
└── static/
    ├── css/style.css
    └── js/app.js
```

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (one-time, ~10–30 min depending on hardware)
```bash
python train_model.py
```
This downloads CIFAR-10 automatically, trains a CNN, and saves:
- `model/cifar10_best.keras`
- `model/metadata.json`

### 3. Start the server
```bash
python app.py
```

### 4. Open in browser
- **You:**      http://localhost:5000
- **Others:**   http://<your-LAN-IP>:5000  (shown in terminal on startup)

Everyone on the same WiFi/network can open the URL and use it simultaneously.
All predictions appear in the **Live Feed** section for all connected users.

---

## 🧠 Model Architecture

| Layer      | Details                          |
|------------|----------------------------------|
| Input      | 32 × 32 × 3 RGB                  |
| Block 1    | Conv2D(32) × 2 + BN + MaxPool    |
| Block 2    | Conv2D(64) × 2 + BN + MaxPool    |
| Block 3    | Conv2D(128) × 2 + BN + MaxPool   |
| Head       | Dense(256) + Dropout → Softmax   |
| Regularization | Dropout 0.2 / 0.3 / 0.4 / 0.5 |
| Augmentation | Flip, rotate, shift, zoom      |

Expected test accuracy: **~82–87%** after 60 epochs.

---

## 🏷️ CIFAR-10 Classes

✈️ airplane · 🚗 automobile · 🐦 bird · 🐱 cat · 🦌 deer  
🐶 dog · 🐸 frog · 🐴 horse · 🚢 ship · 🚚 truck

---

## 🖥️ UI Features

- **Drag & drop** or click-to-browse image upload
- **Real-time prediction** with confidence percentage
- **Probability distribution bar chart** for all 10 classes
- **Live feed** — every user's predictions shown to all connected users
- **User counter** — live count of connected users
- **Model accuracy** displayed in header
- **Confetti animation** for high-confidence predictions (>85%)

---

## 🌐 Multi-User Setup

The server binds to `0.0.0.0:5000` so anyone on the same network can access it.
Socket.IO broadcasts each prediction to all connected clients in real time.

To make it accessible outside your LAN, use [ngrok](https://ngrok.com/):
```bash
ngrok http 5000
```

---

## 💡 Tips

- For faster training, enable GPU: install `tensorflow-gpu` and CUDA.
- Upload any image — the model resizes it to 32×32 internally.
- Best results with clear, single-subject photos matching CIFAR-10 classes.
