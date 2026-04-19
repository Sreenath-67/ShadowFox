from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from model import train_model, predict_profit

app = Flask(__name__)
CORS(app)

# ✅ Load dataset
df = pd.read_csv("data/Sample - Superstore.csv", encoding="latin1")

# ✅ Train model automatically
model = train_model(df)

@app.route("/")
def home():
    return "Backend Running!"

# 🔹 Analysis API
@app.route("/analyze", methods=["GET"])
def analyze():
    return jsonify({
        "total_sales": float(df["Sales"].sum()),
        "total_profit": float(df["Profit"].sum()),
        "avg_sales": float(df["Sales"].mean())
    })

# 🔹 Charts API
@app.route("/charts", methods=["GET"])
def charts():
    category_data = df.groupby("Category")["Sales"].sum().to_dict()
    region_data = df.groupby("Region")["Profit"].sum().to_dict()

    return jsonify({
        "category_sales": category_data,
        "region_profit": region_data
    })

# 🔹 Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sales = float(data["sales"])
    result = predict_profit(model, {"sales": sales})
    return jsonify({"predicted_profit": result})

if __name__ == "__main__":
    app.run(debug=True)