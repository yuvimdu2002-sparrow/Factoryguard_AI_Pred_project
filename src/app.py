from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import json
import numpy as np
import time

app = Flask(__name__)

model = joblib.load("model/baseline_logistic_gridsearch.joblib")

# Web page
@app.route("/")
def index():
    return render_template("result.html")

# Receive JSON file 
@app.route("/predict_file", methods=["POST"])
def predict_file():

    start_time = time.perf_counter()

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    data = json.load(file)          # JSON file → dict
    df = pd.DataFrame([data])       # dict → DataFrame


    prob = model.predict_proba(df)[0][1]

    latency_ms = (time.perf_counter() - start_time) * 1000

    return jsonify({
        "failure_probability": float(prob),
        "latency_ms": round(latency_ms, 2),
        "status": "success"
    })


if __name__ == "__main__":
    app.run(debug=True)