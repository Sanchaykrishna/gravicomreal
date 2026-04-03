from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# load model
model = pickle.load(open("gravicom_model.pkl", "rb"))

@app.route("/")
def home():
    return "GRAVICOM Backend Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["data"]
    data = np.array(data).reshape(1, -1)

    prediction = model.predict(data)[0]
    probability = max(model.predict_proba(data)[0])

    return jsonify({
        "prediction": str(prediction),
        "confidence": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)