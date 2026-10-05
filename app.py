from flask import Flask, render_template, request, redirect, url_for
import pickle
import numpy as np
import os
import pandas as pd

app = Flask(__name__)

FEATURE_COUNT = 10
MODEL_PATH = "model.pkl"


def train_default_model():
    dataset_path = "sonar.all-data.csv"
    if not os.path.exists(dataset_path):
        return None

    data = pd.read_csv(dataset_path, header=None)
    X = data.iloc[:, :FEATURE_COUNT].values
    y = data.iloc[:, -1].values

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    return model


def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        if getattr(model, "n_features_in_", None) == FEATURE_COUNT:
            return model

    return train_default_model() or None


# ✅ Load model safely (check file exists)
model = load_model()

# Prediction history
history = []


@app.route("/")
def index():
    return render_template("index.html", prediction_text="", history=history)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form[f"feature{i}"]) for i in range(1, FEATURE_COUNT + 1)]
        final_features = np.array([features])

        if model is not None:
            prediction = model.predict(final_features)[0]
        else:
            prediction = "Mine" if features[-1] > 0.5 else "Rock"

        history.insert(0, (prediction, features[:5]))
        if len(history) > 5:
            history.pop()

        return render_template("index.html", prediction_text=f"Predicted: {prediction}", history=history)
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}", history=history)


@app.route("/clear-history")
def clear_history():
    history.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
