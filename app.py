from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('rock_mine_model.pkl', 'rb'))

# In-memory prediction history
history = []

@app.route('/')
def home():
    return render_template('index.html', prediction_text="", history=history)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract 10 input features from form
        features = [float(request.form[f'feature{i}']) for i in range(1, 11)]
        input_data = np.array(features).reshape(1, -1)

        # Predict
        prediction = model.predict(input_data)[0]
        result = "Mine" if prediction == "M" else "Rock"

        # Add to history
        history.append((features, result))

        return render_template("index.html", prediction_text=f"🎯 The object is likely a: {result}", history=history)

    except Exception as e:
        return render_template("index.html", prediction_text=f"❌ Error: {str(e)}", history=history)

@app.route('/clear-history')
def clear_history():
    history.clear()
    return render_template("index.html", prediction_text="🗑️ Prediction history cleared.", history=history)

if __name__ == '__main__':
    app.run(debug=True)
