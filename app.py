# This is the Flask app that exposes an API endpoint to return predictions
# based on user input.
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['input']])
    return jsonify({'prediction': int(prediction[0])})

@app.route('/', methods=['GET'])
def predict():
    return "Hello World!"

if __name__ == '__main__':
    app.run(debug=True)
