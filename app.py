from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model_training/house_price_model.pkl')

# Load the training data to get the columns used for training
data_path = r'model_training\data.csv'
df = pd.read_csv(data_path)
df = df.drop('date', axis=1)
categorical_columns = ['street', 'city', 'statezip', 'country']
df = pd.get_dummies(df, columns=categorical_columns)
training_columns = df.drop('price', axis=1).columns

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data from the form
    input_data = {
        'sqft_living': float(request.json['sqft_living']),
        'bedrooms': float(request.json['bedrooms']),
        'bathrooms': float(request.json['bathrooms']),
        'floors': float(request.json['floors']),
        'street': request.json['street'],
        'city': request.json['city'],
        'statezip': request.json['statezip'],
        'country': request.json['country']
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # One-hot encode the input data
    input_df = pd.get_dummies(input_df)

    # Ensure the input data has the same columns as the training data
    input_df = input_df.reindex(columns=training_columns, fill_value=0)

    # Predict the price
    prediction = model.predict(input_df)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)