import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import pickle

# Use the full path to your data file
data_path = r'model_training\data.csv'

df = pd.read_csv(data_path)

# Drop 'date' column as instructed
df = df.drop('date', axis=1)

# Handle categorical data (e.g., 'street', 'city', 'statezip', 'country')
categorical_columns = ['street', 'city', 'statezip', 'country']

# Use one-hot encoding to convert categorical columns to numerical
df = pd.get_dummies(df, columns=categorical_columns)

# Features and labels
X = df.drop('price', axis=1)
y = df['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
print(f'Root Mean Squared Error: {rmse}')

# Save the model
with open('house_price_model.pkl', 'wb') as f:
    joblib.dump(model, f)

# Function to predict price based on input data
def predict_price(input_data):
    return model.predict([input_data])
