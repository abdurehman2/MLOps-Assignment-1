import unittest
import json
from app import app 

class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        # Set up the test client for Flask
        self.app = app.test_client()
        self.app.testing = True

    def test_home(self):
        # Test if the home route returns status code 200
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"House Price Prediction", response.data)  # Check if the correct template is rendered

    def test_predict(self):
        # Define the input payload for prediction
        input_data = {
            'sqft_living': 2000,
            'bedrooms': 3,
            'bathrooms': 2,
            'floors': 1,
            'street': '123 Main St',
            'city': 'Seattle',
            'statezip': '98178',
            'country': 'USA'
        }

        # Make a POST request to the /predict endpoint
        response = self.app.post('/predict', 
                                 data=json.dumps(input_data), 
                                 content_type='application/json')

        # Check if the status code is 200 (OK)
        self.assertEqual(response.status_code, 200)

        # Parse the response data
        response_data = json.loads(response.data)

        # Check if the response contains the key 'prediction'
        self.assertIn('prediction', response_data)

        # Check if the prediction value is an integer (as the model returns)
        self.assertIsInstance(response_data['prediction'], int)


if __name__ == '__main__':
    unittest.main()
