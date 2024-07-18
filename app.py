from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model, scaler, and label encoders
with open('linear_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()

    # Encode categorical features and handle unseen labels
    for col, le in label_encoders.items():
        try:
            data[col] = le.transform([data[col]])[0]
        except ValueError:
            # Handle unseen labels by using a default value or the most frequent category
            data[col] = le.transform([le.classes_[0]])[0]

    # Convert to numpy array and reshape
    feature_order = ['Levy', 'Manufacturer', 'Model', 'Prod. year', 'Category', 'Leather interior', 'Fuel type',
                     'Engine volume', 'Mileage', 'Cylinders', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel',
                     'Color', 'Airbags']

    features = np.array([data[feature] for feature in feature_order]).reshape(1, -1)

    # Standardize features
    features = scaler.transform(features)

    # Predict price
    prediction = model.predict(features)[0]

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
