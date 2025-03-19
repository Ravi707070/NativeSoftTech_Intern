import pickle
import numpy as np
from flask import Flask, request, jsonify

# Task 3: Deploy Model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    print("Task 3: Deployment Started.")
    app.run(debug=True)
