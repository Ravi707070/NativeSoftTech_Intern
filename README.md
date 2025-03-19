# NativeSoftTech_MachineLearning_Intern

This repository contains three tasks as part of the NativeSoftTech Machine Learning Internship:

Task 1: Data Preprocessing

Load and clean a dataset.

Handle missing values and categorical variables.

Normalize or standardize the dataset.

Save the preprocessed dataset.

File: task_1.py

Task 2: Build a Machine Learning Model

Train a supervised learning model (Random Forest Classifier).

Split the dataset into training and testing sets.

Evaluate the model using accuracy, precision, recall, and F1-score.

Save the trained model.

File: task_2.py

Task 3: Deploy the Model

Load the trained model.

Create a Flask web API for predictions.

Deploy the application.

File: task_3.py

How to Run

Run Task 1: Preprocess the dataset and save it.

python task_1.py

Run Task 2: Train the machine learning model and save it.

python task_2.py

Run Task 3: Start the Flask API to make predictions.

python task_3.py

API Usage

Once the Flask API is running, you can send a POST request to http://127.0.0.1:5000/predict with the following JSON format:

{
  "features": [value1, value2, ..., valueN]
}

The API will return the predicted output:

{
  "prediction": predicted_value
}

Requirements

Install dependencies using:

pip install -r requirements.txt

Deployment

You can deploy the Flask API on platforms like Heroku, AWS, or Google Cloud for remote access.

