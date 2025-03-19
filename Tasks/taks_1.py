import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Task 1: Data Preprocessing
df = pd.read_csv('top_rated_movies(tmdb).csv')  # Replace with actual dataset

df.fillna(df.mean(), inplace=True)

# Convert categorical variables
df = pd.get_dummies(df, drop_first=True)

# Normalize dataset
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

# Save preprocessed data
df.to_csv('preprocessed_data.csv', index=False)

print("Task 1: Data Preprocessing completed.")
