# train_model.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data'
columns = [f'feature{i+1}' for i in range(60)] + ['Label']
data = pd.read_csv(url, header=None, names=columns)

# Use only first 10 features
X = data.iloc[:, :10].values
y = LabelEncoder().fit_transform(data['Label'])  # M=1, R=0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model
with open('rock_mine_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as rock_mine_model.pkl")
