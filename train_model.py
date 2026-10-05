# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

FEATURE_COUNT = 10

# 1. Load dataset (UCI Sonar dataset CSV)
data = pd.read_csv("sonar.all-data.csv", header=None)

# Use only the first 10 sonar features for the simplified app
X = data.iloc[:, :FEATURE_COUNT].values
y = data.iloc[:, -1].values

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# 5. Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("🎉 Model saved as model.pkl with 10 features")
