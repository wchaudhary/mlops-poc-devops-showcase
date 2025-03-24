from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Load sample dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Model trained. MSE on test set: {mse:.2f}")

# Save the model to disk
os.makedirs("ml/artifacts", exist_ok=True)
joblib.dump(model, "ml/artifacts/linear_model.joblib")
print("Model saved to ml/artifacts/linear_model.joblib")
