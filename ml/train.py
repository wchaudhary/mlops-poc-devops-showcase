import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# load data
data = pd.read_csv("data.csv")

X = data[["branching", "version"]]
y = data["difficulty"]

# train model
model = LinearRegression()
model.fit(X, y)

# save model
joblib.dump(model, "model.pkl")

print("Model trained and saved as model.pkl")
