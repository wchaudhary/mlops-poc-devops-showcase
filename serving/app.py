from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("ml/artifacts/linear_model.joblib")

# Initialize API
app = FastAPI()

# Define input format
class InputData(BaseModel):
    features: list[float]  # expects an array of 8 numbers

@app.post("/predict")
def predict(input: InputData):
    data = np.array(input.features).reshape(1, -1)
    prediction = model.predict(data)
    return {"prediction": prediction[0]}
