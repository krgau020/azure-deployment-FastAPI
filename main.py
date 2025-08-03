import pickle
import numpy as np
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

# Load the model and scaler
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

app = FastAPI(title="Boston House Price Prediction API")

# Define the expected request format using Pydantic
class HouseData(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    Age: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Boston House Price Prediction API"}

@app.post("/predict")
def predict(data: HouseData):
    # Convert input data to a NumPy array and scale it
    input_data = np.array([[data.CRIM, data.ZN, data.INDUS, data.CHAS, data.NOX,
                            data.RM, data.Age, data.DIS, data.RAD, data.TAX,
                            data.PTRATIO, data.B, data.LSTAT]])
    scaled_data = scaler.transform(input_data)
    
    # Predict using the regression model
    prediction = regmodel.predict(scaled_data)
    return {"predicted_price": prediction[0]}



if __name__ == "__main__":
   
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)