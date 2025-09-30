import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

class ClientData(BaseModel):
    age: int
    income: float
    education: bool
    work: bool
    car: bool

app = FastAPI()
model = joblib.load("models/best_model.pkl")

@app.post("/score")
def score(data: ClientData, approve_rate: float = 0.43):
    features = [data.age, data.income, data.education, data.work, data.car]
    features.append(float(np.log1p(data.income)))

    prob_default = model.predict_proba([features])[0, 1]
    approved = bool(prob_default < approve_rate)
    
    return {"approved": approved}