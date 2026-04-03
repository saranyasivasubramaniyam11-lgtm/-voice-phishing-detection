from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load trained model and vectorizer
with open("voice_phishing_model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)


class RequestData(BaseModel):
    text: str
    language: str


@app.post("/predict")
def predict(data: RequestData):

    text_lower = data.text.lower()

    X = vectorizer.transform([text_lower])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    result = "Phishing" if prediction == 1 else "Safe"

    return {
        "result": result,
        "confidence": round(float(probability), 2)
    }