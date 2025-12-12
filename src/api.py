from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

# Load model and knowledge dataset
model = joblib.load(r"C:\Users\HP\Documents\GitHub\personal_doc_consultant\notebooks\model.ipynb")
info = pd.read_csv(r"C:\Users\HP\Documents\GitHub\personal_doc_consultant\data\knowledge\perdoc2_specific_filled.csv").set_index('disease')

app = FastAPI()

class SymptomInput(BaseModel):
    features: dict  # {"fever":1, "cough":0, "fatigue":1}

@app.post("/predict")
def predict(inp: SymptomInput):
    X = pd.DataFrame([inp.features])
    proba = model.predict_proba(X)[0]
    labels = model.classes_
    ranked = sorted(zip(labels, proba), key=lambda x: x[1], reverse=True)[:3]
    results = [{
        "disease": d,
        "confidence": float(p),
        "info": info.loc[d, ['description','remedy','prevention']].to_dict()
    } for d, p in ranked]
    return {"top": results}