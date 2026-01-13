from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, Any, List
import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

try:
    model = joblib.load(str(MODEL_PATH))
    load_error = None
except Exception as e:
    model = None
    load_error = str(e)


def infer_feature_names(m):
    if m is None:
        return None
    try:
        if hasattr(m, "feature_name_"):
            return list(m.feature_name_)
        if hasattr(m, "feature_names_"):
            return list(m.feature_names_)
        if hasattr(m, "feature_names_in_"):
            return list(m.feature_names_in_)
        if hasattr(m, "booster_"):
            try:
                return list(m.booster_.feature_name())
            except Exception:
                pass
        if hasattr(m, "get_booster"):
            try:
                return list(m.get_booster().feature_name())
            except Exception:
                pass
    except Exception:
        return None
    feat_file = BASE_DIR / "models" / "feature_names.txt"
    if feat_file.exists():
        return [x.strip() for x in feat_file.read_text().splitlines() if x.strip()]
    return None


expected_features = infer_feature_names(model)


class PredictRequest(BaseModel):
    features: Dict[str, Any]


app = FastAPI()


@app.get("/ping")
def ping():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    features = req.features or {}

    # Build DataFrame matching expected features when possible
    if expected_features is not None:
        X = pd.DataFrame(0, index=[0], columns=expected_features)
        for sym, val in features.items():
            if not val:
                continue
            candidates = [sym, sym.lower(), f"symptom_{sym}", f"symptom_{sym.lower()}"]
            for c in candidates:
                if c in X.columns:
                    X.at[0, c] = 1
            for col in X.columns:
                if sym.lower() in col.lower():
                    X.at[0, col] = 1
    else:
        X = pd.DataFrame([features])

    try:
        proba = model.predict_proba(X)[0]
        labels = list(getattr(model, "classes_", []))
        ranked = sorted(list(zip(labels, proba)), key=lambda x: x[1], reverse=True)[:5]
        return {"predictions": [{"disease": d, "probability": float(p)} for d, p in ranked]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))