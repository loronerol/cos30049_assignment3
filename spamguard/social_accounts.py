"""API endpoints for classifying social accounts with the exported model."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter, Depends

from auth import current_user
from schemas import AccountFeatures

router = APIRouter(prefix="/api")
MODEL_PATH = Path(__file__).resolve().parent / "models" / "spamsentinel_model.joblib"
FEATURE_NAMES = list(AccountFeatures.model_fields)


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Trained social-account model not found: {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    if bundle["features"] != FEATURE_NAMES:
        raise RuntimeError(
            "The model feature list does not match the account-classification API. "
            "Export the reduced social-account model again."
        )
    return bundle


@router.get("/accounts/model")
def account_model(user: dict = Depends(current_user)):
    bundle = load_bundle()
    return {
        "model_name": bundle["model_name"],
        "feature_set": bundle["feature_set"],
        "features": bundle["feature_meta"],
        "review_band": bundle["review_band"],
    }


@router.post("/accounts/analyze")
def analyze_account(features: AccountFeatures, user: dict = Depends(current_user)):
    bundle = load_bundle()
    row = pd.DataFrame(
        [{name: getattr(features, name) for name in bundle["features"]}],
        columns=bundle["features"],
    )
    spam_score = float(bundle["model"].predict_proba(row)[0, 1])
    low, high = map(float, bundle["review_band"])
    label = "likely_spam" if spam_score > high else (
        "likely_genuine" if spam_score < low else "needs_review"
    )
    return {
        "model_name": bundle["model_name"],
        "spam_score": spam_score,
        "label": label,
        "needs_review": label == "needs_review",
        "review_band": {"low": low, "high": high},
    }
