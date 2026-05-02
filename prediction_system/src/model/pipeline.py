"""
End-to-end training and inference pipeline.
Orchestrates dataset loading, feature engineering, preprocessing,
model training, SHAP setup, and threshold calibration.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from src.data.dataset import load_dataset
from src.features.lineage_engineering import engineer_lineage_features, get_all_feature_columns
from src.features.preprocessing import PreprocessingPipeline
from src.model.trainer import ModelTrainer
from src.explainability.shap_engine import SHAPEngine
from src.utils.threshold_calibration import compute_cost_optimal_threshold
from src.model.drift_detector import DriftDetector


MODEL_PATH = Path("models/xgboost_model.joblib")
PIPELINE_PATH = Path("models/preprocessing_pipeline.joblib")
SHAP_PATH = Path("models/shap_engine.joblib")
DRIFT_PATH = Path("models/drift_detector.joblib")
METADATA_PATH = Path("models/metadata.joblib")


def train_pipeline(
    n_samples: int = 3000,
    seed: int = 42,
    smote_strategy: str = "smote",
    cost_miss: float = 850,
    cost_fp: float = 45,
) -> dict:
    """
    Full training run. Returns dict with model, shap engine, metrics,
    threshold calibration, and drift detector.
    """
    print("Loading dataset...")
    df = load_dataset(n_samples=n_samples, seed=seed)

    print("Engineering lineage features...")
    df = engineer_lineage_features(df)
    feature_cols = get_all_feature_columns(df)

    labels = df["label"]
    print(f"  Dataset: {len(df)} units, {labels.sum()} failures ({labels.mean()*100:.2f}%)")
    print(f"  Features: {len(feature_cols)}")

    # Train / validation split (80/20 stratified)
    from sklearn.model_selection import train_test_split
    idx_train, idx_val = train_test_split(
        range(len(df)), test_size=0.2, random_state=seed, stratify=labels
    )
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val = df.iloc[idx_val].reset_index(drop=True)
    y_train = labels.iloc[idx_train].reset_index(drop=True)
    y_val = labels.iloc[idx_val].reset_index(drop=True)

    print("Preprocessing + SMOTE...")
    prep = PreprocessingPipeline(smote_strategy=smote_strategy, random_state=seed)
    X_train_res, y_train_res = prep.fit_transform(df_train, feature_cols, y_train)
    X_val = prep.transform(df_val, feature_cols)

    print(f"  After SMOTE: {X_train_res.shape[0]} training samples")

    print("Training XGBoost...")
    trainer = ModelTrainer()
    trainer.train(X_train_res, y_train_res, feature_cols)

    print("Cross-validating...")
    cv_results = trainer.cross_validate(X_train_res, y_train_res, n_splits=5)

    print("Calibrating decision threshold...")
    val_proba = trainer.predict_proba(X_val)
    threshold_result = compute_cost_optimal_threshold(
        y_val.values, val_proba, cost_miss=cost_miss, cost_fp=cost_fp
    )
    optimal_t = threshold_result["optimal_threshold"]
    trainer.threshold = optimal_t

    print(f"  Default threshold 0.50 -> optimal threshold {optimal_t:.3f}")
    print(f"  Cost reduction: {threshold_result['cost_reduction_pct']:.1f}%")

    eval_metrics = trainer.evaluate(X_val, y_val.values, threshold=optimal_t)
    print(f"  Val Precision={eval_metrics['precision']:.3f}  Recall={eval_metrics['recall']:.3f}  AUC={eval_metrics['roc_auc']:.3f}")

    print("Building SHAP explainer...")
    shap_engine = SHAPEngine(trainer.model, feature_cols)

    print("Setting up drift detector...")
    drift_detector = DriftDetector(df_train, feature_cols)

    # Save artefacts
    MODEL_PATH.parent.mkdir(exist_ok=True)
    trainer.save(str(MODEL_PATH))
    prep.save(str(PIPELINE_PATH))
    joblib.dump(shap_engine, str(SHAP_PATH))
    joblib.dump(drift_detector, str(DRIFT_PATH))

    metadata = {
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "failure_rate": float(labels.mean()),
        "optimal_threshold": optimal_t,
        "threshold_result": threshold_result,
        "eval_metrics": eval_metrics,
        "cv_results": cv_results,
        "cost_miss": cost_miss,
        "cost_fp": cost_fp,
        "df_train": df_train,
        "df_val": df_val,
        "y_val": y_val,
        "val_proba": val_proba,
        "X_val": X_val,
    }
    joblib.dump(metadata, str(METADATA_PATH))
    print("Training complete.")
    return metadata


def load_artifacts() -> dict:
    """Load all saved artefacts. Returns dict or None if not trained yet."""
    if not MODEL_PATH.exists():
        return None
    trainer = ModelTrainer.load(str(MODEL_PATH))
    prep = PreprocessingPipeline.load(str(PIPELINE_PATH))
    shap_engine = joblib.load(str(SHAP_PATH))
    drift_detector = joblib.load(str(DRIFT_PATH))
    metadata = joblib.load(str(METADATA_PATH))
    return {
        "trainer": trainer,
        "prep": prep,
        "shap_engine": shap_engine,
        "drift_detector": drift_detector,
        **metadata,
    }


def predict_unit(artifacts: dict, unit_data: dict) -> dict:
    """
    Run inference on a single unit given as a dict of feature values.
    Returns risk score, prediction, SHAP attribution, and NL explanation.
    """
    from src.explainability.translator import translate_shap_to_engineer

    feature_cols = artifacts["feature_cols"]
    prep: PreprocessingPipeline = artifacts["prep"]
    trainer: ModelTrainer = artifacts["trainer"]
    shap_engine: SHAPEngine = artifacts["shap_engine"]

    df_unit = pd.DataFrame([unit_data])
    df_unit = engineer_lineage_features(df_unit)

    X = prep.transform(df_unit, feature_cols)
    proba = float(trainer.predict_proba(X)[0])
    prediction = int(proba >= trainer.threshold)

    top_feats = shap_engine.top_features(X[0], n=10)
    waterfall_bytes = shap_engine.waterfall_figure(X[0])

    actual_values = {col: unit_data.get(col, None) for col in feature_cols}
    explanation = translate_shap_to_engineer(top_feats, actual_values, proba)

    return {
        "risk_score": proba,
        "prediction": prediction,
        "threshold": trainer.threshold,
        "top_features": top_feats,
        "waterfall_bytes": waterfall_bytes,
        "explanation": explanation,
    }


def predict_batch(artifacts: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Run inference on a batch dataframe. Adds risk_score and prediction columns."""
    feature_cols = artifacts["feature_cols"]
    prep: PreprocessingPipeline = artifacts["prep"]
    trainer: ModelTrainer = artifacts["trainer"]

    df = engineer_lineage_features(df)
    X = prep.transform(df, feature_cols)
    probas = trainer.predict_proba(X)
    preds = (probas >= trainer.threshold).astype(int)

    result = df.copy()
    result["risk_score"] = probas
    result["prediction"] = preds
    result["risk_level"] = pd.cut(
        probas, bins=[-0.01, 0.3, 0.6, 1.01],
        labels=["Low", "Medium", "High"]
    )
    return result
