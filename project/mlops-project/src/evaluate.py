import os
import argparse
import joblib
import mlflow
from sklearn.metrics import classification_report
from src.utils import (load_config, load_data, split_data, ensure_dir,
                       plot_and_save_roc, plot_and_save_pr, plot_and_save_confusion,
                       compute_metrics)

def main(config_path: str):
    cfg = load_config(config_path)
    X, y = load_data(cfg["data"]["csv_path"], cfg["data"]["target"])
    X_train, X_test, y_train, y_test = split_data(X, y, cfg["data"]["test_size"], cfg["data"]["random_state"])

    artifact_dir = cfg["mlflow"].get("artifact_dir", "artifacts")
    ensure_dir(artifact_dir)
    model_path = os.path.join(artifact_dir, "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")

    model = joblib.load(model_path)

    # Predictions & probabilities
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Fallback: decision_function -> sigmoid-like ranking
        if hasattr(model, "decision_function"):
            import numpy as np
            scores = model.decision_function(X_test)
            # Min-max to [0,1] for curves
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            raise ValueError("Model does not support probability outputs.")

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Artifacts
    roc_path = os.path.join(artifact_dir, "roc_curve.png")
    pr_path = os.path.join(artifact_dir, "pr_curve.png")
    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plot_and_save_roc(y_test, y_proba, roc_path)
    plot_and_save_pr(y_test, y_proba, pr_path)
    plot_and_save_confusion(y_test, y_pred, cm_path)

    # Save predictions CSV
    import pandas as pd
    pred_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred, "y_proba": y_proba})
    pred_csv = os.path.join(artifact_dir, "predictions.csv")
    pred_df.to_csv(pred_csv, index=False)

    # Log to MLflow
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))
    with mlflow.start_run(run_name="evaluation"):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_artifact(roc_path)
        mlflow.log_artifact(pr_path)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(pred_csv)

    print("Evaluation metrics:", metrics)
    print("Classification report:\n", classification_report(y_test, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
