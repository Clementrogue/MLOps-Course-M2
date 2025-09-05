# MLOps Project — Churn Classifier with Pipelines + MLflow

This project trains a tabular classifier using scikit-learn `Pipeline` + `ColumnTransformer` and tracks everything with **MLflow**. It also prepares artifacts for a FastAPI microservice (`artifacts/model.joblib`).

## Quickstart

```bash
make init
# Put your CSV at data/raw.csv (see config for expected columns/target)
make train
make evaluate
make test
```

To run MLflow UI (if using local file store):
```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

## Config
Edit `configs/config.yaml` to match your dataset (paths, features, model, CV).

## Artifacts
- Trained model: `artifacts/model.joblib`
- Plots: `artifacts/roc_curve.png`, `artifacts/pr_curve.png`, `artifacts/confusion_matrix.png`
- Predictions: `artifacts/predictions.csv`
