import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from src.pipeline import build_pipeline
from src.utils import load_config, load_data, split_data, ensure_dir, save_model

def main(config_path: str):
    cfg = load_config(config_path)
    X, y = load_data(cfg["data"]["csv_path"], cfg["data"]["target"])
    X_train, X_test, y_train, y_test = split_data(X, y, cfg["data"]["test_size"], cfg["data"]["random_state"])

    numeric = cfg["features"]["numeric"]
    categorical = cfg["features"]["categorical"]
    model_type = cfg["model"]["type"]
    grid_map = cfg["model"]["params"]
    param_grid = grid_map[model_type]

    pipe = build_pipeline(numeric, categorical, model_type)

    cv_strategy = StratifiedKFold(n_splits=cfg["cv"]["n_splits"], shuffle=True, random_state=cfg["data"]["random_state"])
    scoring = cfg["cv"]["scoring"]
    n_jobs = cfg["cv"].get("n_jobs", -1)

    artifact_dir = cfg["mlflow"].get("artifact_dir", "artifacts")
    ensure_dir(artifact_dir)

    mlflow.autolog(log_datasets=False)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))

    with mlflow.start_run(run_name=f"{model_type}-gridsearch"):
        grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv_strategy, scoring=scoring, n_jobs=n_jobs, refit=True)
        grid.fit(X_train, y_train)

        # Log best params/score explicitly as well
        mlflow.log_metric("best_cv_score", grid.best_score_)
        for k, v in grid.best_params_.items():
            mlflow.log_param(k, v)

        best_model = grid.best_estimator_

        # Save model artifact for microservice
        model_path = os.path.join(artifact_dir, "model.joblib")
        save_model(best_model, model_path)
        mlflow.log_artifact(model_path)

        # Optional: register
        register_name = cfg["mlflow"].get("register_name", "").strip()
        if register_name:
            result = mlflow.sklearn.log_model(best_model, artifact_path="model", registered_model_name=register_name)
            # Transition stage requires Model Registry permissions; skipping auto-transition here.

    print(f"Training done. Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
