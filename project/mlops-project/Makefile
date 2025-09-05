PY=python3
ENV?=.venv
EXP?=churn-exp

init:
	python3 -m venv $(ENV) && . $(ENV)/bin/activate && pip install -U pip -r requirements.txt

train:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.train --config configs/config.yaml

evaluate:
	MLFLOW_EXPERIMENT_NAME=$(EXP) $(PY) -m src.evaluate --config configs/config.yaml

test:
	pytest -q

lint:
	ruff check .

format:
	ruff check --fix . || true
