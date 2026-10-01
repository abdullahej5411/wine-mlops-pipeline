PYTHON ?= python

.PHONY: install lint test train evaluate ui reset-mlflow clean

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 --max-line-length=100 src tests

test:
	$(PYTHON) -m pytest -v tests

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate

ui:
	$(PYTHON) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

reset-mlflow:
	rm -rf mlruns mlflow.db

clean:
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
	find . -type f -name "*.pyc" -not -path "./.venv/*" -delete
	find . -type f -name "*.tmp" -not -path "./.venv/*" -delete
	rm -rf .pytest_cache
