# Wine Cultivar MLOps Pipeline

![CI](https://github.com/abdullahej5411/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

An automated MLOps pipeline for multi-class wine cultivar classification using
`sklearn.datasets.load_wine` (178 samples, 13 features, 3 classes).

## What is inside

- Modular code in `src/` for data loading, training and evaluation
- Two classifier families: RandomForest and GradientBoosting, 4 configurations each
- 5-fold stratified cross validation, seed 42 everywhere
- MLflow experiment tracking, model signatures, and model registry (`WineClassifier`, alias `champion`)
- Makefile commands for every task
- GitHub Actions CI with a model quality gate

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
make install
```

On Windows (Git Bash) use `python -m venv .venv` and `source .venv/Scripts/activate`.

## Commands

| Command | Purpose |
|---|---|
| `make install` | Install pinned dependencies |
| `make lint` | flake8, max line length 100 |
| `make test` | pytest, includes the model quality gate |
| `make train` | Run all experiments, log to MLflow, register the champion |
| `make evaluate` | Load the champion and score it on the test split |
| `make ui` | Open the MLflow UI at http://127.0.0.1:5000 |
| `make clean` | Remove bytecode and caches |

## Quality gate

`tests/test_model_gate.py` fails the build if the best model has validation macro F1 below 0.88,
batch inference slower than 30 ms, or predicts anything other than classes 0, 1, 2.
