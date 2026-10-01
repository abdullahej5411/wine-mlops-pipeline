import statistics
import time

import numpy as np
import pytest

from src.data import get_train_test_split
from src.train import CONFIGS, build_model, cross_validate_config

MIN_VAL_F1 = 0.88
MAX_LATENCY_MS = 30.0
VALID_CLASSES = {0, 1, 2}
LATENCY_REPEATS = 20


@pytest.fixture(scope="module")
def champion():
    X_train, X_test, y_train, _ = get_train_test_split()
    best_score = -1.0
    best_config = None
    for family, params in CONFIGS:
        score = cross_validate_config(family, params, X_train, y_train)["val_f1_macro"]
        if score > best_score:
            best_score = score
            best_config = (family, params)
    model = build_model(*best_config).fit(X_train, y_train)
    return {"model": model, "val_f1": best_score, "X_test": X_test}


def test_validation_f1_gate(champion):
    assert champion["val_f1"] >= MIN_VAL_F1


def test_inference_latency_gate(champion):
    model = champion["model"]
    batch = champion["X_test"]
    model.predict(batch)
    timings = []
    for _ in range(LATENCY_REPEATS):
        start = time.perf_counter()
        model.predict(batch)
        timings.append((time.perf_counter() - start) * 1000)
    assert statistics.median(timings) <= MAX_LATENCY_MS


def test_output_schema_gate(champion):
    predictions = champion["model"].predict(champion["X_test"])
    assert predictions.shape == (len(champion["X_test"]),)
    assert np.issubdtype(predictions.dtype, np.integer)
    assert set(np.unique(predictions)).issubset(VALID_CLASSES)
