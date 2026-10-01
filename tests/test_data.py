import pytest

from src.data import (
    EXPECTED_FEATURES,
    get_train_test_split,
    load_data,
    validate_data,
)


def test_split_sizes():
    X_train, X_test, y_train, y_test = get_train_test_split()
    assert len(X_train) == 142
    assert len(X_test) == 36
    assert len(y_train) == 142
    assert len(y_test) == 36


def test_split_is_stratified():
    X, y = load_data()
    _, _, y_train, y_test = get_train_test_split()
    for label in (0, 1, 2):
        full_share = (y == label).mean()
        assert abs((y_train == label).mean() - full_share) < 0.02
        assert abs((y_test == label).mean() - full_share) < 0.03


def test_split_is_reproducible():
    first = get_train_test_split()
    second = get_train_test_split()
    assert first[0].equals(second[0])
    assert first[1].equals(second[1])


def test_feature_count():
    X, _ = load_data()
    assert X.shape[1] == EXPECTED_FEATURES


def test_validate_passes_on_clean_data():
    X, y = load_data()
    assert validate_data(X, y) is True


def test_validate_rejects_null_values():
    X, y = load_data()
    X = X.copy()
    X.iloc[0, 0] = None
    with pytest.raises(ValueError):
        validate_data(X, y)


def test_validate_rejects_wrong_feature_count():
    X, y = load_data()
    with pytest.raises(ValueError):
        validate_data(X.iloc[:, :12], y)
