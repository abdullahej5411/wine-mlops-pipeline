from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2
EXPECTED_FEATURES = 13


def load_data():
    dataset = load_wine(as_frame=True)
    return dataset.data, dataset.target


def validate_data(X, y):
    if X.isnull().values.any() or y.isnull().values.any():
        raise ValueError("Dataset contains null values")
    if X.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_FEATURES} features but found {X.shape[1]}"
        )
    return True


def get_train_test_split():
    X, y = load_data()
    validate_data(X, y)
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
