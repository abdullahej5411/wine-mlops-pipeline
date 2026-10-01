import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.data import get_train_test_split
from src.train import CHAMPION_ALIAS, MODEL_NAME, TRACKING_URI


def load_champion():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{CHAMPION_ALIAS}")


def evaluate_champion():
    model = load_champion()
    _, X_test, _, y_test = get_train_test_split()
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    return {
        "test_f1_macro": f1_score(y_test, predictions, average="macro"),
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_log_loss": log_loss(y_test, probabilities),
    }


def main():
    metrics = evaluate_champion()
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
