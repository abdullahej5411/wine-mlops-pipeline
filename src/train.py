import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data import SEED, get_train_test_split

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
RUN_STAGE_TAG = "candidate"
CV_FOLDS = 5

SCORING = {
    "f1_macro": "f1_macro",
    "accuracy": "accuracy",
    "neg_log_loss": "neg_log_loss",
}

MODEL_CLASSES = {
    "RandomForest": RandomForestClassifier,
    "GradientBoosting": GradientBoostingClassifier,
}

CONFIGS = [
    ("RandomForest", {"n_estimators": 50, "max_depth": 3, "min_samples_split": 2}),
    ("RandomForest", {"n_estimators": 100, "max_depth": 5, "min_samples_split": 2}),
    ("RandomForest", {"n_estimators": 150, "max_depth": 8, "min_samples_split": 4}),
    ("RandomForest", {"n_estimators": 200, "max_depth": None, "min_samples_split": 2}),
    ("GradientBoosting", {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 2}),
    ("GradientBoosting", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
    ("GradientBoosting", {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3}),
    ("GradientBoosting", {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 2}),
]


def build_model(family, params):
    return MODEL_CLASSES[family](random_state=SEED, **params)


def cross_validate_config(family, params, X_train, y_train):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_validate(
        build_model(family, params),
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
    )
    return {
        "train_f1_macro": scores["train_f1_macro"].mean(),
        "train_accuracy": scores["train_accuracy"].mean(),
        "train_log_loss": -scores["train_neg_log_loss"].mean(),
        "val_f1_macro": scores["test_f1_macro"].mean(),
        "val_accuracy": scores["test_accuracy"].mean(),
        "val_log_loss": -scores["test_neg_log_loss"].mean(),
    }


def run_candidate(index, family, params, X_train, y_train):
    metrics = cross_validate_config(family, params, X_train, y_train)
    model = build_model(family, params).fit(X_train, y_train)
    signature = infer_signature(X_train, model.predict(X_train))
    with mlflow.start_run(run_name=f"{family}-config-{index}") as run:
        mlflow.log_params({"model_family": family, "cv_folds": CV_FOLDS, "seed": SEED, **params})
        mlflow.log_metrics(metrics)
        mlflow.set_tags({"model_family": family, "stage": RUN_STAGE_TAG})
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )
        return run.info.run_id, metrics


def register_champion(best_run_id):
    version = mlflow.register_model(f"runs:/{best_run_id}/model", MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, version.version)
    return version.version


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_train, X_test, y_train, y_test = get_train_test_split()

    results = []
    for index, (family, params) in enumerate(CONFIGS, start=1):
        run_id, metrics = run_candidate(index, family, params, X_train, y_train)
        results.append((run_id, family, params, metrics))
        print(f"{family} {params} val_f1_macro={metrics['val_f1_macro']:.4f}")

    best_run_id, family, params, metrics = max(
        results, key=lambda item: item[3]["val_f1_macro"]
    )
    version = register_champion(best_run_id)
    print(f"Champion: {family} {params}")
    print(f"Registered {MODEL_NAME} version {version} with alias {CHAMPION_ALIAS}")


if __name__ == "__main__":
    main()
