"""
Train the Employee Attrition model — with MLflow experiment tracking.

Pipeline:
  1. Load data
  2. Quick EDA summary
  3. Feature engineering -> the exact 49-feature schema the Flask API expects
  4. Train/test split (stratified)
  5. Compare several models with cross-validation (classical + a PyTorch MLP)
  6. Hyperparameter-tune RandomForest via GridSearchCV; train the deep MLP
  7. Evaluate every candidate on the held-out test set
  8. Pick the winner by ROC AUC, report feature importances
  9. Persist the winning model -> employee_attrition_model.pkl
 10. Log params, metrics, leaderboard & the model to MLflow (nested runs per model)

Reconstructs the preprocessing implied by the original (lost) notebook, inferred
from the saved model's `feature_names_in_`:
  - Drop constant / id columns: EmployeeCount, EmployeeNumber, Over18, StandardHours
  - Target: Attrition (Yes/No -> 1/0)
  - Label-encode binary cols: Gender (Female=0, Male=1), OverTime (No=0, Yes=1)
  - One-hot encode (all categories kept): BusinessTravel, Department,
    EducationField (prefix 'Education'), JobRole (prefix 'Role'),
    MaritalStatus (prefix 'Status')

Run:
  python train.py                  # tune RF + train MLP, pick best, log to MLflow
  python train.py --no-tune        # skip GridSearchCV (faster)
  python train.py --no-balance     # don't class-balance
  python train.py --no-dl          # skip the deep-learning benchmark
  python train.py --no-mlflow      # disable MLflow logging

View the tracking UI afterwards:
  mlflow ui            # then open http://127.0.0.1:5000  (or --port 5001)
"""
import argparse
import contextlib
import os
import warnings

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

from mlp import TorchMLPClassifier

warnings.filterwarnings("ignore")

DATA_PATH = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_PATH = "employee_attrition_model.pkl"
METRICS_PATH = "metrics.csv"
EXPERIMENT = "employee-attrition"
RANDOM_STATE = 42
TEST_SIZE = 0.2
SELECTION_METRIC = "f1"  # which test metric decides the winning model

# Exact column order the served model expects.
EXPECTED_FEATURES = [
    "Age", "DailyRate", "DistanceFromHome", "Education", "EnvironmentSatisfaction",
    "Gender", "HourlyRate", "JobInvolvement", "JobLevel", "JobSatisfaction",
    "MonthlyIncome", "MonthlyRate", "NumCompaniesWorked", "OverTime",
    "PercentSalaryHike", "PerformanceRating", "RelationshipSatisfaction",
    "StockOptionLevel", "TotalWorkingYears", "TrainingTimesLastYear",
    "WorkLifeBalance", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager",
    "Non-Travel", "Travel_Frequently", "Travel_Rarely",
    "Human Resources", "Research & Development", "Sales",
    "Education_Human Resources", "Education_Life Sciences", "Education_Marketing",
    "Education_Medical", "Education_Other", "Education_Technical Degree",
    "Role_Healthcare Representative", "Role_Human Resources",
    "Role_Laboratory Technician", "Role_Manager", "Role_Manufacturing Director",
    "Role_Research Director", "Role_Research Scientist", "Role_Sales Executive",
    "Role_Sales Representative",
    "Status_Divorced", "Status_Married", "Status_Single",
]


def _safe_key(name):
    """Sanitize a model name into a valid MLflow metric key."""
    return name.replace("(", "").replace(")", "").replace(" ", "_").strip("_")


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def load_data(path):
    section("1. LOAD DATA")
    df = pd.read_csv(path)
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Missing values: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    return df


def explore(df):
    section("2. EDA SUMMARY")
    counts = df["Attrition"].value_counts()
    rate = (df["Attrition"] == "Yes").mean()
    print(f"Attrition distribution: No={counts.get('No', 0)}  Yes={counts.get('Yes', 0)}")
    print(f"Attrition rate: {rate:.1%}  (class imbalance: ~{(1 - rate) / rate:.1f}:1)")

    print("\nAttrition rate by selected categorical features:")
    for col in ["OverTime", "BusinessTravel", "MaritalStatus", "JobRole"]:
        grp = df.groupby(col)["Attrition"].apply(lambda s: (s == "Yes").mean()).sort_values(ascending=False)
        print(f"\n  {col}:")
        for name, val in grp.items():
            print(f"    {name:<28} {val:.1%}")
    return float(rate)


def build_features(df):
    section("3. FEATURE ENGINEERING")
    df = df.copy()

    y = (df["Attrition"] == "Yes").astype(int)
    df = df.drop(
        columns=["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    )

    df["Gender"] = (df["Gender"] == "Male").astype(int)
    df["OverTime"] = (df["OverTime"] == "Yes").astype(int)

    df = pd.get_dummies(df, columns=["BusinessTravel"], prefix="", prefix_sep="")
    df = pd.get_dummies(df, columns=["Department"], prefix="", prefix_sep="")
    df = pd.get_dummies(df, columns=["EducationField"], prefix="Education")
    df = pd.get_dummies(df, columns=["JobRole"], prefix="Role")
    df = pd.get_dummies(df, columns=["MaritalStatus"], prefix="Status")

    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected feature columns: {sorted(missing)}")
    X = df[EXPECTED_FEATURES].astype(float)
    print(f"Engineered feature matrix: {X.shape[1]} features (schema OK)")
    return X, y


def compare_models(X_train, y_train, balance):
    section("5. MODEL COMPARISON (5-fold CV, scoring=ROC AUC)")
    cw = "balanced" if balance else None
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight=cw),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight=cw
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "MLP (PyTorch)": TorchMLPClassifier(balance=balance, random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = {}
    for name, est in candidates.items():
        # The torch model doesn't survive process-based parallelism on Windows
        # (cross_val_score would silently return NaN), so run it single-process.
        n_jobs = 1 if isinstance(est, TorchMLPClassifier) else -1
        s = cross_val_score(est, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
        scores[name] = float(s.mean())
        print(f"  {name:<20} ROC AUC = {s.mean():.4f} (+/- {s.std():.4f})")
    return scores


def tune_random_forest(X_train, y_train, balance, do_tune):
    section("6. HYPERPARAMETER TUNING (RandomForest)")
    cw = "balanced" if balance else None
    if not do_tune:
        print("Skipped (--no-tune). Using sensible defaults.")
        model = RandomForestClassifier(
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight=cw
        )
        model.fit(X_train, y_train)
        params = {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"}
        return model, params, None

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
    }
    base = RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE, class_weight=cw)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(base, param_grid, scoring="roc_auc", cv=cv, n_jobs=-1, verbose=0)
    search.fit(X_train, y_train)
    print(f"Best CV ROC AUC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def train_mlp(X_train, y_train, balance):
    section("6b. DEEP LEARNING (PyTorch MLP)")
    cw = "balanced" if balance else None
    print(f"Training MLP [128, 64] with dropout + BatchNorm, class_weight={cw or 'none'} ...")
    model = TorchMLPClassifier(
        hidden=(128, 64), dropout=0.3, lr=1e-3, max_epochs=200,
        patience=15, balance=balance, random_state=RANDOM_STATE,
    )
    model.fit(X_train.values, y_train.values)
    params = {
        "hidden": "128-64", "dropout": 0.3, "lr": 1e-3,
        "max_epochs": 200, "patience": 15, "optimizer": "adam",
    }
    print("MLP training complete (early-stopped on validation loss).")
    return model, params


def evaluate(model, X_test, y_test):
    section("7. TEST-SET EVALUATION")
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    for k, v in metrics.items():
        print(f"{k.capitalize():<9}: {v:.4f}")

    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, target_names=["Stay", "Leave"])
    cm_text = "Confusion matrix [rows=actual, cols=predicted]\n" + str(cm)
    print("\n" + cm_text + "\n")
    print(report)
    return metrics, report, cm_text


def report_importances(model, top_n=15):
    section("8. TOP FEATURE IMPORTANCES")
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_.")
        return None
    imp = pd.Series(model.feature_importances_, index=EXPECTED_FEATURES).sort_values(ascending=False)
    for name, val in imp.head(top_n).items():
        bar = "#" * int(val / imp.max() * 30)
        print(f"  {name:<30} {val:.4f} {bar}")
    return imp


def main():
    parser = argparse.ArgumentParser(description="Train the employee attrition model.")
    parser.add_argument("--no-tune", action="store_true", help="skip GridSearchCV")
    parser.add_argument("--no-balance", action="store_true", help="don't use class_weight='balanced'")
    parser.add_argument("--no-dl", action="store_true", help="skip the PyTorch MLP benchmark")
    parser.add_argument("--no-mlflow", action="store_true", help="disable MLflow logging")
    args = parser.parse_args()
    balance = not args.no_balance
    use_mlflow = not args.no_mlflow

    df = load_data(DATA_PATH)
    attrition_rate = explore(df)
    X, y = build_features(df)

    section("4. TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
    print(f"class_weight = {'balanced' if balance else 'None'}")

    if use_mlflow:
        # MLflow 3.x requires a database backend; default to a local SQLite file.
        # Honour MLFLOW_TRACKING_URI if the user points at a server/other store.
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment(EXPERIMENT)
        run_ctx = mlflow.start_run(run_name="benchmark")
    else:
        run_ctx = contextlib.nullcontext()

    with run_ctx:
        cv_scores = compare_models(X_train, y_train, balance)

        # --- Train every candidate, evaluate on the held-out test set --------
        rf_model, rf_params, rf_cv = tune_random_forest(
            X_train, y_train, balance, do_tune=not args.no_tune
        )
        candidates = [("RandomForest", rf_model, {f"rf_{k}": v for k, v in rf_params.items()})]
        if not args.no_dl:
            mlp_model, mlp_params = train_mlp(X_train, y_train, balance)
            candidates.append(("MLP (PyTorch)", mlp_model, {f"mlp_{k}": v for k, v in mlp_params.items()}))

        results = []
        for name, mdl, mdl_params in candidates:
            print(f"\n>>> {name}")
            metrics, report, cm_text = evaluate(mdl, X_test, y_test)
            results.append({"model": name, "estimator": mdl, "params": mdl_params,
                            "metrics": metrics, "report": report, "cm": cm_text})
            if use_mlflow:
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.set_tags({"model": name, "class_weight": "balanced" if balance else "none"})
                    mlflow.log_params({**mdl_params, "n_features": X.shape[1],
                                       "n_train": X_train.shape[0], "n_test": X_test.shape[0]})
                    mlflow.log_metrics(metrics)
                    mlflow.log_text(report, "classification_report.txt")
                    mlflow.log_text(cm_text, "confusion_matrix.txt")

        # --- Pick the winner by F1 -----------------------------------------
        # F1 (not ROC AUC) is the selection metric: on this imbalanced data the
        # models tie on ROC AUC, but F1 rewards actually catching leavers
        # (recall), which is the business goal.
        section(f"MODEL LEADERBOARD (winner by {SELECTION_METRIC})")
        results.sort(key=lambda r: r["metrics"][SELECTION_METRIC], reverse=True)
        leaderboard = pd.DataFrame(
            [{"model": r["model"], **{k: round(v, 4) for k, v in r["metrics"].items()}} for r in results]
        )
        print(leaderboard.to_string(index=False))
        leaderboard.to_csv(METRICS_PATH, index=False)
        best = results[0]
        model = best["estimator"]
        print(f"\nWinner: {best['model']}  ({SELECTION_METRIC} = {best['metrics'][SELECTION_METRIC]:.4f})")

        imp = report_importances(model)

        section("9. SAVE MODEL")
        joblib.dump(model, MODEL_PATH)
        print(f"Saved winning model ({best['model']}) -> {MODEL_PATH}")
        print(f"n_features_in_ = {model.n_features_in_}  (API-compatible)")

        if use_mlflow:
            section("10. MLFLOW LOGGING (parent run)")
            mlflow.set_tags({
                "best_model": best["model"],
                "tuned": str(not args.no_tune),
                "class_weight": "balanced" if balance else "none",
                "dataset": os.path.basename(DATA_PATH),
            })
            mlflow.log_params({
                "best_model": best["model"],
                "class_weight": "balanced" if balance else "none",
                "tuned": not args.no_tune,
                "deep_learning": not args.no_dl,
                "n_features": X.shape[1],
                "n_train": X_train.shape[0],
                "n_test": X_test.shape[0],
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "attrition_rate": round(attrition_rate, 4),
                **best["params"],
            })
            mlflow.log_metrics(best["metrics"])
            # MLflow metric names allow only [alnum _ - . space /]; strip parens etc.
            # Skip NaN scores — MLflow 3.x rejects re-logging them during log_model.
            mlflow.log_metrics(
                {f"cv_roc_auc_{_safe_key(k)}": v
                 for k, v in cv_scores.items() if v == v}
            )
            if rf_cv is not None:
                mlflow.log_metric("cv_roc_auc_rf_best", rf_cv)

            mlflow.log_text(best["report"], "classification_report.txt")
            mlflow.log_text(best["cm"], "confusion_matrix.txt")
            mlflow.log_artifact(METRICS_PATH)  # the model leaderboard
            if imp is not None:
                mlflow.log_text(imp.to_csv(header=["importance"]), "feature_importances.csv")
            mlflow.log_artifact(MODEL_PATH)  # the joblib .pkl the Flask app loads

            # Log the model flavor when it's a pure-sklearn winner. The PyTorch
            # MLP is a custom estimator, so we ship it via the joblib artifact
            # above and record its signature instead of the sklearn flavor.
            try:
                signature = infer_signature(X_train, model.predict(X_train.values))
                if best["model"] == "RandomForest":
                    mlflow.sklearn.log_model(
                        model, name="model", signature=signature,
                        input_example=X_train.head(2),
                    )
            except Exception as exc:  # tracking must never fail the run
                print(f"(model-flavor logging skipped: {exc})")
            run = mlflow.active_run()
            print(f"Logged to experiment '{EXPERIMENT}' (run_id={run.info.run_id})")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            print("View with:  mlflow ui")


if __name__ == "__main__":
    main()
