from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import csv
import os

# Importing the wrapper registers the class so joblib can unpickle the model
# whether the winning estimator is the classical RandomForest or the PyTorch MLP
# (both expose the same predict_proba interface used below).
from mlp import TorchMLPClassifier  # noqa: F401

app = Flask(__name__)

# Load the saved model
model = joblib.load('employee_attrition_model.pkl')

# Exact 49-feature order the model expects (matches train.py).
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

# Categorical raw field -> one-hot column prefix.
CATEGORICAL_PREFIX = {
    "BusinessTravel": "",
    "Department": "",
    "EducationField": "Education_",
    "JobRole": "Role_",
    "MaritalStatus": "Status_",
}

# Baseline "typical employee" — dataset medians (numeric) and most-common
# categories. The form only collects the 8 key fields below; every other
# feature is held at this baseline.
BASELINE_RAW = {
    "Age": 36, "DailyRate": 802, "DistanceFromHome": 7, "Education": 3,
    "EnvironmentSatisfaction": 3, "HourlyRate": 66, "JobInvolvement": 3,
    "JobLevel": 2, "JobSatisfaction": 3, "MonthlyIncome": 4919, "MonthlyRate": 14236,
    "NumCompaniesWorked": 2, "PercentSalaryHike": 14, "PerformanceRating": 3,
    "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3, "WorkLifeBalance": 3, "YearsAtCompany": 5,
    "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
    "Gender": "Male", "OverTime": "No", "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development", "EducationField": "Life Sciences",
    "JobRole": "Sales Executive", "MaritalStatus": "Married",
}

# The 8 high-impact fields the UI exposes (everything else uses BASELINE_RAW).
KEY_FIELDS = [
    {"name": "OverTime", "label": "Works Overtime", "type": "toggle", "options": ["No", "Yes"]},
    {"name": "MonthlyIncome", "label": "Monthly Income", "type": "slider", "min": 1000, "max": 20000, "step": 100, "unit": "$"},
    {"name": "Age", "label": "Age", "type": "slider", "min": 18, "max": 60, "step": 1},
    {"name": "TotalWorkingYears", "label": "Total Working Years", "type": "slider", "min": 0, "max": 40, "step": 1},
    {"name": "YearsAtCompany", "label": "Years At Company", "type": "slider", "min": 0, "max": 40, "step": 1},
    {"name": "JobSatisfaction", "label": "Job Satisfaction", "type": "segment", "options": [1, 2, 3, 4], "labels": ["Low", "Medium", "High", "Very High"]},
    {"name": "StockOptionLevel", "label": "Stock Option Level", "type": "segment", "options": [0, 1, 2, 3], "labels": ["None", "Low", "Medium", "High"]},
    {"name": "JobRole", "label": "Job Role", "type": "select",
     "options": ["Healthcare Representative", "Human Resources", "Laboratory Technician",
                 "Manager", "Manufacturing Director", "Research Director",
                 "Research Scientist", "Sales Executive", "Sales Representative"]},
]

# Retention levers suggested when a field is pushing risk UP.
RECOMMENDATIONS = {
    "OverTime": "Rebalance workload to cut overtime.",
    "MonthlyIncome": "Benchmark pay against peers and review compensation.",
    "JobSatisfaction": "Hold a 1:1 to address role fit and engagement.",
    "StockOptionLevel": "Offer stock options or longer-term incentives.",
    "JobRole": "Map a clearer career path within the role.",
    "YearsAtCompany": "Re-engage with new projects or a rotation.",
}


def engineer_features(raw):
    """Turn a full raw-input dict into the ordered 49-feature vector."""
    feat = {col: 0.0 for col in EXPECTED_FEATURES}

    for name, value in raw.items():
        if name in CATEGORICAL_PREFIX:
            column = CATEGORICAL_PREFIX[name] + str(value)
            if column in feat:
                feat[column] = 1.0
        elif name == "Gender":
            feat["Gender"] = 1.0 if str(value) == "Male" else 0.0
        elif name == "OverTime":
            feat["OverTime"] = 1.0 if str(value) == "Yes" else 0.0
        elif name in feat:
            feat[name] = float(value)

    return [feat[col] for col in EXPECTED_FEATURES]


def risk_of(raw):
    """Predicted attrition probability for a full raw profile."""
    X = np.array([engineer_features(raw)], dtype=float)
    return float(model.predict_proba(X)[0][1])


def explain(merged):
    """Per-field contribution: how much each key field shifts risk vs. baseline."""
    current = risk_of(merged)
    factors = []
    for spec in KEY_FIELDS:
        name = spec["name"]
        if merged[name] == BASELINE_RAW[name]:
            continue  # at baseline -> no contribution to highlight
        counterfactual = dict(merged)
        counterfactual[name] = BASELINE_RAW[name]
        delta = current - risk_of(counterfactual)  # >0 means this field raises risk
        if abs(delta) < 0.005:
            continue
        increases = delta > 0
        factors.append({
            "field": name,
            "label": spec["label"],
            "value": merged[name],
            "impact": round(delta, 4),
            "direction": "up" if increases else "down",
            "recommendation": RECOMMENDATIONS.get(name) if increases else None,
        })
    factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
    return current, factors[:5]


def _predict_json(data):
    # Backward-compatible: a pre-built 49-length 'features' vector.
    if 'features' in data:
        X = np.array([list(data['features'])], dtype=float)
        prob = float(model.predict_proba(X)[0][1])
        return {'prediction': int(prob >= 0.5), 'probability': round(prob, 4)}

    # Otherwise: merge the submitted key fields over the baseline profile.
    merged = dict(BASELINE_RAW)
    for spec in KEY_FIELDS:
        if spec["name"] not in data:
            continue
        value = data[spec["name"]]
        if spec["type"] in ("slider", "segment"):
            value = float(value)  # arrives as a string from the form
        merged[spec["name"]] = value

    prob, factors = explain(merged)
    return {
        'prediction': int(prob >= 0.5),
        'probability': round(prob, 4),
        'factors': factors,
    }


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predictor')
def predictor():
    return render_template('predict.html', key_fields=KEY_FIELDS, baseline=BASELINE_RAW)


# --- Metrics page ---------------------------------------------------------
METRICS_PATH = "metrics.csv"
SELECTION_METRIC = "f1"  # the metric train.py uses to pick the served model
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall",
    "f1": "F1 Score", "roc_auc": "ROC AUC",
}


def load_metrics():
    """Read the model leaderboard written by train.py (metrics.csv)."""
    if not os.path.exists(METRICS_PATH):
        return None
    rows = []
    with open(METRICS_PATH, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"model": r["model"], **{c: float(r[c]) for c in METRIC_COLS}})
    if not rows:
        return None
    rows.sort(key=lambda x: x[SELECTION_METRIC], reverse=True)
    return {
        "rows": rows,
        "winner": rows[0]["model"],
        "served": type(model).__name__,
        "best": {c: max(r[c] for r in rows) for c in METRIC_COLS},
        "selection_metric": SELECTION_METRIC,
        "metric_cols": METRIC_COLS,
        "metric_labels": METRIC_LABELS,
    }


@app.route('/metrics')
def metrics():
    return render_template('metrics.html', data=load_metrics())


@app.route('/api/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])  # legacy endpoint kept working
def predict():
    return jsonify(_predict_json(request.get_json(force=True)))


if __name__ == '__main__':
    app.run(debug=True)
