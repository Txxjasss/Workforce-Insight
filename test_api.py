"""
End-to-end tests for the Flask attrition API, driven through Flask's test
client (no network, no running server needed) so they run in CI.

Run:  pytest -q
"""
import json

import pytest

from app import EXPECTED_FEATURES, app, engineer_features


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# -- pages ------------------------------------------------------------------
def test_home_ok(client):
    assert client.get("/").status_code == 200


def test_predictor_page_ok(client):
    resp = client.get("/predictor")
    assert resp.status_code == 200


# -- feature engineering ----------------------------------------------------
def test_engineer_features_length_and_order():
    vec = engineer_features({"Age": 30, "OverTime": "Yes", "JobRole": "Manager"})
    assert len(vec) == len(EXPECTED_FEATURES) == 49
    # OverTime -> 1.0, the Manager one-hot is set, others default to 0.
    assert vec[EXPECTED_FEATURES.index("OverTime")] == 1.0
    assert vec[EXPECTED_FEATURES.index("Role_Manager")] == 1.0
    assert vec[EXPECTED_FEATURES.index("Age")] == 30.0


def test_engineer_features_unknown_category_ignored():
    vec = engineer_features({"JobRole": "Wizard"})  # not a real role
    assert sum(vec) == 0.0  # nothing matched, no crash


# -- prediction API ---------------------------------------------------------
def test_predict_key_fields(client):
    payload = {"OverTime": "Yes", "MonthlyIncome": 2500, "Age": 24,
               "JobSatisfaction": 1, "StockOptionLevel": 0}
    resp = client.post("/api/predict", data=json.dumps(payload),
                       content_type="application/json")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["factors"], list)


def test_predict_legacy_features_vector(client):
    vec = [0.0] * 49
    resp = client.post("/predict", data=json.dumps({"features": vec}),
                       content_type="application/json")
    assert resp.status_code == 200
    body = resp.get_json()
    assert set(body) >= {"prediction", "probability"}


def test_high_risk_profile_scores_higher_than_low_risk(client):
    """Sanity check the model: an overworked, low-paid junior should score
    higher attrition risk than a well-paid, satisfied senior."""
    high = {"OverTime": "Yes", "MonthlyIncome": 2000, "Age": 22,
            "JobSatisfaction": 1, "StockOptionLevel": 0, "TotalWorkingYears": 1}
    low = {"OverTime": "No", "MonthlyIncome": 18000, "Age": 45,
           "JobSatisfaction": 4, "StockOptionLevel": 3, "TotalWorkingYears": 25}

    def prob(p):
        r = client.post("/api/predict", data=json.dumps(p), content_type="application/json")
        return r.get_json()["probability"]

    assert prob(high) > prob(low)
