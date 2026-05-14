"""
Tests for GOTCHA-ID API endpoints.
Requires models to be trained first (python train_model.py).
"""

import pytest
from fastapi.testclient import TestClient
from main import app


client = TestClient(app)

NORMAL_TXN = {
    "transaction_id": "TXN-TEST-001",
    "platform_type": "e_wallet",
    "transaction_type": "qris_payment",
    "amount_idr": 45000,
    "interbank_transfer": False,
    "merchant_category": "F&B",
    "is_merchant_blacklisted": False,
    "sender_account_age_days": 720,
    "device_changed_recently": False,
    "sender_os": "Android",
    "sender_province": "Jawa Barat",
    "receiver_type": "merchant_qris",
    "receiver_account_age_days": 365,
    "receiver_id_match_blacklist": False,
    "trx_count_last_1h": 1,
    "trx_count_last_24h": 4,
    "amount_vs_avg_ratio": 0.9,
    "hour_of_day": 12,
    "is_outside_normal_hours": False,
    "time_since_last_trx_minutes": 45.0,
    "is_emulator": False,
    "amount_roundness": 0.75,
    "receiver_unique_senders_1h": 1,
    "sender_unique_receivers_1h": 1,
    "amount_trend_3trx": 1.0,
    "avg_time_between_trx_1h": 35.0,
}

FRAUD_TXN = {
    "transaction_id": "TXN-TEST-002",
    "platform_type": "e_wallet",
    "transaction_type": "transfer",
    "amount_idr": 24500000,
    "interbank_transfer": True,
    "merchant_category": None,
    "is_merchant_blacklisted": False,
    "sender_account_age_days": 365,
    "device_changed_recently": True,
    "sender_os": "Android",
    "sender_province": "Jawa Barat",
    "receiver_type": "personal",
    "receiver_account_age_days": 1,
    "receiver_id_match_blacklist": False,
    "trx_count_last_1h": 2,
    "trx_count_last_24h": 3,
    "amount_vs_avg_ratio": 12.5,
    "hour_of_day": 22,
    "is_outside_normal_hours": True,
    "time_since_last_trx_minutes": 3.2,
    "is_emulator": False,
    "amount_roundness": 0.0,
    "receiver_unique_senders_1h": 4,
    "sender_unique_receivers_1h": 1,
    "amount_trend_3trx": 3.5,
    "avg_time_between_trx_1h": 5.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_stats():
    r = client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["features"] == 25
    assert "graph" in data["feature_types"]
    assert "sequence" in data["feature_types"]


def test_predict_normal():
    r = client.post("/predict", json=NORMAL_TXN)
    assert r.status_code == 200
    data = r.json()
    assert data["transaction_id"] == "TXN-TEST-001"
    assert 0.0 <= data["risk_score"] <= 1.0
    assert data["risk_score"] < 0.5
    assert data["is_fraud"] is False
    assert "analysis" in data
    assert "signals" in data


def test_predict_fraud():
    r = client.post("/predict", json=FRAUD_TXN)
    assert r.status_code == 200
    data = r.json()
    assert data["transaction_id"] == "TXN-TEST-002"
    assert 0.0 <= data["risk_score"] <= 1.0
    signals = data["signals"]
    active = sum(1 for v in signals.values() if v is True)
    assert active >= 3, f"Suspicious transaction should trigger multiple signals, got {active}"
    assert data["anomaly_flag"] is True or data["is_fraud"] is True
    assert "analysis" in data


def test_predict_output_schema():
    r = client.post("/predict", json=NORMAL_TXN)
    data = r.json()
    assert "risk_score" in data
    assert "is_fraud" in data
    assert "fraud_type_predicted" in data
    assert "anomaly_flag" in data
    assert "analysis" in data
    analysis = data["analysis"]
    assert "explanation" in analysis
    assert "risk_factors" in analysis
    assert "investigation_steps" in analysis
    assert "recommended_action" in analysis
    assert analysis["recommended_action"] in ("BLOCK", "REVIEW", "ALLOW")
    assert "confidence_note" in analysis


def test_predict_unknown_category():
    txn = NORMAL_TXN.copy()
    txn["sender_province"] = "Unknown Province"
    txn["transaction_id"] = "TXN-TEST-003"
    r = client.post("/predict", json=txn)
    assert r.status_code == 200


def test_predict_missing_optional_fields():
    txn = {
        "transaction_id": "TXN-TEST-004",
        "platform_type": "e_wallet",
        "transaction_type": "transfer",
        "amount_idr": 100000,
        "interbank_transfer": False,
        "sender_account_age_days": 100,
        "sender_os": "Android",
        "sender_province": "DKI Jakarta",
        "receiver_type": "personal",
        "receiver_account_age_days": 200,
        "trx_count_last_1h": 1,
        "trx_count_last_24h": 5,
        "amount_vs_avg_ratio": 1.0,
        "hour_of_day": 14,
        "time_since_last_trx_minutes": 30.0,
    }
    r = client.post("/predict", json=txn)
    assert r.status_code == 200


def test_risk_score_calibration():
    """Risk scores from calibrated model should be between 0 and 1."""
    r = client.post("/predict", json=NORMAL_TXN)
    score = r.json()["risk_score"]
    assert 0.0 <= score <= 1.0

    r = client.post("/predict", json=FRAUD_TXN)
    score = r.json()["risk_score"]
    assert 0.0 <= score <= 1.0
