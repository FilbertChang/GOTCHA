"""
GOTCHA-ID Backend API
=====================
FastAPI backend untuk fraud detection real-time.

Endpoints:
  POST /predict   → terima transaksi, return risk score + explanation
  GET  /health    → cek status server
  GET  /stats     → statistik model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ─────────────────────────────────────────────
# SETUP APLIKASI
# ─────────────────────────────────────────────

app = FastAPI(
    title="GOTCHA-ID Fraud Detection API",
    description="Guard & Observe Transactions with Cognitive Hybrid AI",
    version="1.0.0",
)

# CORS — izinkan frontend mengakses backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────

print("⏳ Loading models...")

with open("models/random_forest.pkl", "rb") as f:
    rf_model = pickle.load(f)

with open("models/isolation_forest.pkl", "rb") as f:
    iso_model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

with open("models/feature_columns.pkl", "rb") as f:
    FEATURE_COLS = pickle.load(f)

print("✅ Models loaded.")

# ─────────────────────────────────────────────
# SETUP GITHUB MODELS CLIENT
# ─────────────────────────────────────────────

github_client = OpenAI(
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_ENDPOINT"),
)

GITHUB_MODEL = os.getenv("GITHUB_MODEL", "gpt-4o")

# ─────────────────────────────────────────────
# SCHEMA INPUT / OUTPUT
# ─────────────────────────────────────────────

class TransactionInput(BaseModel):
    transaction_id: str
    platform_type: str                          # e_wallet / mobile_banking
    transaction_type: str                       # transfer / qris_payment / dll
    amount_idr: float
    interbank_transfer: bool
    merchant_category: Optional[str] = None
    is_merchant_blacklisted: bool = False
    sender_account_age_days: int
    device_changed_recently: bool = False
    sender_os: str                              # Android / iOS
    sender_province: str
    receiver_type: str                          # personal / merchant_qris / merchant_online
    receiver_account_age_days: int
    receiver_id_match_blacklist: bool = False
    trx_count_last_1h: int
    trx_count_last_24h: int
    amount_vs_avg_ratio: float
    hour_of_day: int
    is_outside_normal_hours: bool = False
    time_since_last_trx_minutes: float
    is_emulator: bool = False
    amount_roundness: float = 0.0


class PredictionOutput(BaseModel):
    transaction_id: str
    risk_score: float                           # 0.0 – 1.0
    is_fraud: bool
    fraud_type_predicted: Optional[str]
    anomaly_flag: bool
    explanation: str                            # dari GPT-4o
    recommended_action: str                     # BLOCK / REVIEW / ALLOW
    signals: dict                               # sinyal yang terdeteksi


# ─────────────────────────────────────────────
# HELPER: PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(trx: TransactionInput) -> np.ndarray:
    """Encode input transaksi menjadi array numerik untuk model."""

    data = {
        "platform_type":              trx.platform_type,
        "transaction_type":           trx.transaction_type,
        "amount_idr":                 trx.amount_idr,
        "interbank_transfer":         int(trx.interbank_transfer),
        "merchant_category":          trx.merchant_category or "none",
        "is_merchant_blacklisted":    int(trx.is_merchant_blacklisted),
        "sender_account_age_days":    trx.sender_account_age_days,
        "device_changed_recently":    int(trx.device_changed_recently),
        "sender_os":                  trx.sender_os,
        "sender_province":            trx.sender_province,
        "receiver_type":              trx.receiver_type,
        "receiver_account_age_days":  trx.receiver_account_age_days,
        "receiver_id_match_blacklist": int(trx.receiver_id_match_blacklist),
        "trx_count_last_1h":          trx.trx_count_last_1h,
        "trx_count_last_24h":         trx.trx_count_last_24h,
        "amount_vs_avg_ratio":        trx.amount_vs_avg_ratio,
        "hour_of_day":                trx.hour_of_day,
        "is_outside_normal_hours":    int(trx.is_outside_normal_hours),
        "time_since_last_trx_minutes": trx.time_since_last_trx_minutes,
        "is_emulator":                int(trx.is_emulator),
        "amount_roundness":           trx.amount_roundness,
    }

    # Encode categorical
    CATEGORICAL_COLS = [
        "platform_type", "transaction_type", "merchant_category",
        "sender_os", "sender_province", "receiver_type",
    ]
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        val = data[col]
        if val in le.classes_:
            data[col] = int(le.transform([val])[0])
        else:
            data[col] = 0  # unknown category → 0

    # Susun sesuai urutan FEATURE_COLS
    row = [data[col] for col in FEATURE_COLS]
    return np.array(row).reshape(1, -1)


# ─────────────────────────────────────────────
# HELPER: GENERATE EXPLANATION
# ─────────────────────────────────────────────

def generate_explanation(trx: TransactionInput, risk_score: float,
                          fraud_type: Optional[str], signals: dict) -> str:
    """Panggil GPT-4o via GitHub Models untuk generate penjelasan."""

    # Susun sinyal yang paling relevan
    top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:4]
    signals_text = ", ".join([f"{k}" for k, v in top_signals if v])

    prompt = f"""Kamu adalah sistem AI fraud detection untuk platform keuangan digital Indonesia bernama GOTCHA.

Analisis transaksi berikut dan berikan penjelasan singkat dalam Bahasa Indonesia (2-3 kalimat):

Data transaksi:
- Platform: {trx.platform_type}
- Jenis: {trx.transaction_type}
- Nominal: Rp {trx.amount_idr:,.0f}
- Jam transaksi: {trx.hour_of_day}:00
- Umur rekening penerima: {trx.receiver_account_age_days} hari
- Rasio nominal vs rata-rata: {trx.amount_vs_avg_ratio:.1f}x
- Transaksi dalam 1 jam terakhir: {trx.trx_count_last_1h}
- Transfer antar bank: {trx.interbank_transfer}

Risk score: {risk_score:.0%}
Prediksi jenis fraud: {fraud_type or 'tidak ada'}
Sinyal yang terdeteksi: {signals_text}

Berikan penjelasan yang jelas dan mudah dipahami oleh analis keuangan. Jangan gunakan jargon teknis berlebihan."""

    try:
        response = github_client.chat.completions.create(
            model=GITHUB_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Transaksi ini memiliki risk score {risk_score:.0%} dan terdeteksi sebagai mencurigakan berdasarkan pola anomali yang ditemukan."


# ─────────────────────────────────────────────
# HELPER: DETERMINE ACTION
# ─────────────────────────────────────────────

def determine_action(risk_score: float) -> str:
    if risk_score >= 0.80:
        return "BLOCK"
    elif risk_score >= 0.50:
        return "REVIEW"
    else:
        return "ALLOW"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "Random Forest + Isolation Forest",
        "explainer": GITHUB_MODEL,
        "version": "1.0.0",
    }


@app.get("/stats")
def model_stats():
    return {
        "model_type":       "Random Forest Classifier",
        "anomaly_layer":    "Isolation Forest",
        "features":         len(FEATURE_COLS),
        "fraud_types": [
            "social_engineering",
            "rekening_mule",
            "qris_fraud_substitusi",
            "qris_fraud_merchant_fiktif",
            "pinjol_ilegal",
        ],
        "roc_auc":          1.0,
        "dataset_rows":     1_000_000,
        "dataset_source":   "OJK-BI Calibrated Synthetic Dataset",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(trx: TransactionInput):
    try:
        # Preprocessing
        X = preprocess(trx)

        # Random Forest — risk score dan fraud type
        rf_prob      = rf_model.predict_proba(X)[0][1]   # probabilitas fraud
        rf_pred      = rf_model.predict(X)[0]

        # Fraud type — ambil dari class dengan probabilitas tertinggi
        # Model kita binary (fraud/normal), fraud_type kita infer dari sinyal
        fraud_type = None
        if rf_pred == 1:
            # Infer fraud type dari sinyal terkuat
            if trx.receiver_type == "merchant_qris":
                if trx.trx_count_last_1h > 20:
                    fraud_type = "qris_fraud_substitusi"
                else:
                    fraud_type = "qris_fraud_merchant_fiktif"
            elif trx.is_merchant_blacklisted and trx.merchant_category in ["game_topup", "tagihan"]:
                fraud_type = "pinjol_ilegal"
            elif trx.receiver_account_age_days <= 1 and trx.amount_vs_avg_ratio > 8 and trx.is_outside_normal_hours:
                fraud_type = "social_engineering"
            elif trx.receiver_account_age_days <= 1 and trx.device_changed_recently:
                fraud_type = "social_engineering"
            elif trx.receiver_id_match_blacklist and trx.receiver_account_age_days <= 3:
                fraud_type = "rekening_mule"
            else:
                fraud_type = "rekening_mule"

        # Isolation Forest — anomaly flag
        iso_pred     = iso_model.predict(X)[0]
        anomaly_flag = iso_pred == -1

        # Sinyal yang terdeteksi — convert semua ke Python native bool
        signals = {
            "receiver_account_age_days_low": bool(trx.receiver_account_age_days <= 7),
            "amount_ratio_high":             bool(trx.amount_vs_avg_ratio > 5),
            "burst_transactions":            bool(trx.trx_count_last_1h > 10),
            "outside_normal_hours":          bool(trx.is_outside_normal_hours),
            "interbank_transfer":            bool(trx.interbank_transfer),
            "merchant_blacklisted":          bool(trx.is_merchant_blacklisted),
            "receiver_blacklisted":          bool(trx.receiver_id_match_blacklist),
            "emulator_detected":             bool(trx.is_emulator),
            "device_changed":                bool(trx.device_changed_recently),
            "anomaly_detected":              bool(anomaly_flag),
        }

        # Generate explanation
        explanation = generate_explanation(trx, rf_prob, fraud_type, signals)

        return PredictionOutput(
            transaction_id=       trx.transaction_id,
            risk_score=           round(float(rf_prob), 4),
            is_fraud=             bool(rf_pred),
            fraud_type_predicted= fraud_type,
            anomaly_flag=         bool(anomaly_flag),
            explanation=          explanation,
            recommended_action=   determine_action(rf_prob),
            signals=              signals,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))