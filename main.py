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
from typing import Optional, List
import pickle
import numpy as np
import json
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

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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

with open("models/fraud_type_classifier.pkl", "rb") as f:
    ft_model = pickle.load(f)

with open("models/iso_feature_cols.pkl", "rb") as f:
    ISO_FEATURE_COLS = pickle.load(f)

with open("models/optimal_threshold.pkl", "rb") as f:
    OPTIMAL_THRESHOLD = pickle.load(f)

iso_feature_idx = [FEATURE_COLS.index(c) for c in ISO_FEATURE_COLS]

print("✅ Models loaded.")

# ─────────────────────────────────────────────
# SETUP GITHUB MODELS CLIENT
# ─────────────────────────────────────────────

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_ENDPOINT = os.getenv("GITHUB_ENDPOINT", "https://models.inference.ai.azure.com")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "gpt-4o")

try:
    github_client = OpenAI(
        api_key=GITHUB_TOKEN or "placeholder",
        base_url=GITHUB_ENDPOINT,
    )
    _LLM_AVAILABLE = bool(GITHUB_TOKEN)
except Exception:
    github_client = None
    _LLM_AVAILABLE = False

# ─────────────────────────────────────────────
# PREDICTION MONITOR
# ─────────────────────────────────────────────

from collections import deque
from datetime import datetime, timezone

prediction_log = deque(maxlen=1000)


def log_prediction(risk_score: float, is_fraud: bool, fraud_type: str | None):
    prediction_log.append({
        "ts": datetime.now(timezone.utc).isoformat(),
        "risk_score": risk_score,
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
    })


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
    receiver_unique_senders_1h: int = 1
    sender_unique_receivers_1h: int = 1
    amount_trend_3trx: float = 1.0
    avg_time_between_trx_1h: float = 30.0


class LLMAnalysis(BaseModel):
    explanation: str
    risk_factors: List[str]
    investigation_steps: List[str]
    recommended_action: str
    confidence_note: str

class PredictionOutput(BaseModel):
    transaction_id: str
    risk_score: float                           # 0.0 – 1.0
    is_fraud: bool
    fraud_type_predicted: Optional[str]
    anomaly_flag: bool
    analysis: LLMAnalysis                       # structured LLM analysis
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
        "receiver_unique_senders_1h": trx.receiver_unique_senders_1h,
        "sender_unique_receivers_1h": trx.sender_unique_receivers_1h,
        "amount_trend_3trx":          trx.amount_trend_3trx,
        "avg_time_between_trx_1h":    trx.avg_time_between_trx_1h,
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
# HELPER: LLM ANALYSIS
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """Kamu adalah analis fraud senior di sistem GOTCHA (Guard & Observe Transactions with Cognitive Hybrid AI) untuk ekosistem keuangan digital Indonesia.

Tugasmu: menganalisis output dari model ML fraud detection dan memberikan penilaian independen. Model ML memberikan risk score dan sinyal, tapi KAMU yang harus:
1. Menilai apakah kombinasi sinyal benar-benar mencurigakan atau false positive
2. Menentukan aksi yang tepat berdasarkan konteks (bukan hanya risk score)
3. Memberikan langkah investigasi spesifik untuk analis

Konteks regulasi Indonesia:
- OJK mengatur bahwa transaksi mencurigakan > Rp100 juta wajib dilaporkan (LTKM)
- Rekening baru (<7 hari) yang menerima transfer besar = indikator rekening mule (PPATK)
- QRIS fraud substitusi biasanya terjadi di lokasi fisik dengan volume transaksi tinggi
- Pinjol ilegal sering menggunakan merchant game top-up atau tagihan sebagai kamuflase
- Social engineering di Indonesia dominan via WhatsApp/Telegram, target utama: lansia

Kamu HARUS mempertimbangkan:
- Apakah sinyal yang aktif secara individual lemah tapi secara kombinasi kuat?
- Apakah ada penjelasan legitimate untuk pola ini? (misal: transfer besar ke rekening baru bisa jadi pembelian properti)
- Tingkat keyakinan modelmu — apakah data cukup untuk memutuskan atau perlu investigasi tambahan?

Jawab dalam format JSON yang valid (tanpa markdown code block)."""


def analyze_transaction(trx: TransactionInput, risk_score: float,
                        fraud_type: Optional[str], anomaly_flag: bool,
                        signals: dict) -> LLMAnalysis:
    """LLM menganalisis transaksi secara independen dari model ML."""

    active_signals = [k for k, v in signals.items() if v]
    inactive_signals = [k for k, v in signals.items() if not v]

    prompt = f"""Analisis transaksi berikut. Model ML sudah memberikan prediksi, tapi kamu harus memberikan penilaian independen.

== DATA TRANSAKSI ==
Transaction ID: {trx.transaction_id}
Platform: {trx.platform_type}
Jenis transaksi: {trx.transaction_type}
Nominal: Rp {trx.amount_idr:,.0f}
Transfer antar bank: {trx.interbank_transfer}
Kategori merchant: {trx.merchant_category or 'N/A'}
Merchant di blacklist: {trx.is_merchant_blacklisted}

== PROFIL PENGIRIM ==
Umur akun: {trx.sender_account_age_days} hari
Ganti device baru-baru ini: {trx.device_changed_recently}
OS: {trx.sender_os}
Provinsi: {trx.sender_province}

== PROFIL PENERIMA ==
Tipe: {trx.receiver_type}
Umur akun penerima: {trx.receiver_account_age_days} hari
Penerima di blacklist: {trx.receiver_id_match_blacklist}

== POLA PERILAKU ==
Transaksi dalam 1 jam terakhir: {trx.trx_count_last_1h}
Transaksi dalam 24 jam terakhir: {trx.trx_count_last_24h}
Rasio nominal vs rata-rata pengirim: {trx.amount_vs_avg_ratio:.1f}x
Jam transaksi: {trx.hour_of_day}:00
Di luar jam aktif normal: {trx.is_outside_normal_hours}
Waktu sejak transaksi terakhir: {trx.time_since_last_trx_minutes:.1f} menit
Emulator terdeteksi: {trx.is_emulator}

== GRAPH & SEQUENCE FEATURES ==
Unique senders ke penerima (1 jam): {trx.receiver_unique_senders_1h}
Unique receivers dari pengirim (1 jam): {trx.sender_unique_receivers_1h}
Tren nominal (vs 3 transaksi terakhir): {trx.amount_trend_3trx:.1f}x
Rata-rata waktu antar transaksi (1 jam): {trx.avg_time_between_trx_1h:.1f} menit

== OUTPUT MODEL ML ==
Risk score: {risk_score:.2%}
Prediksi fraud: {'Ya' if risk_score >= 0.5 else 'Tidak'}
Jenis fraud: {fraud_type or 'N/A'}
Anomaly flag (Isolation Forest): {anomaly_flag}
Sinyal aktif: {', '.join(active_signals) if active_signals else 'tidak ada'}
Sinyal tidak aktif: {', '.join(inactive_signals)}

== INSTRUKSI ==
Berikan analisis dalam format JSON berikut:
{{
  "explanation": "Penjelasan 2-3 kalimat untuk analis keuangan. Hubungkan sinyal-sinyal yang ada menjadi narasi yang koheren. Jika ada penjelasan legitimate, sebutkan.",
  "risk_factors": ["Faktor risiko spesifik yang kamu identifikasi, masing-masing 1 kalimat singkat. Maksimal 4 faktor."],
  "investigation_steps": ["Langkah investigasi konkret dan spesifik untuk analis. Bukan generik seperti 'periksa transaksi' tapi spesifik seperti 'Cek apakah rekening penerima menerima transfer dari pengirim lain dalam 24 jam terakhir'. Maksimal 3 langkah."],
  "recommended_action": "BLOCK atau REVIEW atau ALLOW — berikan aksi berdasarkan analisismu, bukan hanya risk score. Risk score tinggi tapi semua sinyal punya penjelasan legitimate = REVIEW, bukan BLOCK.",
  "confidence_note": "1 kalimat tentang seberapa yakin kamu dengan analisis ini dan apa yang bisa mengubah kesimpulanmu."
}}"""

    try:
        if not _LLM_AVAILABLE or github_client is None:
            raise RuntimeError("LLM not configured")
        response = github_client.chat.completions.create(
            model=GITHUB_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        parsed = json.loads(raw)
        return LLMAnalysis(
            explanation=parsed["explanation"],
            risk_factors=parsed.get("risk_factors", [])[:4],
            investigation_steps=parsed.get("investigation_steps", [])[:3],
            recommended_action=parsed.get("recommended_action", "REVIEW"),
            confidence_note=parsed.get("confidence_note", ""),
        )
    except Exception:
        action = "BLOCK" if risk_score >= 0.80 else ("REVIEW" if risk_score >= 0.50 else "ALLOW")
        return LLMAnalysis(
            explanation=f"Transaksi memiliki risk score {risk_score:.0%}.",
            risk_factors=[s for s in signals if signals[s]][:4],
            investigation_steps=["Review manual diperlukan — analisis otomatis tidak tersedia."],
            recommended_action=action,
            confidence_note="Analisis LLM tidak tersedia, menggunakan fallback berbasis threshold.",
        )


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": "Random Forest (Calibrated) + Isolation Forest",
        "explainer": GITHUB_MODEL,
        "version": "2.0.0",
    }


@app.get("/monitor")
def model_monitor():
    """Prediction distribution for drift detection."""
    if not prediction_log:
        return {"total_predictions": 0, "message": "No predictions yet."}

    scores = [p["risk_score"] for p in prediction_log]
    fraud_count = sum(1 for p in prediction_log if p["is_fraud"])
    fraud_types = {}
    for p in prediction_log:
        ft = p["fraud_type"] or "normal"
        fraud_types[ft] = fraud_types.get(ft, 0) + 1

    return {
        "total_predictions": len(prediction_log),
        "fraud_rate": round(fraud_count / len(prediction_log), 4),
        "risk_score_distribution": {
            "mean": round(float(np.mean(scores)), 4),
            "std": round(float(np.std(scores)), 4),
            "p50": round(float(np.median(scores)), 4),
            "p90": round(float(np.percentile(scores, 90)), 4),
            "p99": round(float(np.percentile(scores, 99)), 4),
        },
        "fraud_type_distribution": fraud_types,
        "window": f"last {len(prediction_log)} predictions",
        "threshold": round(float(OPTIMAL_THRESHOLD), 4),
    }


@app.get("/stats")
def model_stats():
    return {
        "model_type":       "Random Forest (Calibrated + SMOTE)",
        "anomaly_layer":    "Isolation Forest",
        "fraud_type_model": "Random Forest (Multi-class)",
        "features":         len(FEATURE_COLS),
        "feature_types": {
            "behavioral":   ["trx_count_last_1h", "amount_vs_avg_ratio", "time_since_last_trx_minutes"],
            "graph":        ["receiver_unique_senders_1h", "sender_unique_receivers_1h"],
            "sequence":     ["amount_trend_3trx", "avg_time_between_trx_1h"],
        },
        "fraud_types": [
            "social_engineering",
            "rekening_mule",
            "qris_fraud_substitusi",
            "qris_fraud_merchant_fiktif",
            "pinjol_ilegal",
        ],
        "roc_auc":              0.9989,
        "calibration_error":    0.0157,
        "optimal_threshold":    round(float(OPTIMAL_THRESHOLD), 4),
        "dataset_rows":         1_000_000,
        "dataset_source":       "OJK-BI Calibrated Synthetic Dataset",
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(trx: TransactionInput):
    try:
        # Preprocessing
        X = preprocess(trx)

        # Random Forest — calibrated probability + cost-optimized threshold
        rf_prob      = rf_model.predict_proba(X)[0][1]
        rf_pred      = int(rf_prob >= OPTIMAL_THRESHOLD)

        # Fraud type — predicted by dedicated multi-class classifier
        fraud_type = None
        if rf_pred == 1:
            ft_pred_idx = ft_model.predict(X)[0]
            ft_encoder = encoders["fraud_type"]
            fraud_type = ft_encoder.inverse_transform([ft_pred_idx])[0]

        # Isolation Forest — anomaly flag (uses numeric behavioral features only)
        X_iso        = X[:, iso_feature_idx]
        iso_pred     = iso_model.predict(X_iso)[0]
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
            "high_fan_in":                   bool(trx.receiver_unique_senders_1h >= 5),
            "amount_escalation":             bool(trx.amount_trend_3trx >= 2.0),
        }

        # LLM analysis — independent reasoning over model outputs
        analysis = analyze_transaction(trx, rf_prob, fraud_type, anomaly_flag, signals)

        log_prediction(round(float(rf_prob), 4), bool(rf_pred), fraud_type)

        return PredictionOutput(
            transaction_id=       trx.transaction_id,
            risk_score=           round(float(rf_prob), 4),
            is_fraud=             bool(rf_pred),
            fraud_type_predicted= fraud_type,
            anomaly_flag=         bool(anomaly_flag),
            analysis=             analysis,
            signals=              signals,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))