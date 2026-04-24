"""
GOTCHA-ID Model Training
========================
Melatih dua model fraud detection:
  1. Random Forest Classifier  — model utama (supervised)
  2. Isolation Forest          — anomaly layer (unsupervised)

Output:
  - models/random_forest.pkl
  - models/isolation_forest.pkl
  - models/feature_columns.pkl
  - models/model_report.txt
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# BAGIAN 1: LOAD & PREPROCESSING
# ─────────────────────────────────────────────

print("=" * 55)
print("  GOTCHA-ID Model Training")
print("=" * 55)

# Load dataset
print("\n⏳ Loading dataset...")
df = pd.read_csv("gotcha_id_fraud_dataset.csv")
print(f"  ✅ {len(df):,} baris loaded, {len(df.columns)} kolom")
print(f"  Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")

# ── Feature Selection ─────────────────────────
# Kolom yang dipakai sebagai fitur model
# Kita exclude: ID columns, timestamp, dan label target
FEATURE_COLS = [
    # Platform & transaksi
    "platform_type",
    "transaction_type",
    "amount_idr",
    "interbank_transfer",
    "merchant_category",
    "is_merchant_blacklisted",

    # Sender
    "sender_account_age_days",
    "device_changed_recently",
    "sender_os",
    "sender_province",

    # Receiver
    "receiver_type",
    "receiver_account_age_days",
    "receiver_id_match_blacklist",

    # Sinyal perilaku
    "trx_count_last_1h",
    "trx_count_last_24h",
    "amount_vs_avg_ratio",
    "hour_of_day",
    "is_outside_normal_hours",
    "time_since_last_trx_minutes",
    "is_emulator",
    "amount_roundness",
]

TARGET_COL = "is_fraud"

print(f"\n  Fitur yang dipakai : {len(FEATURE_COLS)} kolom")

# ── Encode Categorical Columns ────────────────
# Random Forest butuh angka, bukan string
# Kita encode semua kolom kategorikal

print("\n⏳ Encoding categorical features...")

CATEGORICAL_COLS = [
    "platform_type", "transaction_type", "merchant_category",
    "sender_os", "sender_province", "receiver_type",
]

encoders = {}
df_encoded = df[FEATURE_COLS + [TARGET_COL]].copy()

# Isi nilai None/NaN di merchant_category dengan "none"
df_encoded["merchant_category"] = df_encoded["merchant_category"].fillna("none")

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    encoders[col] = le
    print(f"  {col:35s}: {len(le.classes_)} kategori")

# Convert boolean ke integer
BOOL_COLS = [
    "interbank_transfer", "device_changed_recently",
    "receiver_id_match_blacklist", "is_outside_normal_hours",
    "is_emulator", "is_merchant_blacklisted",
]
for col in BOOL_COLS:
    df_encoded[col] = df_encoded[col].astype(int)

print("  ✅ Encoding selesai.")

# ─────────────────────────────────────────────
# BAGIAN 2: TRAIN-TEST SPLIT
# ─────────────────────────────────────────────

print("\n⏳ Membagi dataset train/test (80/20)...")

X = df_encoded[FEATURE_COLS].values
y = df_encoded[TARGET_COL].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y       # pastikan proporsi fraud sama di train dan test
)

print(f"  Train : {len(X_train):,} baris ({y_train.sum():,} fraud)")
print(f"  Test  : {len(X_test):,} baris ({y_test.sum():,} fraud)")

# ─────────────────────────────────────────────
# BAGIAN 3: TRAINING RANDOM FOREST
# ─────────────────────────────────────────────

print("\n⏳ Training Random Forest Classifier...")
print("  (ini mungkin memakan waktu 1–3 menit)")

rf_model = RandomForestClassifier(
    n_estimators=100,      # 100 pohon keputusan
    max_depth=20,          # kedalaman maksimal tiap pohon
    min_samples_leaf=5,    # minimal 5 sampel per daun
    class_weight="balanced",  # handle imbalanced data (1.5% fraud)
    random_state=42,
    n_jobs=-1,             # pakai semua CPU core
    verbose=0,
)

rf_model.fit(X_train, y_train)
print("  ✅ Random Forest selesai ditraining.")

# Evaluasi
print("\n📊 Evaluasi Random Forest:")
y_pred    = rf_model.predict(X_test)
y_prob    = rf_model.predict_proba(X_test)[:, 1]

roc_auc   = roc_auc_score(y_test, y_prob)
avg_prec  = average_precision_score(y_test, y_prob)

print(f"\n  ROC-AUC Score     : {roc_auc:.4f}")
print(f"  Avg Precision     : {avg_prec:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# Feature importance
print("\n  Top 10 fitur terpenting:")
importances = rf_model.feature_importances_
feat_imp = sorted(
    zip(FEATURE_COLS, importances),
    key=lambda x: x[1], reverse=True
)
for feat, imp in feat_imp[:10]:
    bar = "█" * int(imp * 200)
    print(f"  {feat:35s}: {imp:.4f} {bar}")

# ─────────────────────────────────────────────
# BAGIAN 4: TRAINING ISOLATION FOREST
# ─────────────────────────────────────────────

print("\n⏳ Training Isolation Forest (anomaly layer)...")

# Isolation Forest hanya ditraining pada data NORMAL
X_train_normal = X_train[y_train == 0]
print(f"  Training pada {len(X_train_normal):,} transaksi normal saja")

iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.015,   # sesuai fraud rate kita (1.5%)
    random_state=42,
    n_jobs=-1,
)

iso_model.fit(X_train_normal)
print("  ✅ Isolation Forest selesai ditraining.")

# Evaluasi Isolation Forest
iso_pred_raw = iso_model.predict(X_test)
# Isolation Forest: -1 = anomali, 1 = normal
# Kita convert ke 0/1 agar konsisten
iso_pred = (iso_pred_raw == -1).astype(int)

iso_roc = roc_auc_score(y_test, iso_pred)
print(f"\n📊 Evaluasi Isolation Forest:")
print(f"  ROC-AUC Score     : {iso_roc:.4f}")
print(classification_report(y_test, iso_pred, target_names=["Normal", "Fraud"]))

# ─────────────────────────────────────────────
# BAGIAN 5: SIMPAN MODEL
# ─────────────────────────────────────────────

print("\n⏳ Menyimpan model...")

os.makedirs("models", exist_ok=True)

# Simpan Random Forest
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)

# Simpan Isolation Forest
with open("models/isolation_forest.pkl", "wb") as f:
    pickle.dump(iso_model, f)

# Simpan encoder dan feature columns
# Ini penting agar backend bisa encode input yang sama caranya
with open("models/encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(FEATURE_COLS, f)

# Simpan report ke file teks
report_text = f"""
GOTCHA-ID Model Training Report
================================
Dataset       : gotcha_id_fraud_dataset.csv
Total rows    : {len(df):,}
Fraud rate    : {df['is_fraud'].mean()*100:.2f}%
Features      : {len(FEATURE_COLS)}

Random Forest
-------------
ROC-AUC       : {roc_auc:.4f}
Avg Precision : {avg_prec:.4f}
TN={cm[0,0]}  FP={cm[0,1]}
FN={cm[1,0]}  TP={cm[1,1]}

Top 5 Features:
{chr(10).join([f"  {f}: {i:.4f}" for f, i in feat_imp[:5]])}

Isolation Forest
----------------
ROC-AUC       : {iso_roc:.4f}
Contamination : 0.015
"""

with open("models/model_report.txt", "w") as f:
    f.write(report_text)

print("  ✅ Semua model tersimpan di folder models/")
print("\n  File yang dihasilkan:")
for fname in os.listdir("models"):
    size = os.path.getsize(f"models/{fname}") / 1024
    print(f"  {fname:35s}: {size:.1f} KB")

print("\n" + "=" * 55)
print("  TRAINING SELESAI")
print("=" * 55)