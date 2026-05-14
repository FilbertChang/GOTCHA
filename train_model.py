"""
GOTCHA-ID Model Training
========================
Melatih model fraud detection:
  1. Random Forest Classifier  — model utama (supervised, calibrated)
  2. Isolation Forest          — anomaly layer (unsupervised)
  3. Fraud Type Classifier     — multi-class (5 fraud types)

Techniques:
  - SMOTE oversampling for class imbalance
  - Isotonic calibration for reliable probability estimates
  - Cost-sensitive threshold optimization

Output:
  - models/random_forest.pkl     (calibrated)
  - models/isolation_forest.pkl
  - models/fraud_type_classifier.pkl
  - models/feature_columns.pkl
  - models/optimal_threshold.pkl
  - models/model_report.txt
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score,
    precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from imblearn.over_sampling import SMOTE

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

    # Graph features — network topology
    "receiver_unique_senders_1h",
    "sender_unique_receivers_1h",

    # Sequence features — temporal patterns
    "amount_trend_3trx",
    "avg_time_between_trx_1h",
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
# BAGIAN 2B: SMOTE OVERSAMPLING
# ─────────────────────────────────────────────

print("\n⏳ Applying SMOTE oversampling...")
print(f"  Before SMOTE: {y_train.sum():,} fraud / {len(y_train):,} total")

smote = SMOTE(sampling_strategy=0.1, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"  After SMOTE : {y_train_sm.sum():,} fraud / {len(y_train_sm):,} total")

# ─────────────────────────────────────────────
# BAGIAN 3: TRAINING RANDOM FOREST + CALIBRATION
# ─────────────────────────────────────────────

print("\n⏳ Training Random Forest Classifier (with SMOTE data)...")

base_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    verbose=0,
)

base_rf.fit(X_train_sm, y_train_sm)
print("  ✅ Base Random Forest selesai.")

print("\n⏳ Calibrating probabilities (isotonic regression)...")
rf_model = CalibratedClassifierCV(base_rf, method="isotonic", cv=3)
rf_model.fit(X_train, y_train)
print("  ✅ Calibrated model selesai.")

# Evaluasi
print("\n📊 Evaluasi Calibrated Random Forest:")
y_pred    = rf_model.predict(X_test)
y_prob    = rf_model.predict_proba(X_test)[:, 1]

roc_auc   = roc_auc_score(y_test, y_prob)
avg_prec  = average_precision_score(y_test, y_prob)

print(f"\n  ROC-AUC Score     : {roc_auc:.4f}")
print(f"  Avg Precision     : {avg_prec:.4f}")
print(f"\n  Classification Report (default threshold=0.5):")
print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))

print("  Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")

# Calibration quality check
frac_pos, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
cal_error = np.mean(np.abs(frac_pos - mean_pred))
print(f"\n  Mean Calibration Error: {cal_error:.4f}")

# ─────────────────────────────────────────────
# BAGIAN 3B: COST-SENSITIVE THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────

print("\n⏳ Optimizing decision threshold (cost-sensitive)...")

# Cost matrix: missing fraud is 10x worse than false alarm
# (blocking a legitimate Rp100k txn vs missing a Rp25M fraud)
COST_FN = 10.0   # cost of missing a fraud
COST_FP = 1.0    # cost of blocking legitimate

precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_prob)

best_threshold = 0.5
best_cost = float("inf")

for t in thresholds:
    y_t = (y_prob >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_t)
    tn, fp, fn, tp = cm_t.ravel()
    cost = (fn * COST_FN) + (fp * COST_FP)
    if cost < best_cost:
        best_cost = cost
        best_threshold = t

print(f"  Cost ratio (FN:FP) : {COST_FN:.0f}:{COST_FP:.0f}")
print(f"  Optimal threshold  : {best_threshold:.4f}")

y_opt = (y_prob >= best_threshold).astype(int)
cm_opt = confusion_matrix(y_test, y_opt)
print(f"\n  Classification Report (optimized threshold={best_threshold:.4f}):")
print(classification_report(y_test, y_opt, target_names=["Normal", "Fraud"]))
print(f"  TN={cm_opt[0,0]:,}  FP={cm_opt[0,1]:,}")
print(f"  FN={cm_opt[1,0]:,}  TP={cm_opt[1,1]:,}")

with open("models/optimal_threshold.pkl", "wb") as f:
    pickle.dump(best_threshold, f)

# Feature importance from base estimator
print("\n  Top 10 fitur terpenting:")
importances = base_rf.feature_importances_
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

# Isolation Forest works best on numeric behavioral features —
# label-encoded categoricals create spurious splits
ISO_FEATURE_COLS = [
    "amount_idr",
    "sender_account_age_days",
    "receiver_account_age_days",
    "trx_count_last_1h",
    "trx_count_last_24h",
    "amount_vs_avg_ratio",
    "hour_of_day",
    "time_since_last_trx_minutes",
    "amount_roundness",
    "receiver_unique_senders_1h",
    "sender_unique_receivers_1h",
    "amount_trend_3trx",
    "avg_time_between_trx_1h",
]

iso_feature_idx = [FEATURE_COLS.index(c) for c in ISO_FEATURE_COLS]

X_train_iso = X_train[:, iso_feature_idx]
X_test_iso  = X_test[:, iso_feature_idx]
X_train_normal_iso = X_train_iso[y_train == 0]

print(f"  Training pada {len(X_train_normal_iso):,} transaksi normal")
print(f"  Fitur Isolation Forest: {len(ISO_FEATURE_COLS)} (numeric behavioral only)")

iso_model = IsolationForest(
    n_estimators=300,
    max_samples=512,
    contamination=0.015,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)

iso_model.fit(X_train_normal_iso)
print("  ✅ Isolation Forest selesai ditraining.")

# Evaluasi — use decision_function scores for proper ROC-AUC
iso_scores = -iso_model.decision_function(X_test_iso)  # negate: higher = more anomalous
iso_pred_raw = iso_model.predict(X_test_iso)
iso_pred = (iso_pred_raw == -1).astype(int)

iso_roc = roc_auc_score(y_test, iso_scores)
print(f"\n📊 Evaluasi Isolation Forest:")
print(f"  ROC-AUC Score     : {iso_roc:.4f}")
print(classification_report(y_test, iso_pred, target_names=["Normal", "Fraud"]))

# Save feature list for backend
with open("models/iso_feature_cols.pkl", "wb") as f:
    pickle.dump(ISO_FEATURE_COLS, f)

# ─────────────────────────────────────────────
# BAGIAN 4B: TRAINING FRAUD TYPE CLASSIFIER
# ─────────────────────────────────────────────

print("\n⏳ Training Fraud Type Classifier (multi-class)...")

# Train only on fraud rows — predicts which type of fraud
df_fraud = df_encoded[df_encoded[TARGET_COL] == 1].copy()
fraud_type_raw = df[df["is_fraud"] == True]["fraud_type"]

ft_encoder = LabelEncoder()
ft_labels = ft_encoder.fit_transform(fraud_type_raw.values)
encoders["fraud_type"] = ft_encoder

X_ft = df_fraud[FEATURE_COLS].values
y_ft = ft_labels

X_ft_train, X_ft_test, y_ft_train, y_ft_test = train_test_split(
    X_ft, y_ft, test_size=0.2, random_state=42, stratify=y_ft
)

print(f"  Train : {len(X_ft_train):,} fraud rows")
print(f"  Test  : {len(X_ft_test):,} fraud rows")
print(f"  Classes: {list(ft_encoder.classes_)}")

ft_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

ft_model.fit(X_ft_train, y_ft_train)
print("  ✅ Fraud Type Classifier selesai ditraining.")

ft_pred = ft_model.predict(X_ft_test)
print(f"\n📊 Evaluasi Fraud Type Classifier:")
print(classification_report(y_ft_test, ft_pred, target_names=ft_encoder.classes_))

# ─────────────────────────────────────────────
# BAGIAN 5: SIMPAN MODEL
# ─────────────────────────────────────────────

print("\n⏳ Menyimpan model...")

os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/random_forest.pkl", compress=3)
joblib.dump(iso_model, "models/isolation_forest.pkl", compress=3)
joblib.dump(ft_model, "models/fraud_type_classifier.pkl", compress=3)

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
  incl. graph : receiver_unique_senders_1h, sender_unique_receivers_1h
  incl. seq   : amount_trend_3trx, avg_time_between_trx_1h

Random Forest (Calibrated, SMOTE)
---------------------------------
ROC-AUC              : {roc_auc:.4f}
Avg Precision        : {avg_prec:.4f}
Mean Calibration Err : {cal_error:.4f}
Optimal Threshold    : {best_threshold:.4f} (cost ratio FN:FP = {COST_FN:.0f}:{COST_FP:.0f})

Default threshold (0.5):
  TN={cm[0,0]}  FP={cm[0,1]}
  FN={cm[1,0]}  TP={cm[1,1]}

Optimized threshold ({best_threshold:.4f}):
  TN={cm_opt[0,0]}  FP={cm_opt[0,1]}
  FN={cm_opt[1,0]}  TP={cm_opt[1,1]}

Top 5 Features:
{chr(10).join([f"  {f}: {i:.4f}" for f, i in feat_imp[:5]])}

Isolation Forest
----------------
ROC-AUC       : {iso_roc:.4f}
Contamination : 0.015

Fraud Type Classifier
---------------------
Classes: {list(ft_encoder.classes_)}
{classification_report(y_ft_test, ft_pred, target_names=ft_encoder.classes_)}
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