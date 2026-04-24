"""
GOTCHA-ID Fraud Detection Dataset Generator
============================================
Dataset sintetis transaksi keuangan digital Indonesia
Dikalibrasi dari data OJK IASC 2023 dan Bank Indonesia 2024

Struktur script:
  Bagian 1 - Setup & Konstanta
  Bagian 2 - Generate User Pool
  Bagian 3 - Generate Transaksi Normal
  Bagian 4 - Inject Fraud
  Bagian 5 - Gabungkan & Shuffle
  Bagian 6 - Export ke CSV
"""

import pandas as pd
import numpy as np
import uuid
import random
from datetime import datetime, timedelta
from tqdm import tqdm

# Seed untuk reproducibility — hasil generate selalu sama setiap dijalankan
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ─────────────────────────────────────────────
# BAGIAN 1: SETUP & KONSTANTA
# ─────────────────────────────────────────────

# Jumlah total baris
TOTAL_ROWS       = 1_000_000
FRAUD_RATE       = 0.015          # 1.5% fraud — sumber: OJK IASC 2023
TOTAL_FRAUD      = int(TOTAL_ROWS * FRAUD_RATE)    # 15.000
TOTAL_NORMAL     = TOTAL_ROWS - TOTAL_FRAUD        # 985.000

# Jumlah user unik dalam simulasi
USER_POOL_SIZE   = 50_000

# Rentang waktu simulasi: 1 tahun penuh 2024
DATE_START       = datetime(2024, 1, 1)
DATE_END         = datetime(2024, 12, 31, 23, 59, 59)

# ── Platform ──────────────────────────────────
# Sumber: BI 2024 — e-wallet mendominasi volume transaksi digital
PLATFORM_DIST = {
    "e_wallet":       0.65,
    "mobile_banking": 0.35,
}

# ── Jenis Transaksi ───────────────────────────
# Sumber: BI QRIS 2024 — 6,24 miliar transaksi QRIS mendominasi
TRANSACTION_TYPE_DIST = {
    "qris_payment":        0.40,
    "transfer":            0.35,
    "top_up":              0.15,
    "tarik_tunai":         0.05,
    "pembayaran_tagihan":  0.05,
}

# ── Nominal (amount_idr) ──────────────────────
# Sumber: BI — rata-rata QRIS Rp105rb per transaksi 2024
# Distribusi log-normal: lebih realistis dari normal biasa
# (kebanyakan transaksi kecil, sedikit yang besar)
AMOUNT_PARAMS = {
    # (mean_log, std_log, min, max) — dalam Rupiah
    "qris_payment":        (11.5, 0.8,   5_000,    2_000_000),
    "transfer":            (13.5, 1.2,  10_000,   25_000_000),
    "top_up":              (11.9, 0.7,  50_000,    1_000_000),
    "tarik_tunai":         (13.1, 0.9, 100_000,   10_000_000),
    "pembayaran_tagihan":  (12.5, 0.8,  10_000,    5_000_000),
}

# ── Provinsi ──────────────────────────────────
# Sumber: OJK IASC — sebaran laporan fraud per provinsi
PROVINCE_DIST = {
    "Jawa Barat":          0.18,
    "DKI Jakarta":         0.14,
    "Jawa Timur":          0.12,
    "Jawa Tengah":         0.10,
    "Banten":              0.06,
    "Sumatera Utara":      0.05,
    "Sumatera Selatan":    0.04,
    "Sulawesi Selatan":    0.04,
    "Bali":                0.03,
    "Kalimantan Timur":    0.03,
    "Lainnya":             0.21,
}

# ── OS Pengirim ───────────────────────────────
# Sumber: Statcounter Indonesia 2024
SENDER_OS_DIST = {
    "Android": 0.90,
    "iOS":     0.10,
}

# ── IP Country ────────────────────────────────
# Mayoritas transaksi dari Indonesia; fraud sering dari luar negeri
IP_COUNTRY_DIST = {
    "Indonesia": 0.96,
    "Vietnam":   0.015,
    "Nigeria":   0.01,
    "Kamboja":   0.01,
    "Malaysia":  0.005,
}

# ── Merchant Category ─────────────────────────
MERCHANT_CATEGORIES = [
    "F&B", "retail", "transportasi", "tagihan",
    "game_topup", "kesehatan", "pendidikan", "fashion"
]

# ── Interbank Transfer ────────────────────────
# Probabilitas transfer ANTAR bank/platform (True = antar, False = sesama)
# Sumber: keputusan berdasarkan perilaku pengguna Indonesia
INTERBANK_PROB = {
    "e_wallet":       0.10,   # sesama 90%, antar 10%
    "mobile_banking": 0.25,   # sesama 75%, antar 25%
}

# ── Distribusi Jam (hour_of_day) ──────────────
# Fraud lebih terkonsentrasi di malam & dini hari
# Normal: mayoritas transaksi di jam sibuk 07.00–21.00
HOUR_WEIGHTS_NORMAL = [
    0.5, 0.3, 0.2, 0.2, 0.3, 0.5,   # 00–05 (dini hari, rendah)
    1.0, 2.5, 3.5, 4.0, 4.5, 4.5,   # 06–11 (pagi, naik)
    4.0, 4.0, 4.5, 4.5, 4.0, 4.0,   # 12–17 (siang, stabil)
    4.5, 4.5, 3.5, 2.5, 1.5, 0.8,   # 18–23 (malam, turun)
]

# ── Fraud Type Distribution ───────────────────
# Dari total 15.000 transaksi fraud
# Sumber: OJK IASC 2023 — proporsi kasus per jenis
FRAUD_TYPE_DIST = {
    "social_engineering":      0.40,
    "rekening_mule":           0.25,
    "qris_fraud_substitusi":   0.12,
    "qris_fraud_merchant_fiktif": 0.08,
    "pinjol_ilegal":           0.15,
}

# ── Rata-rata Kerugian per Fraud Type ─────────
# Sumber: OJK IASC 2023
FRAUD_AMOUNT_AVG = {
    "social_engineering":         38_300_000,   # Rp38,3 juta rata-rata
    "rekening_mule":              24_000_000,   # estimasi
    "qris_fraud_substitusi":       1_500_000,   # akumulasi banyak korban kecil
    "qris_fraud_merchant_fiktif": 15_000_000,   # nominal lebih besar
    "pinjol_ilegal":               8_500_000,   # Rp8,5 juta rata-rata OJK
}

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def weighted_choice(distribution: dict) -> str:
    """Pilih key dari dict berdasarkan bobot probabilitasnya."""
    keys   = list(distribution.keys())
    weights = list(distribution.values())
    return np.random.choice(keys, p=np.array(weights) / sum(weights))


def generate_amount(transaction_type: str) -> float:
    """
    Generate nominal transaksi dengan distribusi log-normal.
    Log-normal dipilih karena lebih realistis:
    kebanyakan transaksi bernilai kecil, sedikit yang bernilai besar.
    """
    mean_log, std_log, min_val, max_val = AMOUNT_PARAMS[transaction_type]
    amount = np.random.lognormal(mean=mean_log, sigma=std_log)
    amount = np.clip(amount, min_val, max_val)
    # Bulatkan ke ratusan Rupiah terdekat (lebih realistis)
    return round(amount / 100) * 100


def generate_timestamp() -> datetime:
    """Generate timestamp acak dalam rentang simulasi 2024."""
    delta = DATE_END - DATE_START
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return DATE_START + timedelta(seconds=random_seconds)


def generate_id(prefix: str) -> str:
    """Generate ID unik dengan prefix, contoh: TXN-A3F2B1C9"""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def compute_amount_roundness(amount: float) -> float:
    """
    Hitung seberapa 'bulat' sebuah nominal (0.0 = sangat ganjil, 1.0 = sangat bulat).
    Fraud sering pakai nominal ganjil untuk menghindari rule-based detection.
    """
    # Cek apakah habis dibagi denominasi umum
    denominators = [1000, 5000, 10000, 50000, 100000]
    for d in sorted(denominators, reverse=True):
        if amount % d == 0:
            return denominators.index(d) / (len(denominators) - 1)
    return 0.0


# ─────────────────────────────────────────────
# JALANKAN CEK AWAL
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# BAGIAN 2: GENERATE USER POOL
# ─────────────────────────────────────────────
# Kita buat 50.000 user unik dulu sebelum generate transaksi.
# Setiap user punya profil tetap: provinsi, OS, device, akun baru/lama.
# Ini penting supaya transaksi dari user yang sama terasa konsisten —
# misalnya satu user selalu dari Jawa Barat dan selalu pakai Android.

def generate_user_pool(n: int) -> dict:
    """
    Buat pool user dengan profil masing-masing.
    Return: dict dengan user_id sebagai key, profil sebagai value.
    """
    print(f"\n⏳ Generating {n:,} users...")

    users = {}
    province_keys   = list(PROVINCE_DIST.keys())
    province_weights = list(PROVINCE_DIST.values())
    os_keys         = list(SENDER_OS_DIST.keys())
    os_weights      = list(SENDER_OS_DIST.values())

    for _ in tqdm(range(n), desc="  User pool"):
        user_id = generate_id("USR")

        # Umur akun dalam hari — mayoritas akun sudah lama
        # Distribusi: sebagian besar 30–1825 hari (1 bulan – 5 tahun)
        account_age = int(np.random.exponential(scale=400))
        account_age = max(1, min(account_age, 1825))

        # Setiap user punya rata-rata transaksi historis sendiri
        # Ini dipakai nanti untuk hitung amount_vs_avg_ratio
        avg_amount = generate_amount(
            np.random.choice(
                list(TRANSACTION_TYPE_DIST.keys()),
                p=list(TRANSACTION_TYPE_DIST.values())
            )
        )

        # Jam aktif normal user: mayoritas aktif di siang hari
        # normal_active_start dan end dipakai untuk is_outside_normal_hours
        active_start = random.randint(6, 9)    # mulai aktif jam 6–9 pagi
        active_end   = random.randint(20, 23)  # berhenti aktif jam 8–11 malam

        users[user_id] = {
            "province":          np.random.choice(province_keys, p=province_weights),
            "os":                np.random.choice(os_keys, p=os_weights),
            "device_id":         generate_id("DEV"),
            "account_age_days":  account_age,
            "avg_trx_amount":    avg_amount,
            "active_start":      active_start,
            "active_end":        active_end,
            "platform":          weighted_choice(PLATFORM_DIST),
        }

    print(f"  ✅ {n:,} users berhasil dibuat.")
    return users


# ─────────────────────────────────────────────
# JALANKAN CEK AWAL
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# BAGIAN 3: GENERATE TRANSAKSI NORMAL
# ─────────────────────────────────────────────
# Generate 985.000 transaksi normal.
# Setiap transaksi merujuk ke user dari pool yang sudah dibuat.

def generate_normal_transactions(users: dict, n: int) -> list:
    """Generate n transaksi normal. Return list of dicts."""
    print(f"\n⏳ Generating {n:,} transaksi normal...")

    user_ids = list(users.keys())
    rows     = []

    for _ in tqdm(range(n), desc="  Normal txn"):
        sender_id  = random.choice(user_ids)
        receiver_id = random.choice(user_ids)

        # Sender dan receiver tidak boleh sama
        while receiver_id == sender_id:
            receiver_id = random.choice(user_ids)

        sender   = users[sender_id]
        receiver = users[receiver_id]

        platform         = sender["platform"]
        trx_type         = weighted_choice(TRANSACTION_TYPE_DIST)
        amount           = generate_amount(trx_type)
        ts               = generate_timestamp()
        hour             = ts.hour

        # Apakah di luar jam aktif normal user ini?
        outside_hours    = not (sender["active_start"] <= hour <= sender["active_end"])

        # Rasio nominal ini vs rata-rata historis sender
        avg              = sender["avg_trx_amount"] or 1
        amount_ratio     = round(amount / avg, 3)

        # Merchant category — hanya relevan untuk qris dan tagihan
        if trx_type in ["qris_payment", "pembayaran_tagihan"]:
            merchant_cat = random.choice(MERCHANT_CATEGORIES)
        else:
            merchant_cat = None

        # Receiver type
        if trx_type == "qris_payment":
            receiver_type = "merchant_qris"
        elif trx_type == "pembayaran_tagihan":
            receiver_type = "merchant_online"
        else:
            receiver_type = "personal"

        rows.append({
            # Grup A
            "transaction_id":             generate_id("TXN"),
            "timestamp":                  ts.strftime("%Y-%m-%d %H:%M:%S"),
            "platform_type":              platform,
            # Grup B
            "sender_id":                  sender_id,
            "sender_province":            sender["province"],
            "sender_account_age_days":    sender["account_age_days"],
            "sender_device_fingerprint":  sender["device_id"],
            "device_changed_recently":    False,
            "sender_os":                  sender["os"],
            # Grup C
            "receiver_id":                receiver_id,
            "receiver_type":              receiver_type,
            "receiver_account_age_days":  receiver["account_age_days"],
            "receiver_province":          receiver["province"],
            "receiver_id_match_blacklist": False,
            # Grup D
            "transaction_type":           trx_type,
            "amount_idr":                 amount,
            "merchant_category":          merchant_cat,
            "is_merchant_blacklisted":    False,
            "interbank_transfer":         np.random.random() < INTERBANK_PROB[platform],
            # Grup E
            "trx_count_last_1h":          random.randint(0, 3),
            "trx_count_last_24h":         random.randint(1, 15),
            "amount_vs_avg_ratio":        amount_ratio,
            "hour_of_day":                hour,
            "is_outside_normal_hours":    outside_hours,
            "time_since_last_trx_minutes": round(random.expovariate(1/60), 2),
            "is_emulator":                False,
            "ip_country":                 weighted_choice(IP_COUNTRY_DIST),
            "amount_roundness":           compute_amount_roundness(amount),
            # Grup F
            "is_fraud":                   False,
            "fraud_type":                 None,
            "fraud_amount_idr":           None,
        })

    print(f"  ✅ {n:,} transaksi normal berhasil dibuat.")
    return rows



# ─────────────────────────────────────────────
# BAGIAN 4: INJECT FRAUD
# ─────────────────────────────────────────────

def generate_fraud_transactions(users: dict, n: int) -> list:
    """Generate n transaksi fraud dengan logika per fraud type."""
    print(f"\n⏳ Generating {n:,} transaksi fraud...")

    user_ids = list(users.keys())
    rows     = []

    fraud_types = list(FRAUD_TYPE_DIST.keys())
    fraud_weights = list(FRAUD_TYPE_DIST.values())

    for _ in tqdm(range(n), desc="  Fraud txn"):
        fraud_type = np.random.choice(fraud_types, p=fraud_weights)

        sender_id   = random.choice(user_ids)
        sender      = users[sender_id]
        platform    = sender["platform"]

        # ── Defaults (akan di-override per fraud type) ──
        receiver_account_age = random.randint(1, 7)
        receiver_province    = sender["province"]
        receiver_type        = "personal"
        merchant_cat         = None
        hour                 = random.randint(20, 23)
        is_emulator          = False
        is_blacklisted       = False
        receiver_blacklisted = False
        interbank            = True
        trx_1h               = random.randint(1, 5)
        trx_24h              = random.randint(3, 20)
        device_changed       = False
        ip_country           = "Indonesia"
        amount_roundness_override = None

        # ── Social Engineering ──────────────────────────
        if fraud_type == "social_engineering":
            hour = random.randint(20, 23) if random.random() < 0.7 else random.randint(0, 1)
            receiver_account_age = random.randint(1, 24)   # jam, tapi kita pakai hari → < 1
            receiver_account_age = 1
            interbank            = True
            trx_1h               = random.randint(1, 3)
            device_changed       = random.random() < 0.3   # 30% ganti device
            avg                  = sender["avg_trx_amount"] or 1
            amount               = min(avg * random.uniform(5, 20), 25_000_000)
            amount               = round(amount / 100) * 100
            amount_roundness_override = 0.0                # nominal ganjil
            receiver_blacklisted = random.random() < 0.25  # 25% sudah dilaporkan

        # ── Rekening Mule ───────────────────────────────
        elif fraud_type == "rekening_mule":
            receiver_account_age = random.randint(1, 2)    # < 48 jam
            interbank            = True
            trx_1h               = random.randint(3, 10)   # pass-through burst
            trx_24h              = random.randint(10, 40)
            is_emulator          = random.random() < 0.4   # 40% pakai emulator
            receiver_blacklisted = random.random() < 0.5   # 50% sudah dilaporkan
            avg                  = sender["avg_trx_amount"] or 1
            amount               = min(avg * random.uniform(3, 15), 25_000_000)
            amount               = round(amount / 100) * 100
            amount_roundness_override = 0.0

        # ── QRIS Fraud Substitusi ───────────────────────
        elif fraud_type == "qris_fraud_substitusi":
            receiver_type        = "merchant_qris"
            merchant_cat         = random.choice(["F&B", "retail"])
            receiver_province    = sender["province"]      # harus provinsi sama
            trx_1h               = random.randint(30, 100) # burst banyak payer
            trx_24h              = random.randint(50, 200)
            interbank            = random.random() < 0.3
            receiver_account_age = random.randint(1, 30)
            hour                 = random.randint(9, 21)   # jam toko buka
            amount               = generate_amount("qris_payment")
            receiver_blacklisted = random.random() < 0.3

        # ── QRIS Fraud Merchant Fiktif ──────────────────
        elif fraud_type == "qris_fraud_merchant_fiktif":
            receiver_type        = "merchant_qris"
            merchant_cat         = random.choice(["game_topup", "retail", "fashion"])
            receiver_account_age = random.randint(1, 14)   # merchant baru
            is_blacklisted       = random.random() < 0.4
            receiver_blacklisted = random.random() < 0.4
            avg                  = sender["avg_trx_amount"] or 1
            amount               = min(avg * random.uniform(4, 20), 25_000_000)
            amount               = round(amount / 100) * 100
            trx_1h               = random.randint(1, 8)
            hour                 = random.randint(8, 22)

        # ── Pinjol Ilegal ───────────────────────────────
        elif fraud_type == "pinjol_ilegal":
            merchant_cat         = random.choice(["game_topup", "tagihan"])
            receiver_type        = "merchant_online"
            is_blacklisted       = True                    # selalu blacklisted
            receiver_blacklisted = True
            trx_1h               = random.randint(3, 12)   # top-up berulang
            trx_24h              = random.randint(10, 30)
            hour                 = random.randint(0, 4) if random.random() < 0.4 else random.randint(20, 23)
            amount               = random.choice([
                random.randint(50_000, 200_000),           # cicilan kecil
                random.randint(200_000, 500_000),
            ])
            amount               = round(amount / 100) * 100
            receiver_account_age = random.randint(30, 365) # platform sudah lama
            interbank            = random.random() < 0.6

        else:
            amount = generate_amount("transfer")

        # Timestamp dengan jam yang sudah ditentukan per fraud type
        ts = generate_timestamp()
        ts = ts.replace(hour=hour, minute=random.randint(0, 59))

        outside_hours = not (sender["active_start"] <= hour <= sender["active_end"])
        avg           = sender["avg_trx_amount"] or 1
        amount_ratio  = round(amount / avg, 3)
        roundness     = amount_roundness_override if amount_roundness_override is not None \
                        else compute_amount_roundness(amount)

        # Receiver fiktif — bukan dari user pool
        receiver_id = generate_id("RCV")

        fraud_avg   = FRAUD_AMOUNT_AVG[fraud_type]
        fraud_loss  = round(amount * random.uniform(0.8, 1.2) / 100) * 100
        fraud_loss  = min(fraud_loss, fraud_avg * 2)

        rows.append({
            # Grup A
            "transaction_id":             generate_id("TXN"),
            "timestamp":                  ts.strftime("%Y-%m-%d %H:%M:%S"),
            "platform_type":              platform,
            # Grup B
            "sender_id":                  sender_id,
            "sender_province":            sender["province"],
            "sender_account_age_days":    sender["account_age_days"],
            "sender_device_fingerprint":  sender["device_id"],
            "device_changed_recently":    device_changed,
            "sender_os":                  sender["os"],
            # Grup C
            "receiver_id":                receiver_id,
            "receiver_type":              receiver_type,
            "receiver_account_age_days":  receiver_account_age,
            "receiver_province":          receiver_province,
            "receiver_id_match_blacklist": receiver_blacklisted,
            # Grup D
            "transaction_type":           "transfer" if receiver_type == "personal" else "qris_payment",
            "amount_idr":                 amount,
            "merchant_category":          merchant_cat,
            "is_merchant_blacklisted":    is_blacklisted,
            "interbank_transfer":         interbank,
            # Grup E
            "trx_count_last_1h":          trx_1h,
            "trx_count_last_24h":         trx_24h,
            "amount_vs_avg_ratio":        amount_ratio,
            "hour_of_day":                hour,
            "is_outside_normal_hours":    outside_hours,
            "time_since_last_trx_minutes": round(random.expovariate(1/5), 2),
            "is_emulator":                is_emulator,
            "ip_country":                 ip_country,
            "amount_roundness":           roundness,
            # Grup F
            "is_fraud":                   True,
            "fraud_type":                 fraud_type,
            "fraud_amount_idr":           fraud_loss,
        })

    print(f"  ✅ {n:,} transaksi fraud berhasil dibuat.")
    return rows


    # ─────────────────────────────────────────────
# BAGIAN 5 & 6: GABUNGKAN, SHUFFLE, EXPORT CSV
# ─────────────────────────────────────────────

def build_and_export(normal_rows: list, fraud_rows: list, output_path: str):
    """Gabungkan normal + fraud, acak urutannya, export ke CSV."""

    print(f"\n⏳ Menggabungkan dan mengacak dataset...")
    all_rows = normal_rows + fraud_rows
    random.shuffle(all_rows)

    df = pd.DataFrame(all_rows)

    # Pastikan urutan kolom sesuai skema
    column_order = [
        "transaction_id", "timestamp", "platform_type",
        "sender_id", "sender_province", "sender_account_age_days",
        "sender_device_fingerprint", "device_changed_recently", "sender_os",
        "receiver_id", "receiver_type", "receiver_account_age_days",
        "receiver_province", "receiver_id_match_blacklist",
        "transaction_type", "amount_idr", "merchant_category",
        "is_merchant_blacklisted", "interbank_transfer",
        "trx_count_last_1h", "trx_count_last_24h", "amount_vs_avg_ratio",
        "hour_of_day", "is_outside_normal_hours", "time_since_last_trx_minutes",
        "is_emulator", "ip_country", "amount_roundness",
        "is_fraud", "fraud_type", "fraud_amount_idr",
    ]
    df = df[column_order]

    print(f"  Total baris    : {len(df):,}")
    print(f"  Total kolom    : {len(df.columns)}")
    print(f"  Fraud count    : {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    print(f"\n  Distribusi fraud_type:")
    print(df[df['is_fraud']]['fraud_type'].value_counts().to_string())

    print(f"\n⏳ Menyimpan ke {output_path}...")
    df.to_csv(output_path, index=False)
    size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"  ✅ Dataset berhasil disimpan!")
    print(f"  Estimasi ukuran : {size_mb:.1f} MB (in-memory)")



if __name__ == "__main__":
    print("=" * 55)
    print("  GOTCHA-ID Dataset Generator")
    print("=" * 55)
    print(f"  Total baris      : {TOTAL_ROWS:,}")
    print(f"  Total normal     : {TOTAL_NORMAL:,} ({(1-FRAUD_RATE)*100:.1f}%)")
    print(f"  Total fraud      : {TOTAL_FRAUD:,} ({FRAUD_RATE*100:.1f}%)")
    print(f"  User pool        : {USER_POOL_SIZE:,}")
    print(f"  Periode simulasi : 1 Jan 2024 – 31 Des 2024")
    print("=" * 55)

    # Bagian 2
    users = generate_user_pool(USER_POOL_SIZE)

    # Bagian 3 — generate FULL 985.000
    normal_rows = generate_normal_transactions(users, TOTAL_NORMAL)

    # Bagian 4 — generate FULL 15.000
    fraud_rows = generate_fraud_transactions(users, TOTAL_FRAUD)

    # Bagian 5 & 6 — gabungkan dan export
    build_and_export(normal_rows, fraud_rows, "gotcha_id_fraud_dataset.csv")

    print("\n" + "=" * 55)
    print("  SELESAI — gotcha_id_fraud_dataset.csv")
    print("=" * 55)