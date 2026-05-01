# G.O.T.C.H.A
### Guard & Observe Transactions with Cognitive Hybrid AI

> Sistem deteksi fraud transaksi keuangan digital real-time berbasis AI, dikalibrasi khusus untuk pola kejahatan keuangan digital Indonesia.

---

## 🎯 Tentang Proyek

GOTCHA adalah platform intelijen fraud detection yang dirancang untuk ekosistem keuangan digital Indonesia. Sistem ini memadukan **Random Forest Classifier**, **Isolation Forest**, dan **GPT-4o** untuk mendeteksi dan menjelaskan transaksi mencurigakan secara real-time.

Dibangun untuk **Microsoft Elevate AI Impact Challenge** — tema Fraud Detection & Risk Management.

---

## 🛡️ Jenis Fraud yang Dideteksi

| Jenis Fraud | Deskripsi |
|---|---|
| **Social Engineering** | Penipuan via WhatsApp/Telegram yang memanipulasi korban |
| **Rekening Mule** | Rekening boneka untuk menampung dana hasil kejahatan |
| **QRIS Fraud Substitusi** | QR code palsu ditempel di atas QRIS merchant asli |
| **QRIS Merchant Fiktif** | Merchant QRIS palsu untuk menampung pembayaran |
| **Pinjol Ilegal** | Platform pinjaman online ilegal dalam daftar hitam OJK |

---

## 🏗️ Arsitektur

```
Frontend (React + Vite)
        ↓
Backend (FastAPI)
        ↓
┌───────────────────────────────┐
│  Random Forest Classifier     │  ← Model utama (supervised)
│  Isolation Forest             │  ← Anomaly layer (unsupervised)
│  GPT-4o (GitHub Models)       │  ← Explainability engine
└───────────────────────────────┘
        ↓
Azure Services
├── Azure OpenAI
└── Azure App Service (deployment)
```

---

## 📊 Dataset

**GOTCHA-ID Fraud Simulation Dataset** — dataset sintetis 1 juta baris yang dikalibrasi dari statistik resmi:

- **Bank Indonesia 2024** — distribusi transaksi QRIS, nominal rata-rata, platform
- **OJK IASC 2023** — distribusi jenis fraud, nominal kerugian, sebaran geografis

Dataset dibuat sendiri menggunakan `generate_dataset.py` dan tidak mengandung data pribadi apapun.

| Parameter | Nilai |
|---|---|
| Total baris | 1.000.000 |
| Fraud rate | 1.5% (15.000 transaksi) |
| Jumlah kolom | 31 |
| Periode simulasi | 1 Jan – 31 Des 2024 |

---

## 🤖 Model AI

| Model | Tipe | ROC-AUC | Fungsi |
|---|---|---|---|
| Random Forest | Supervised | 1.0000 | Scoring & klasifikasi fraud |
| Isolation Forest | Unsupervised | 0.5515 | Anomaly detection layer |
| GPT-4o | LLM | — | Explainability dalam Bahasa Indonesia |

---

## ⚙️ Tech Stack

**Backend:**
- Python 3.13
- FastAPI + Uvicorn
- scikit-learn (Random Forest, Isolation Forest)
- OpenAI SDK (GitHub Models / Azure OpenAI)

**Frontend:**
- React 19 + Vite
- Recharts (visualisasi)
- Lucide React (ikon)
- Axios (HTTP client)
- Premium Fintech UI (Revolut-inspired Design System)

**Azure Services:**
- Azure OpenAI (GPT-4o)
- Azure App Service (deployment)

---

## 🚀 Cara Menjalankan

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### 1. Clone repository
```bash
git clone https://github.com/FilbertChang/GOTCHA.git
cd GOTCHA
```

### 2. Setup backend
```bash
pip install fastapi uvicorn scikit-learn pandas numpy python-dotenv openai tqdm
```

Buat file `.env` di root folder:
```
GITHUB_TOKEN=your_github_pat_token
GITHUB_MODEL=gpt-4o
GITHUB_ENDPOINT=https://models.inference.ai.azure.com
```

### 3. Generate dataset & train model
```bash
python generate_dataset.py
python train_model.py
```

### 4. Jalankan backend
```bash
uvicorn main:app --reload
```

Backend berjalan di `http://localhost:8000`

### 5. Setup & jalankan frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend berjalan di `http://localhost:5173`

---

## 📡 API Endpoints

| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/health` | Cek status server |
| `GET` | `/stats` | Statistik model |
| `POST` | `/predict` | Analisis transaksi |

### Contoh request `/predict`:
```json
{
  "transaction_id": "TXN-001",
  "platform_type": "e_wallet",
  "transaction_type": "transfer",
  "amount_idr": 24500000,
  "interbank_transfer": true,
  "sender_account_age_days": 365,
  "receiver_account_age_days": 1,
  "amount_vs_avg_ratio": 12.5,
  "hour_of_day": 22,
  "is_outside_normal_hours": true,
  "...": "..."
}
```

### Contoh response:
```json
{
  "transaction_id": "TXN-001",
  "risk_score": 1.0,
  "is_fraud": true,
  "fraud_type_predicted": "social_engineering",
  "anomaly_flag": false,
  "explanation": "Transaksi ini terindikasi sebagai social engineering...",
  "recommended_action": "BLOCK",
  "signals": { "receiver_account_age_days_low": true, "..." : "..." }
}
```

---

## 📁 Struktur Project

```
GOTCHA/
├── main.py                  # FastAPI backend
├── train_model.py           # Script training model
├── generate_dataset.py      # Script generate dataset
├── models/
│   ├── random_forest.pkl
│   ├── isolation_forest.pkl
│   ├── encoders.pkl
│   ├── feature_columns.pkl
│   └── model_report.txt
└── frontend/
    ├── src/
    │   ├── App.jsx          # Dashboard utama
    │   ├── main.jsx
    │   └── index.css
    ├── package.json
    └── vite.config.js
```

---

## 👤 Pengembang

**Filbert Chang**
Microsoft Elevate AI Impact Challenge 2025

---

## 📄 Lisensi Dataset

Dataset GOTCHA-ID dibuat sendiri oleh peserta berdasarkan statistik agregat publik dari OJK dan Bank Indonesia. Tidak mengandung data pribadi atau data yang dilindungi hak cipta.