<img width="1919" height="916" alt="Screenshot 2026-05-25 192920" src="https://github.com/user-attachments/assets/73721d4b-a6a8-4bec-a53b-312248240851" />
<img width="1919" height="921" alt="Screenshot 2026-05-25 192835" src="https://github.com/user-attachments/assets/c05ed409-d320-42ad-979f-657d11b387b4" />
<img width="1919" height="914" alt="Screenshot 2026-05-25 192904" src="https://github.com/user-attachments/assets/477e5528-7f2b-4fbc-b7db-f5adc0cd688e" />
# GOTCHA

> Fraud detection system for Indonesia's digital financial ecosystem, calibrated from OJK and Bank Indonesia statistics.

---

## About the Project

GOTCHA detects 5 common fraud types in Indonesia's digital financial ecosystem using **supervised classification** (calibrated Random Forest + SMOTE), **unsupervised anomaly detection** (Isolation Forest), and **LLM-based risk analysis** (GPT-4o) that provides an independent assessment of model output.

Built for the **Microsoft Elevate AI Impact Challenge 2026** — Fraud Detection & Risk Management theme.

---

## 🛡️ Types of Fraud Detected

| Fraud Type | Description |
|---|---|
| **Social Engineering** | Scams via WhatsApp/Telegram that manipulate victims |
| **Mule Accounts** | Dummy accounts used to hold proceeds from crimes |
| **QRIS Substitution Fraud** | Fake QR codes placed over legitimate merchant QRIS |
| **Fictitious QRIS Merchant** | Fake QRIS merchants used to collect payments |
| **Illegal P2P Lending** | Illegal online lending platforms on OJK's blacklist |

---

## 🏗️ Architecture

```
Frontend (React + Vite)
        ↓
Backend (FastAPI)
        ↓
┌─────────────────────────────────────────┐
│  Random Forest (Calibrated + SMOTE)     │  ← Binary fraud scoring (supervised)
│  Random Forest (Multi-class)            │  ← Fraud type classification
│  Isolation Forest (numeric features)    │  ← Anomaly detection (unsupervised)
│  GPT-4o (GitHub Models)                 │  ← Independent risk analysis
└─────────────────────────────────────────┘
        ↓
Azure Services
├── Azure OpenAI
└── Azure App Service (deployment)
```

---

## 📊 Dataset

**GOTCHA-ID Fraud Simulation Dataset** — a synthetic dataset of 1 million rows calibrated from official statistics:

- **Bank Indonesia 2024** — QRIS transaction distribution, average amounts, platforms
- **OJK IASC 2023** — fraud type distribution, financial losses, geographic spread

The dataset was generated using `generate_dataset.py` and contains no personal data whatsoever.

| Parameter | Value |
|---|---|
| Total rows | 1,000,000 |
| Fraud rate | 1.5% (15,000 transactions) |
| Number of columns | 31 |
| Simulation period | Jan 1 – Dec 31, 2024 |

---

## 🤖 AI Models

| Model | Type | Metric | Function |
|---|---|---|---|
| Random Forest (Calibrated) | Supervised | ROC-AUC 0.9989 | Binary fraud scoring |
| Random Forest | Multi-class | Accuracy 0.99 | Fraud type classification |
| Isolation Forest | Unsupervised | ROC-AUC 0.7100 | Anomaly detection layer |
| GPT-4o | LLM | — | Independent risk analysis |

**ML techniques used:**
- **SMOTE oversampling** to handle class imbalance (1.5% fraud rate)
- **Isotonic calibration** for accurate probability output (calibration error: 0.016)
- **Cost-sensitive threshold** — threshold optimized with cost ratio FN:FP = 10:1
- **Graph features** — `receiver_unique_senders_1h`, `sender_unique_receivers_1h` for mule network detection
- **Sequence features** — `amount_trend_3trx`, `avg_time_between_trx_1h` for escalation detection

---

## ⚙️ Tech Stack

**Backend:**
- Python 3.13
- FastAPI + Uvicorn
- scikit-learn (Random Forest, Isolation Forest)
- OpenAI SDK (GitHub Models / Azure OpenAI)

**Frontend:**
- React 19 + Vite
- Recharts (data visualization)
- Lucide React (icons)
- Axios (HTTP client)
- Premium Fintech UI (Revolut-inspired Design System)

**Azure Services:**
- Azure OpenAI (GPT-4o)
- Azure App Service (deployment)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/FilbertChang/GOTCHA.git
cd GOTCHA
```

### 2. Set up the backend
```bash
pip install fastapi uvicorn scikit-learn pandas numpy python-dotenv openai tqdm imbalanced-learn
```

Create a `.env` file in the root folder (see `.env.example`):
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

### 4. Run the backend
```bash
uvicorn main:app --reload
```

Backend runs at `http://localhost:8000`

### 5. Set up & run the frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Check server status |
| `GET` | `/stats` | Model statistics |
| `POST` | `/predict` | Analyze a transaction |
| `GET` | `/monitor` | Prediction monitoring & drift stats |

### Example `/predict` request:
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

### Example response:
```json
{
  "transaction_id": "TXN-001",
  "risk_score": 0.94,
  "is_fraud": true,
  "fraud_type_predicted": "social_engineering",
  "anomaly_flag": false,
  "analysis": {
    "explanation": "Large transfer (12.5x average) to a new account (1 day old) at 22:00 forms a classic social engineering pattern...",
    "risk_factors": [
      "Recipient account created only 1 day ago",
      "Amount is 12.5x the sender's average"
    ],
    "investigation_steps": [
      "Check if the recipient account received transfers from other senders in the last 24 hours",
      "Contact sender to verify — ask if someone reached out via WhatsApp/Telegram"
    ],
    "recommended_action": "BLOCK",
    "confidence_note": "High confidence — 4 signals active simultaneously is rare in legitimate transactions."
  },
  "signals": { "receiver_account_age_days_low": true, "...": "..." }
}
```

---

## 📁 Project Structure

```
GOTCHA/
├── main.py                  # FastAPI backend
├── train_model.py           # Model training script
├── generate_dataset.py      # Dataset generation script
├── .env.example             # Environment variable template
├── tests/
│   └── test_api.py          # API endpoint tests
├── models/                      # Generated by train_model.py (.gitignored)
│   ├── random_forest.pkl        # Calibrated RF (binary fraud scoring)
│   ├── fraud_type_classifier.pkl # RF (multi-class fraud type)
│   ├── isolation_forest.pkl
│   ├── encoders.pkl
│   ├── feature_columns.pkl
│   ├── iso_feature_cols.pkl
│   ├── optimal_threshold.pkl
│   └── model_report.txt
└── frontend/
    ├── src/
    │   ├── App.jsx          # Main dashboard
    │   ├── main.jsx
    │   └── index.css
    ├── package.json
    └── vite.config.js
```

---

## 👤 Developer

**Filbert Chang**
Microsoft Elevate AI Impact Challenge 2026

---

## 📄 Dataset License

The GOTCHA-ID dataset was created by the participant based on aggregate public statistics from OJK and Bank Indonesia. It contains no personal data or copyrighted material.
