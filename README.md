# G.O.T.C.H.A
### Guard & Observe Transactions with Cognitive Hybrid AI

> A real-time AI-powered fraud detection system for digital financial transactions, calibrated specifically for Indonesian digital financial crime patterns.

---

## 🎯 About the Project

GOTCHA is a fraud detection intelligence platform designed for Indonesia's digital financial ecosystem. It combines a **Random Forest Classifier**, **Isolation Forest**, and **GPT-4o** to detect and explain suspicious transactions in real-time.

Built for the **Microsoft Elevate AI Impact Challenge** — Fraud Detection & Risk Management theme.

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
┌───────────────────────────────┐
│  Random Forest Classifier     │  ← Main model (supervised)
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

| Model | Type | ROC-AUC | Function |
|---|---|---|---|
| Random Forest | Supervised | 1.0000 | Fraud scoring & classification |
| Isolation Forest | Unsupervised | 0.5515 | Anomaly detection layer |
| GPT-4o | LLM | — | Explainability in Bahasa Indonesia |

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
pip install fastapi uvicorn scikit-learn pandas numpy python-dotenv openai tqdm
```

Create a `.env` file in the root folder:
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
  "risk_score": 1.0,
  "is_fraud": true,
  "fraud_type_predicted": "social_engineering",
  "anomaly_flag": false,
  "explanation": "This transaction is flagged as social engineering...",
  "recommended_action": "BLOCK",
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
├── models/
│   ├── random_forest.pkl
│   ├── isolation_forest.pkl
│   ├── encoders.pkl
│   ├── feature_columns.pkl
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

The GOTCHA-ID dataset was created by the developer based on publicly available aggregate statistics from OJK and Bank Indonesia. It contains no personal data or copyrighted data.
