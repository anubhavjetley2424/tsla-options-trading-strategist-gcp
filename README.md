# 🚀 TSLA Options Trading Strategist — Full GCP Pipeline  
**Real-Time Market Microstructure + Catalyst Intelligence + Multi-Agent Options Strategy Engine**

This repository contains the **entire production pipeline** for the TSLA Options Strategist project, structured into three deployable services and one frontend dashboard:

root/
│
├── airflow/ # ETL + ingestion + automation
│ ├── dags/
│ ├── scripts/
│ ├── docker-compose.yaml
│ ├── Dockerfile
│ └── requirements.txt
│
├── cloud_run/ # TSLA Strategy Agent API (Gemini + Grok)
│ ├── main.py
│ ├── Dockerfile
│ ├── .env
│ └── requirements.txt
│
├── market_service/ # Real-time market + options chain API
│ ├── market_service.py
│ ├── Dockerfile
│ └── requirements.txt
│
└── react-app/
└── tsla-agent-app/ # Production React dashboard
├── src/
├── public/
└── package.json

yaml
Copy code

---

# 🧠 Overview of Each Component

### **1. `market_service/` – Real-Time Market Data API**
Your internal service providing:
- TSLA price  
- Full real-time options chain  
- IV, delta, gamma, OI, volume microstructure  
- Expiries list  

Used by the strategy engine to compute:
- Liquidity shock score  
- Gamma squeeze pressure  
- IV crush probability  
- OI walls and positioning bias  

Runs on Cloud Run or Cloud Run Jobs.

---

### **2. `cloud_run/` – TSLA Agent Backend (Gemini + Grok Fusion)**
This is **your main intelligent options strategist** containing:

- Catalyst extraction (Agent 2B → Grok RL refinement)  
- Needle-mover filtering (Agent 2C)  
- Scenario forecasting (Agent 2A)  
- Chain intelligence fusion (gamma/IV/OI computation)  
- Strategy generation (Agent 3A)  
- Strict JSON compiler (Agent 3B)  
- Fallback ATM call logic  
- BigQuery + Qdrant RAG snapshot integration  

Deployed as:
gcloud run deploy tsla-agent-api

yaml
Copy code

---

### **3. `airflow/` – Automated Data Ingestion & Snapshot Builder**
Airflow handles:
- Scheduled ingestion of macro/news datasets  
- EventArc → GCS CSV → BigQuery load jobs  
- Qdrant snapshot rebuilding (`/embed-snapshot`)  
- Daily consistency checks  

Runs in Docker Compose locally or in GKE/Cloud Composer.

---

### **4. `react-app/tsla-agent-app/` – Frontend Dashboard**
Production dashboard displaying:
- TSLA chart  
- Options chain  
- Liquidity pressure charts  
- Gamma/IV analytics  
- Catalyst list  
- Strategy output (formatted JSON → UI cards)  

Set the API endpoints in `.env` or `src/config.js`:

REACT_APP_STRATEGY_API=https://tsla-agent-api-xxxxx.a.run.app
REACT_APP_MARKET_API=https://market-data-xxxxx.a.run.app

yaml
Copy code

---

# 🏗️ Installation & Setup

## **1. Clone repository**
```bash
git clone https://github.com/<your-username>/tsla-options-trading-strategist-gcp.git
cd tsla-options-trading-strategist-gcp
2. Set up Market Service
bash
Copy code
cd market_service
pip install -r requirements.txt

# Local run
python market_service.py
Deploy to Cloud Run:

bash
Copy code
gcloud builds submit --tag gcr.io/$PROJECT_ID/market-service
gcloud run deploy market-service \
  --image gcr.io/$PROJECT_ID/market-service \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated
3. Set up TSLA Agent (cloud_run/)
bash
Copy code
cd cloud_run
pip install -r requirements.txt
Build and deploy:

bash
Copy code
gcloud builds submit --tag gcr.io/$PROJECT_ID/tsla-agent-api
gcloud run deploy tsla-agent-api \
  --image gcr.io/$PROJECT_ID/tsla-agent-api \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars PROJECT_ID=$PROJECT_ID \
  --set-env-vars VERTEX_REGION=us-central1 \
  --set-env-vars MARKET_API=<MARKET_API_URL> \
  --set-env-vars XAI_API_KEY=$XAI_API_KEY
4. Set up Airflow
From repo root:

bash
Copy code
cd airflow
docker-compose up --build
5. Set up React Dashboard
bash
Copy code
cd react-app/tsla-agent-app
npm install
npm start
For production (Vercel):

bash
Copy code
vercel deploy
📡 Calling the Strategy API
bash
Copy code
curl -X POST \
  $TSLA_AGENT_API/plan \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Momentum into CPI",
        "starting_capital": 15000,
        "risk_level": "high",
        "target_date": "2025-12-19"
      }'
🧬 Core Intelligence
✔ Grok real-time catalyst engine
✔ Real-time options chain gamma/IV analytics
✔ Liquidity shock detection
✔ Backed by Vertex Gemini Pro models
✔ RAG optional for deep macro context
✔ JSON-strict strategies for frontend execution
🔄 Workflow Summary
Frontend → Strategy API →
Grok RL Catalyst Engine →
Needle-Mover Filter →
Chain Intelligence Fusion →
Forecast Engine →
Strategy Agents →
Final JSON → UI.

