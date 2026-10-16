# 🍔 Swiggy Enterprise Analytics Copilot (Autonomous SQL + RAG)

An enterprise-grade decision-intelligence application combining a 6-table MySQL Star Schema (158K+ records) with ChromaDB semantic vector retrieval over 44K+ customer reviews, deployed as a dual-pane Streamlit operations control tower.

---

## 🚀 Key Architectural Features
- **Autonomous Tool-Calling Agent:** Leverages LangChain and Groq (`openai/gpt-oss-120b`) to dynamically route between relational Text-to-SQL querying, unstructured sentiment analysis, and general business Q&A.
- **Defensive Self-Healing SQL:** Implements read-only AST safety validation, automatic `LIMIT 50` caps, and dynamic exception handling that feeds MySQL tracebacks back to the agent for automated query correction.
- **Unstructured Vector RAG:** Local semantic search over 44K+ review notes using Hugging Face `all-MiniLM-L6-v2` dense embeddings stored in persistent ChromaDB.
- **Temporal Circuit Breaker:** Detects relative real-time queries (e.g., current month) and bypasses database connections to prevent `CURRENT_DATE()` null returns on historical synthetic cohorts.
- **Executive Operations UI:** Dual-pane Streamlit interface featuring live platform telemetry KPIs (158.4K orders, ₹35.5M GMV), automatic Plotly dynamic bar charts, 1-click CSV exports, and multi-session chat workspaces.

---

## 🗄️ Data Architecture (Star Schema)
- **Fact Table:** `fact_transactions` (158,400 rows)
- **Dimension Tables:** `dim_restaurant_product`, `dim_customer`, `dim_coupon`, `dim_campaign`, `dim_geo`

---

## 🛠️ Tech Stack
- **Languages & Frameworks:** Python 3.10+, LangChain, Streamlit, Plotly Express
- **Databases:** MySQL 8.0 (Relational Star Schema), ChromaDB (Vector Store)
- **Models:** Groq Cloud LPUs (`openai/gpt-oss-120b`), Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
- **Database Connectors:** SQLAlchemy, PyMySQL

---

## ⚡ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/swiggy-enterprise-analytics-copilot.git](https://github.com/your-username/swiggy-enterprise-analytics-copilot.git)
   cd swiggy-enterprise-analytics-copilot
