# Diabetes Prediction Analytics

## Project Overview
Diabetes Mellitus represents one of the fastest-growing global chronic health crises. Conventional medical protocols frequently rely on reactive management, diagnosing conditions only after metabolic damage has manifested. Pre-diabetes represents a critical, reversible window of opportunity where proactive lifestyle modifications can reduce progression to Type 2 diabetes by up to 58% (ADA clinical evidence).

A State of the Art Machine Learning approach. Transforming clinical, laboratory, and lifestyle data into actionable preventive healthcare insights using an optimized **LightGBM Classifier** (91.84% accuracy, 0.92 weighted F1-score, and 1.00 recall on critical stages) for real-time diabetes risk stratification across three distinct stages: **No Diabetes**, **Pre-Diabetes**, and **Diabetes**.

---

## Objectives
1. **Early Interception of Pre-Diabetes:** Proactively stratifying patients into three clinical stages ('No Diabetes', 'Pre-Diabetes', and 'Diabetes') to enable early lifestyle and medical intervention before irreversible vascular complications occur.
2. **Identifying Key Metabolic Drivers:** Analyzing feature importance across 28 health indicators to reveal the most influential clinical factors (fasting glucose, age, physical activity, and HbA1c).
3. **Providing Scalable & Interpretable Solutions:** Deploying a containerized, decoupled architecture comprising a high-performance FastAPI backend service and an intuitive, interactive Streamlit clinical dashboard.

---

## Tech Stack
* **Machine Learning:** LightGBM Classifier, Scikit-learn (StandardScaler, StratifiedKFold), Joblib.
* **Backend:** FastAPI, Uvicorn, Pydantic v2, Python-dotenv.
* **Frontend:** Streamlit, PIL (Pillow), Requests.
* **Data Processing & Analytics:** Pandas, NumPy, Matplotlib, Seaborn.

---

## Project Structure
```text
.
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   └── services.py
│   ├── test_backend.py
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   ├── assets/
│   │   ├── certificate.png
│   │   ├── correlation.png
│   │   └── feature_importance.png
│   ├── app.py
│   ├── .env.example
│   └── requirements.txt
├── notebook/
│   ├── artifacts/
│   │   ├── feature_columns.json
│   │   ├── lgbm_model.joblib
│   │   ├── metadata.json
│   │   ├── presets.json
│   │   └── scaler.joblib
│   └── Diabetes_Predictive_Analytics.ipynb
└── README.md
```

## Getting Started
**Prerequisites**
- Python 3.9 or higher
- Virtual Environment (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/gentahape/diabetes-predictive-analytics.git
cd diabetes-predictive-analytics
```

### 2. Setup Environment Variables
Copy `.env.example` to `.env` in both `backend` and `frontend` directories and adjust the values:
```bash
# Setup Backend ENV
cd backend
cp .env.example .env

# Setup Frontend ENV
cd ../frontend
cp .env.example .env
cd ..
```

### 3. Backend Installation & Run
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
> API Docs (Swagger UI) available at `http://127.0.0.1:8000/docs`

### 4. Frontend Installation & Run
Open a new terminal:
```bash
cd frontend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```
> Interactive Web Dashboard available at `http://localhost:8501`

## Project Origins
This Machine Learning project originated as a final submission for the **Machine Learning Terapan** certification at [Dicoding](https://www.dicoding.com/academies/319). It was rigorously reviewed by industry experts, and I actively implemented their feedback to further deepen my practical understanding of advanced predictive modeling, hyperparameter tuning, and production-ready machine learning deployment.