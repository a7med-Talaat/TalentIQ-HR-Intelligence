# ⚡ TalentIQ: Candidate Progression & Talent Recommendation System

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Optuna%20Tuned-green.svg)](https://lightgbm.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-orange.svg)](https://shap.readthedocs.io/)

> **TalentIQ** is an end-to-end Machine Learning and Decision Intelligence platform that identifies which candidates are genuinely ready to make a career transition and algorithmically ranks the **Top 10 recommended applicants** using Multi-Criteria Decision Analysis (MCDA) and Explainable AI (SHAP).
> TalentIQ Team : Ahmed Talaat, Linda Nasser, Mohamed Ibrahim

---

## 💻 Project Demo :

https://talentiq-hr-system.streamlit.app/

---

## 📌 Project Highlights

- **Predictive Intelligence**: LightGBM classifier tuned with Optuna and calibrated with **Isotonic Regression** to produce statistically accurate transition probabilities.
- **Decision Engine (MCDA)**: Synthesizes transition intent with verified tenure, education level, upskilling intensity, and domain relevance into a **0–100% Recruiter Match Score**.
- **Explainable AI (XAI)**: Integrated **TreeSHAP** generates candidate-level waterfall contribution charts and cohort diagnostics.
- **Dual Interface**:
  - **FastAPI Backend**: Production REST API with Swagger documentation (`http://localhost:8000/docs`).
  - **Streamlit Dashboard**: Recruiter visual control center featuring podiums, competency radar charts, and a 360° candidate inspector (`http://localhost:8501`).
- **Comprehensive Project Report**: Full technical specification and defense prep available as [PROJECT_REPORT.pdf](./PROJECT_REPORT.pdf).

---

## 🏗️ Architecture & Pipeline Flow

```
Datasets/1_Raw_Data ─────────► data_preprocessing.py
                                       │ (Imputation, City Target Encoding, SMOTE)
                                       ▼
                              feature_engineering.py
                                       │ (6 Domain Interaction Ratios)
                                       ▼
                              feature_selection.py
                                       │ (Variance Filter, VIF Diagnosis, RFE)
                                       ▼
                              train_model.py
                                       │ (Optuna Tuning + Isotonic Calibration)
                                       ▼
                              Trained_Model/
                                ├── best_model.pkl
                                ├── preprocessor.pkl
                                └── feature_pipeline.pkl
                                       │
                 ┌─────────────────────┴─────────────────────┐
                 ▼                                           ▼
            app.py (FastAPI)                       dashboard.py (Streamlit)
        REST Endpoints / Swagger                      Interactive Recruiter UI
```

---

## 📂 Repository Structure

```
.
├── Datasets/
│   ├── 1_Raw_Data/                  # Original unprocessed datasets (aug_train.csv, aug_test.csv)
│   ├── 2_Preprocessed_Data/         # Cleaned & encoded datasets
│   ├── 3_Feature_Engineered_Data/   # Enriched with domain ratios
│   ├── 4_Selected_Features/         # Reduced to 15 optimal features
│   ├── 5_Predictions_and_Rankings/  # Scored candidates & Top 10 shortlists
│   └── README.md                    # Detailed dataset stage breakdown
│
├── Trained_Model/                   # ⭐ Standalone Submission Deliverable Folder
│   ├── best_model.pkl               # Calibrated champion model
│   ├── preprocessor.pkl             # Fitted preprocessing pipeline
│   ├── feature_pipeline.pkl         # Fitted feature pipeline
│   ├── selected_features.json       # Top 15 selected features
│   ├── model_metrics.json           # Performance metrics audit
│   ├── test_model_loading.py        # Automated test with random candidate simulation
│   ├── run_test.bat                 # 1-Click double-clickable test launcher
│   └── README.md                    # Deliverable verification guide
│
├── outputs/                         # SHAP figures, summary plots, and CSV reports
│   ├── shap_plots/                  # Candidate waterfall plots
│   ├── shap_summary.png             # Global cohort SHAP summary
│   └── vif_analysis.csv             # Multicollinearity analysis
│
├── app.py                           # FastAPI REST service
├── candidate_ranking.py             # Multi-criteria talent ranking engine
├── dashboard.py                     # Streamlit web dashboard
├── data_preprocessing.py            # Preprocessing & SMOTE pipeline
├── explainability.py                # SHAP explanation module
├── feature_engineering.py           # Domain feature engineering
├── feature_selection.py             # VIF analysis & RFE feature selector
├── train_model.py                   # Automated Optuna tuning & calibration
├── requirements.txt                 # Project dependencies
├── PROJECT_REPORT.pdf               # Comprehensive technical report (PDF)
└── README.md                        # This file
```

---

## 🚀 Quick Start Guide

### 1. Clone & Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/talentiq.git
cd talentiq

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Deliverable (1-Click Test)

Run the standalone verification script (simulates a random applicant from the test set):

```bash
python Trained_Model/test_model_loading.py
```
*On Windows, you can also simply double-click `Trained_Model/run_test.bat`.*

### 3. Launch the FastAPI Backend

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
- Open Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Available endpoints: `/predict`, `/predict/batch`, `/rank/top10`, `/candidates/top10`, `/candidates/all`, `/explain/{id}`

### 4. Launch the Recruiter Dashboard

```bash
streamlit run dashboard.py
```
- Open in your browser: [http://localhost:8501](http://localhost:8501)
- Explore Top 10 leaderboards, adjust weighting sliders, inspect 360° candidate profiles, and review SHAP interview probes.

---

## 🎯 Candidate Scoring Formulas & Presets

The composite score ($0–100\%$) evaluates candidates across 5 pillars:

$$\text{Score} = \left( w_{\text{intent}} S_{\text{intent}} + w_{\text{exp}} S_{\text{exp}} + w_{\text{edu}} S_{\text{edu}} + w_{\text{train}} S_{\text{train}} + w_{\text{rel}} S_{\text{rel}} \right) \times 100$$

| Strategy Preset | Intent | Experience | Education | Training | Relevance | Focus |
|---|---|---|---|---|---|---|
| **Balanced** | 35% | 25% | 20% | 10% | 10% | Standard recruiting (Recommended) |
| **Senior Leadership** | 20% | 40% | 25% | 5% | 10% | Staff / Principal / Leadership hiring |
| **Fast-Hire** | 60% | 15% | 10% | 5% | 10% | Urgent requisitions prioritizing immediate mobility |
| **High-Growth** | 25% | 15% | 15% | 35% | 10% | Upskillers, apprentices & junior high-achievers |

---

## 📄 Complete Project Report

For exhaustive technical documentation, mathematical formulations, feature selection rationale, calibration curves, and instructor viva Q&A, please refer to:
👉 **[PROJECT_REPORT.pdf](./PROJECT_REPORT.pdf)**

---

## 👥 Contributors & License

- Developed as part of the **Information Technology Institute (ITI) AI & Machine Learning Track**.
- Distributed under the MIT License.
