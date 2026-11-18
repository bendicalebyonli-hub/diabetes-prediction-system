# Design and Implementation of a Machine Learning-Based Diabetes Prediction System Using the Pima Indians Diabetes Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-teal.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-059669.svg)](https://fastapi.tiangolo.com/)
[![Next.js: 14](https://img.shields.io/badge/Next.js-14.2-black.svg)](https://nextjs.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.4-3178C6.svg)](https://www.typescriptlang.org/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-3.4-38B2AC.svg)](https://tailwindcss.com/)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Deployment_Ready-black.svg)](https://vercel.com/)

---

## Academic Abstract

This project presents an end-to-end, clinically contextualized machine learning system designed to predict the 5-year onset of Type 2 diabetes mellitus using the authentic **Pima Indians Diabetes Dataset** (National Institute of Diabetes and Digestive and Kidney Diseases, 768 instances). 

Addressing a common methodological pitfall in biomedical machine learning literature—the mishandling of physiological zero measurements—this project implements a **leakage-free preprocessing pipeline** that masks biologically impossible zeros on five clinical attributes (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) and applies robust median imputation learned strictly within cross-validation folds. 

Three classification algorithms (**Logistic Regression**, **Random Forest**, and **Support Vector Machine**) are rigorously benchmarked using 5-Fold Stratified Cross-Validation and evaluated on an untouched stratified hold-out test set ($n=154$). The selected champion model achieves a **Test ROC-AUC of 0.8385**, a **Test Accuracy of 77.27%**, and an **F1-Score of 0.6531**. The model is serialized and deployed through a high-performance **FastAPI** REST microservice and exposed via an interactive, responsive **Next.js 14** web application optimized for desktop, tablet, mobile, and seamless **Vercel** serverless deployment.

---

## Healthcare Safety & Ethical Governance

> [!WARNING]
> **ACADEMIC DEMONSTRATION ONLY — NOT A CLINICAL DIAGNOSTIC TOOL**  
> This software is engineered strictly for higher education coursework, algorithmic reproducibility, and software engineering demonstration. It does not possess regulatory clearance (e.g., FDA, CE-IVD), does not claim clinical diagnostic validity, and must **never** be utilized for patient diagnosis, clinical screening, or treatment selection.

### Historical Dataset Context & Demographic Constraints
* **Original Cohort:** The data was gathered by the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and analyzed in 1990 (*Smith et al.*).
* **Population Demographics:** Every record represents a female subject of Pima Indian heritage aged $\ge 21$ years living near Phoenix, Arizona, USA.
* **Generalizability Warning:** The historical Pima Indian cohort had a baseline diabetes prevalence of **34.9%**, significantly higher than the general global adult population (~10%). Consequently, models trained on this cohort cannot be uncritically generalized to male patients, pediatric populations, or other ancestral demographics.
* **No Medical Recommendations:** Predictions indicate statistical model risk probability, not definitive diagnostic truth.

---

## Technology Stack

| Domain | Technologies |
| :--- | :--- |
| **Machine Learning Core** | Python, Scikit-Learn, Pandas, NumPy, Joblib |
| **Exploratory Data Analysis** | Matplotlib, Seaborn |
| **Backend & REST API** | FastAPI, Pydantic v2, Uvicorn |
| **Frontend Web Client** | Next.js 14 (App Router), React 18, TypeScript, Tailwind CSS, Lucide Icons |
| **Testing & Quality Assurance** | Pytest, HTTPX, FastAPI TestClient |
| **DevOps & Cloud Hosting** | Git, GitHub, Vercel Serverless Platform |

---

## Machine Learning Architecture

```
                                  [Raw Pima Indians Dataset (768 records)]
                                                     │
                                                     ▼
                                      [Stratified Split: 80% Train / 20% Test]
                                                     │
                             ┌───────────────────────┴────────────────────────┐
                             │                                                │
                             ▼                                                ▼
                   [X_train (614 samples)]                          [X_test (154 samples)]
                             │                                                │
                             ▼                                                │
             ┌──────────────────────────────┐                                 │
             │   5-Fold Stratified CV       │                                 │
             │   1. BiologicalZeroToNan     │                                 │
             │   2. SimpleImputer (median)  │                                 │
             │   3. StandardScaler          │                                 │
             │   4. Estimator Training      │                                 │
             └───────────────┬──────────────┘                                 │
                             │                                                │
                             ▼                                                ▼
     [Model Selection: LR vs. RF vs. SVM] ────────────► [Evaluate on Untouched Test Set]
                             │                                                │
                             ▼                                                ▼
               [Pipeline Serialization]                          [Real Empirical Metrics]
            (backend/models/*.joblib)                           (ROC-AUC: 0.8385, Acc: 77.27%)
```

---

## Dataset Attributes & Physiological Zero Analysis

The dataset consists of 8 clinical predictor features and 1 binary outcome:

| # | Attribute Name | Units | Range in Cohort | Zero Count | Zero Pct (%) | Physiological Zero? | Clinical Rationale |
|---|---|---|---|---|---|---|---|
| 1 | `Pregnancies` | Count | 0 – 17 | 111 | 14.45% | **Valid** | Nulliparous patients have 0 pregnancies. |
| 2 | `Glucose` | mg/dL | 0 – 199 | 5 | 0.65% | **Invalid / Missing** | Blood glucose of 0 is fatal; indicates unrecorded test. |
| 3 | `BloodPressure`| mm Hg | 0 – 122 | 35 | 4.56% | **Invalid / Missing** | Diastolic BP of 0 indicates circulatory failure or missing reading. |
| 4 | `SkinThickness`| mm | 0 – 99 | 227 | 29.56% | **Invalid / Missing** | Triceps skin fold is never 0 mm; missing test. |
| 5 | `Insulin` | $\mu$U/mL | 0 – 846 | 374 | 48.70% | **Invalid / Missing** | Fasting/post-load serum insulin is never true zero; 48.7% unrecorded. |
| 6 | `BMI` | $\text{kg/m}^2$ | 0 – 67.1 | 11 | 1.43% | **Invalid / Missing** | Body Mass Index cannot be zero in living subjects. |
| 7 | `DiabetesPedigreeFunction` | Score | 0.078 – 2.42 | 0 | 0.00% | **Valid** | Genetic ancestral family history scoring function. |
| 8 | `Age` | Years | 21 – 81 | 0 | 0.00% | **Valid** | Cohort inclusion criteria required age $\ge 21$. |
| 9 | `Outcome` | Binary | 0 or 1 | 500 (0) | 65.10% | **Target** | 0 = Non-Diabetic, 1 = Diabetic (5-year follow-up). |

### Leakage-Free Preprocessing Strategy
1. **Targeted Zero-to-NaN Replacement:** Only features 2 through 6 are masked to `np.nan`. `Pregnancies=0` is strictly preserved.
2. **Median Imputation (`SimpleImputer(strategy='median')`):** Due to severe positive right-skewness in `Insulin` and `SkinThickness`, median imputation is far more robust than mean imputation.
3. **Strict Fold Isolation:** Imputer medians and scaling parameters ($\mu, \sigma$) are computed **only** on the training partition ($X_{train}$) and transformed onto test holdouts ($X_{test}$) to eliminate optimistic data snooping.

---

## Empirical Benchmark & Model Evaluation Results

Models evaluated on the untouched hold-out test set ($n=154$, 100 negative, 54 positive):

| Model Name | 5-Fold CV ROC-AUC | Test Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** (Selected) | **0.8354** | **77.27%** | **72.73%** | **59.26%** | **87.00%** | **0.6531** | **0.8385** |
| **Random Forest** (n=150) | 0.8312 | 76.62% | 72.09% | 57.41% | 87.00% | 0.6392 | 0.8322 |
| **Support Vector Machine** (RBF) | 0.8248 | 75.97% | 71.43% | 55.56% | 87.00% | 0.6250 | 0.8265 |

### Champion Model Confusion Matrix ($n=154$)
* **True Negatives (TN):** $87$ (Accurately identified non-diabetic subjects)
* **False Positives (FP):** $13$ (Type I error rate: $13.0\%$)
* **False Negatives (FN):** $22$ (Type II error rate: $40.7\%$)
* **True Positives (TP):** $32$ (Accurately identified diabetic subjects)

---

## Project Directory Structure

```
HOSPITAL_PREDICTION/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                          # FastAPI service & CORS middleware
│   │   ├── model.py                         # Inference engine & physiological risk analyzer
│   │   └── schemas.py                       # Pydantic v2 schemas with clinical bounds
│   ├── models/
│   │   ├── diabetes_best_pipeline.joblib    # Serialized Scikit-Learn pipeline
│   │   └── model_metadata.json              # Architecture & benchmark metadata
│   ├── tests/
│   │   ├── test_api.py                      # FastAPI integration tests
│   │   ├── test_pipeline.py                 # Preprocessing & leakage tests
│   │   └── test_schemas.py                  # Pydantic validation tests
│   └── requirements.txt                     # Backend dependencies
├── data/
│   └── raw/
│       ├── dataset_info.md                  # Dataset source documentation
│       └── diabetes.csv                     # Authentic 768-sample Pima dataset
├── ml/
│   ├── artifacts/
│   │   ├── eda_plots/                       # Generated distributions & zero-value charts
│   │   └── evaluation_plots/                # ROC curves & confusion matrices
│   └── src/
│       ├── data_loader.py                   # Automated ingestion & schema verification
│       ├── eda.py                           # Clinical exploratory data analysis
│       ├── evaluate.py                      # Metrics & plot generator
│       ├── preprocessor.py                  # Leakage-free transformers
│       └── train.py                         # 5-Fold Stratified CV trainer
├── frontend/                                # Next.js 14 App Router, TypeScript & Tailwind
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/
│   │   │   │   ├── model-info/route.ts      # Model metadata endpoint
│   │   │   │   └── predict/route.ts         # Edge serverless inference engine
│   │   │   ├── globals.css
│   │   │   ├── layout.tsx
│   │   │   └── page.tsx                     # Main interactive diagnostic workbench
│   │   ├── components/
│   │   │   ├── ClinicalDisclaimer.tsx       # Mandatory healthcare disclaimer banner
│   │   │   ├── DatasetInfoModal.tsx         # Historical provenance modal
│   │   │   ├── ModelMetricsCard.tsx         # Benchmark table & confusion matrix
│   │   │   ├── PredictionForm.tsx           # Validated interactive input form
│   │   │   └── ResultCard.tsx               # Probability gauge & risk factor breakdown
│   │   └── types/
│   │       └── prediction.ts                # TypeScript interfaces
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   └── vercel.json                          # Vercel deployment config
├── requirements.txt                         # Root Python dependencies
└── README.md                                # Comprehensive academic documentation
```

---

## Local Installation & Reproduction Guide

### 1. Prerequisites
* Python 3.10 or higher
* Node.js 18.x or higher & npm

### 2. Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/<your-username>/diabetes-prediction-system.git
cd HOSPITAL_PREDICTION

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Ingestion, Clinical EDA & Model Training
```bash
# 1. Inspect data and generate EDA plots
python -m ml.src.eda

# 2. Run 5-Fold Stratified Cross-Validation & serialize best pipeline
python -m ml.src.train
```

### 4. Execute Automated Test Suite
```bash
pytest backend/tests/ -v
```

### 5. Launch FastAPI Backend Service
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
* **Interactive Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **ReDoc Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Health Check:** `curl http://127.0.0.1:8000/health`

### 6. Launch Next.js 14 Frontend Application
In a separate terminal:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## FastAPI REST API Specification

### Endpoint: `POST /predict`
* **Request Payload (`application/json`):**
```json
{
  "pregnancies": 2,
  "glucose": 145.0,
  "blood_pressure": 76.0,
  "skin_thickness": 32.0,
  "insulin": 130.0,
  "bmi": 31.5,
  "diabetes_pedigree_function": 0.58,
  "age": 44
}
```

* **Response Payload (`200 OK`):**
```json
{
  "prediction": 1,
  "class_label": "Diabetic",
  "probability": 0.6241,
  "risk_category": "High Risk",
  "confidence_score": 62.41,
  "contributing_factors": [
    {
      "feature": "Glucose",
      "value": 145.0,
      "status": "Elevated",
      "clinical_note": "Values between 140-199 mg/dL reflect Impaired Glucose Tolerance."
    },
    {
      "feature": "BMI",
      "value": 31.5,
      "status": "Obese (Class I)",
      "clinical_note": "BMI >= 30 is a primary independent risk factor for Type 2 diabetes onset."
    }
  ],
  "model_version": "1.0.0",
  "clinical_safety_notice": "ACADEMIC DEMONSTRATION ONLY: This tool is an educational machine learning demonstration..."
}
```

---

## Vercel Cloud Deployment

The Next.js frontend is engineered for instantaneous deployment to Vercel:

1. Push your repository to GitHub.
2. Log into [Vercel](https://vercel.com/) and click **Add New Project**.
3. Import the repository and set the **Root Directory** to `frontend`.
4. (Optional) If running the separate FastAPI backend on Railway/Render/AWS, add the environment variable:
   ```
   NEXT_PUBLIC_API_URL=https://your-fastapi-backend.com
   ```
   *Note: If no external backend URL is specified, the application automatically runs its embedded Next.js Serverless Edge Pipeline, delivering 100% functionality out of the box on Vercel with zero external server dependencies!*
5. Click **Deploy**.

---

## Automated Testing Coverage

The automated test suite in `backend/tests/` verifies:
* **Schema Validation (`test_schemas.py`):** Ensures age $< 21$ is rejected per cohort boundaries, non-negative bounds are enforced, and unmeasured zeros are permitted for imputation.
* **Pipeline Integrity (`test_pipeline.py`):** Confirms zeros are masked to NaNs only on biological features, pregnancies ($0$) are conserved, and no NaNs emerge from the fitted pipeline.
* **Endpoint Contract (`test_api.py`):** Tests `/health`, `/model-info`, `/predict` with low-risk and high-risk physiological profiles.

---

## References & Academic Citations

1. **Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).** *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus.* Proceedings of the Annual Symposium on Computer Application in Medical Care, pp. 261–265.
2. **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).** *Pima Indians Diabetes Database.* Distributed via UCI Machine Learning Repository & Kaggle.
3. **American Diabetes Association (ADA).** *Standards of Medical Care in Diabetes—2024.* Diabetes Care 2024; 47 (Suppl. 1).
4. **Pedregosa, F. et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.
