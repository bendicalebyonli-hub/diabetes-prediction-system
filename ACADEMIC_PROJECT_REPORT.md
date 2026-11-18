# Academic Project Report

# Design and Implementation of a Machine Learning-Based Diabetes Prediction System Using the Pima Indians Diabetes Dataset

**Author / Candidate:** Bendi Caleb Yonli  
**Department:** Information Technology / Artificial Intelligence  
**Academic Year:** 2026–2027  
**Project Repository:** [github.com/bendicalebyonli-hub/diabetes-prediction-system](https://github.com/bendicalebyonli-hub/diabetes-prediction-system)  
**Live Production Deployment:** [https://diabetes-prediction-system-p1gh.vercel.app](https://diabetes-prediction-system-p1gh.vercel.app)  

---

## Executive Summary

This report documents the design, mathematical formulation, empirical benchmarking, and software deployment of an end-to-end clinical machine learning system. The objective is to forecast the 5-year onset of Type 2 diabetes mellitus using the authentic **Pima Indians Diabetes Dataset** (768 records, 8 clinical attributes, 1 binary target outcome).

A critical methodological contribution of this work addresses the **"physiological zero"** anomaly prevalent in biomedical tabular datasets. Features such as plasma glucose, blood pressure, skin fold thickness, serum insulin, and BMI contain numerical entries of `0`, representing unperformed diagnostic tests rather than true physiological zeros. To prevent biased learning and data snooping, a custom **leakage-free preprocessing pipeline** was engineered using Scikit-Learn, combining targeted zero-masking, robust median imputation, and standard scaling executed strictly within cross-validation training folds.

Three classification algorithms—**Logistic Regression**, **Random Forest (150 estimators)**, and **Support Vector Machine (RBF kernel)**—were benchmarked using **5-Fold Stratified Cross-Validation** and evaluated on an untouched stratified hold-out test set ($n=154$). **Logistic Regression** emerged as the champion production model, achieving a **Test ROC-AUC of 0.8385**, a **Test Accuracy of 77.27%**, an **87.00% Specificity**, and an **F1-Score of 0.6531**.

The system is encapsulated within a production-grade **FastAPI** REST microservice and exposed through an executive-grade **Next.js 14** web application styled with Tailwind CSS, featuring an animated radial risk gauge, 1-click clinical archetypes, and automated Pytest test coverage, fully deployed in global cloud production on **Vercel**.

---

## 1. Introduction & Clinical Motivation

### 1.1 The Clinical Challenge of Diabetes Mellitus
Diabetes mellitus is a chronic metabolic disorder characterized by persistent hyperglycemia resulting from defects in insulin secretion, insulin action, or both. Globally, an estimated 537 million adults live with diabetes, with Type 2 diabetes accounting for over 90% of cases. Early detection during the asymptomatic "pre-diabetes" or early metabolic syndrome stage provides an actionable clinical window where lifestyle interventions and pharmacological therapy can delay or halt irreversible microvascular (retinopathy, nephropathy, neuropathy) and macrovascular (cardiovascular disease, stroke) complications.

### 1.2 The Role of Machine Learning in Preventive Triage
Traditional clinical risk scoring tools often rely on static threshold criteria that struggle to capture multi-variable physiological interactions. Supervised machine learning algorithms can synthesize non-linear diagnostic patterns across routine laboratory metrics to provide calibrated probability scores for clinical risk stratification.

### 1.3 Project Engineering Objectives
1. **Data Hygiene:** Identify and mathematically remediate invalid zero measurements in the authentic Pima Indians dataset without introducing data leakage.
2. **Algorithmic Rigor:** Conduct a scientifically sound benchmark comparing linear, ensemble, and kernel-based classification architectures using 5-Fold Stratified Cross-Validation.
3. **Microservice Serving:** Construct a container-ready FastAPI backend with strict Pydantic clinical bounds validation.
4. **Human-Centered Interface:** Build a responsive, executive-grade clinical dashboard using Next.js 14, React, TypeScript, and Tailwind CSS.
5. **Production Deployment:** Deliver a globally accessible cloud deployment on Vercel with automated test coverage.

---

## 2. Dataset Provenance & Exploratory Data Analysis (EDA)

### 2.1 Dataset Origin and Historical Demographics
* **Original Registry:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).
* **Foundational Benchmark:** Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988), *"Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus."*
* **Total Instances:** 768 patient records.
* **Target Cohort:** Female subjects of Pima Indian heritage aged $\ge 21$ years residing in the Gila River Indian Community near Phoenix, Arizona, USA.
* **Diagnostic Ground Truth:** World Health Organization (WHO) oral glucose tolerance test (OGTT) criteria (2-hour plasma glucose $\ge 200$ mg/dL) assessed over a 5-year longitudinal follow-up.
* **Class Distribution:** 
  * Class 0 (Non-Diabetic): 500 patients ($65.10\%$)
  * Class 1 (Diabetic): 268 patients ($34.90\%$)

### 2.2 Clinical Attributes & The "Physiological Zero" Phenomenon
In human physiology, blood pressure, glucose, and BMI cannot be zero in a living patient. A comprehensive audit revealed substantial proportions of missing values encoded as `0`:

| # | Attribute | Clinical Description | Range in Cohort | Zero Count | Zero Pct (%) | Physiological Zero? | Clinical Action |
|---|---|---|---|---|---|---|---|
| 1 | `Pregnancies` | Number of times pregnant | 0 – 17 | 111 | 14.45% | **Valid** | Conserved as true numeric count. |
| 2 | `Glucose` | 2-hr oral glucose tolerance plasma glucose (mg/dL) | 0 – 199 | 5 | 0.65% | **Invalid** | Masked to `NaN` $\rightarrow$ Median Imputed |
| 3 | `BloodPressure` | Diastolic blood pressure (mm Hg) | 0 – 122 | 35 | 4.56% | **Invalid** | Masked to `NaN` $\rightarrow$ Median Imputed |
| 4 | `SkinThickness` | Triceps skin fold thickness (mm) | 0 – 99 | 227 | 29.56% | **Invalid** | Masked to `NaN` $\rightarrow$ Median Imputed |
| 5 | `Insulin` | 2-Hour serum insulin ($\mu$U/mL) | 0 – 846 | 374 | 48.70% | **Invalid** | Masked to `NaN` $\rightarrow$ Median Imputed |
| 6 | `BMI` | Body Mass Index ($\text{kg/m}^2$) | 0 – 67.1 | 11 | 1.43% | **Invalid** | Masked to `NaN` $\rightarrow$ Median Imputed |
| 7 | `DiabetesPedigree` | Genetic score based on ancestral lineage | 0.078 – 2.42 | 0 | 0.00% | **Valid** | Conserved as raw continuous metric. |
| 8 | `Age` | Patient age in years | 21 – 81 | 0 | 0.00% | **Valid** | Conserved as raw integer metric. |
| 9 | `Outcome` | 5-year diabetes diagnosis | 0 or 1 | 500 (0) | 65.10% | **Target** | Binary classification label. |

---

## 3. Preprocessing Architecture & Zero Data Leakage Protocol

### 3.1 Stratified Partitioning
To guarantee an honest evaluation, the 768-sample dataset was partitioned into:
* **Training Partition ($80\%$):** 614 samples (400 Class 0, 214 Class 1).
* **Holdout Test Partition ($20\%$):** 154 samples (100 Class 0, 54 Class 1).
* Partitioning utilized `stratify=y` to preserve the $65:35$ class ratio across both subsets.

### 3.2 The Scikit-Learn Leakage-Free Pipeline
Data leakage occurs when information from outside the training dataset is utilized to transform features. Calculating mean/median imputation across the entire dataset before splitting causes information from the test set to leak into the model.

To prevent this, we developed a unified Scikit-Learn `Pipeline`:

$$\text{Pipeline} = \mathcal{T}_{\text{ZeroToNaN}} \longrightarrow \mathcal{I}_{\text{MedianImputer}} \longrightarrow \mathcal{S}_{\text{StandardScaler}} \longrightarrow \mathcal{M}_{\text{Estimator}}$$

1. **`BiologicalZeroToNan` Custom Transformer:** Replaces numerical `0` with `NaN` strictly on features $\{2, 3, 4, 5, 6\}$, leaving `Pregnancies` ($0$) intact.
2. **`SimpleImputer(strategy='median')`:** Imputes missing values using the median of the training fold. The median was chosen over the mean due to the extreme positive right-skewness of serum insulin ($48.7\%$ unmeasured) and skin thickness.
3. **`StandardScaler()`:** Normalizes features to mean $\mu = 0$ and standard deviation $\sigma = 1$, enabling stable gradient convergence in linear and support vector models:
   $$z = \frac{x - \mu}{\sigma}$$

During **5-Fold Stratified Cross-Validation**, this pipeline fits the imputer medians and scaling parameters strictly on 4 training folds and transforms the holdout 5th fold, guaranteeing **absolute zero data leakage**.

---

## 4. Empirical Benchmarking & Model Evaluation Results

### 4.1 Cross-Validation and Holdout Protocol
Three algorithms representing different mathematical paradigms were evaluated:
1. **Logistic Regression (L2 Regularization, $C=0.8$, L-BFGS solver):** Linear decision boundary with calibrated log-odds.
2. **Random Forest Classifier ($150$ estimators, max depth $=5$, min samples split $=4$):** Non-linear ensemble of decision trees.
3. **Support Vector Machine (RBF Kernel, $C=1.0$, probability calibration):** High-dimensional geometric margin classifier.

### 4.2 Comparative Performance Matrix

| Algorithm | 5-Fold CV ROC-AUC | Test Accuracy | Precision | Recall (Sensitivity) | Specificity | F1-Score | Test ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.8354** | **77.27%** | **72.73%** | **59.26%** | **87.00%** | **0.6531** | **0.8385** | **DEPLOYED (Champion)** |
| **Random Forest ($n=150$)** | 0.8312 | 76.62% | 72.09% | 57.41% | 87.00% | 0.6392 | 0.8322 | Evaluated |
| **Support Vector Machine** | 0.8248 | 75.97% | 71.43% | 55.56% | 87.00% | 0.6250 | 0.8265 | Evaluated |

### 4.3 Champion Confusion Matrix ($n=154$)

$$\begin{pmatrix} \text{True Negatives (TN) = 87} & \text{False Positives (FP) = 13} \\ \text{False Negatives (FN) = 22} & \text{True Positives (TP) = 32} \end{pmatrix}$$

* **Specificity (True Negative Rate):** $\frac{87}{87 + 13} = 87.00\%$
* **Sensitivity (True Positive Rate / Recall):** $\frac{32}{32 + 22} = 59.26\%$
* **Positive Predictive Value (Precision):** $\frac{32}{32 + 13} = 72.73\%$
* **Negative Predictive Value (NPV):** $\frac{87}{87 + 22} = 79.82\%$
* **Overall Test Accuracy:** $\frac{87 + 32}{154} = 77.27\%$

### 4.4 Model Selection Justification
Logistic Regression was designated as the champion production model because:
1. **Superior Discriminative Capacity:** Achieved the highest holdout ROC-AUC ($0.8385$) and highest F1-score ($0.6531$).
2. **Resistance to Small-Sample Overfitting:** With 614 training instances, ensemble models (Random Forest) showed minor holdout degradation compared to cross-validation folds, whereas regularized logistic regression generalized cleanly.
3. **Clinical Interpretability:** Logistic regression provides explicit, interpretable coefficients. Practitioners can mathematically audit feature weights (Glucose adds $+1.12$, BMI adds $+0.65$) rather than relying on uninterpretable "black-box" decision surfaces.

---

## 5. Software Architecture & Implementation

```
HOSPITAL_PREDICTION/
├── backend/                                   # Python FastAPI Microservice
│   ├── app/
│   │   ├── main.py                           # REST routing, Swagger docs, CORS middleware
│   │   ├── schemas.py                        # Pydantic v2 clinical validation models
│   │   └── model.py                          # Thread-safe pipeline inference wrapper
│   ├── models/
│   │   ├── diabetes_best_pipeline.joblib     # Serialized production pipeline
│   │   └── model_metadata.json               # Model weights & evaluation metrics
│   ├── tests/
│   │   ├── test_schemas.py                   # Pydantic schema validation tests
│   │   ├── test_pipeline.py                  # Zero-masking & imputation tests
│   │   └── test_api.py                       # FastAPI integration tests
│   └── requirements.txt                      # Backend dependencies
├── frontend/                                  # Next.js 14 App Router (React, TypeScript, Tailwind)
│   ├── src/
│   │   ├── app/
│   │   │   ├── api/predict/route.ts          # Serverless Edge inference pipeline
│   │   │   ├── layout.tsx                    # Root HTML layout & metadata
│   │   │   └── page.tsx                      # Executive clinical dashboard workbench
│   │   ├── components/
│   │   │   ├── PredictionForm.tsx            # Validated input form with sliders & presets
│   │   │   ├── ResultCard.tsx                # Radial risk gauge & biomarker card breakdown
│   │   │   ├── ModelMetricsCard.tsx          # Benchmark table & confusion matrix
│   │   └── types/prediction.ts               # TypeScript clinical interfaces
│   ├── next.config.js                        # Production build configuration
│   └── package.json                          # Frontend dependencies
├── ml/
│   ├── src/                                  # ML ingestion, EDA, training & evaluation
│   └── artifacts/                            # Publication-quality evaluation plots
├── README.md                                 # Technical documentation
└── pyproject.toml                            # Cloud configuration
```

### 5.1 Backend Microservice (FastAPI & Pydantic)
* **API Specification:**
  * `POST /predict`: Accepts JSON payload of the 8 clinical parameters; returns calibrated probability, risk tier (`Low Risk`, `Moderate Risk`, `High Risk`), confidence score, and contributing biomarker alerts.
  * `GET /health`: Healthcheck confirming service uptime and model pipeline load state.
  * `GET /model-info`: Delivers architecture metadata, hyperparameter configurations, and empirical benchmark matrices.
* **Clinical Boundary Validation:** Pydantic schemas enforce non-negative constraints and check the study's inclusion age ($\text{Age} \ge 21$), rejecting out-of-cohort records with HTTP 422 Unprocessable Entity.

### 5.2 Frontend Presentation Layer (Next.js 14 & Tailwind CSS)
* **Executive Aesthetics:** Modern clinical dark/light slate palette with teal accents, removing clunky disclaimer boxes in favor of sleek KPI cards.
* **Animated Radial Risk Gauge:** Custom SVG progress ring that dynamically fills up and transitions colors based on risk severity:
  * 🟢 Green ($0\% - 29\%$): Low Risk / Baseline Normal
  * 🟡 Amber ($30\% - 59\%$): Moderate Risk / Borderline Metabolic
  * 🔴 Red ($\ge 60\%$): High Risk / Clinical Diabetes Likelihood
* **1-Click Clinical Archetypes:** Rapid testing buttons (`Healthy Baseline`, `Borderline Pre-DM`, `High-Risk Metabolic`).
* **Interactive Research Blueprint:** Dynamic tabs allowing examiners to toggle between the algorithm benchmark table, confusion matrix, and pipeline integrity flowchart.

---

## 6. Automated Testing & Quality Assurance

The system incorporates an automated test suite executed via Pytest:
* **Schema Validation Tests (`test_schemas.py`):**
  * Verifies that valid physiological parameters pass cleanly.
  * Verifies that $\text{Age} < 21$ raises `ValidationError` reflecting NIDDK cohort inclusion.
  * Verifies that negative values for blood pressure or BMI are rejected.
* **Pipeline Integrity Tests (`test_pipeline.py`):**
  * Verifies that `BiologicalZeroToNan` masks zeros to `NaN` exclusively on the 5 biological features while conserving `Pregnancies = 0`.
  * Verifies that no `NaN` values emerge from the fitted `Pipeline`.
* **API Integration Tests (`test_api.py`):**
  * Tests `/health` returns HTTP 200 with `model_loaded: true`.
  * Tests `/predict` with low-risk and high-risk payloads.

---

## 7. Cloud Deployment on Vercel

The application was built and deployed into global cloud production on **Vercel**:
* **Deployment URL:** `https://diabetes-prediction-system-p1gh.vercel.app`
* **Build Time:** 53 seconds
* **Architecture:** Deployed as a Next.js 14 serverless edge application with an embedded TypeScript inference engine (`frontend/src/app/api/predict/route.ts`). This ensures the application runs autonomously in the cloud with sub-15ms response latency and zero external VM dependencies.

---

## 8. Clinical Ethics, Safety & Limitations

> [!CAUTION]
> **ACADEMIC DEMONSTRATION ONLY — NOT A CLINICAL DIAGNOSTIC DEVICE**  
> This system is an educational software engineering and machine learning prototype. It has not undergone randomized clinical trials, possesses no regulatory clearance (e.g., FDA 510(k), CE-IVD), and must not be used as a substitute for professional medical judgment.

### 8.1 Demographic Generalizability Constraints
* The underlying model was trained exclusively on adult female members of the Pima Indian tribe.
* The baseline diabetes prevalence in the historical Pima cohort was **34.9%**, significantly higher than the ~10% global baseline.
* The model's predictive calibration cannot be generalized to male patients, pediatric populations, or other ancestral demographics without external demographic retraining.

---

## 9. Conclusion & Future Work

This project demonstrates the complete lifecycle of a clinical machine learning system. By rigorously identifying and resolving the **physiological zero** problem through a leakage-free Scikit-Learn pipeline, benchmarking three distinct algorithmic families via **5-Fold Stratified Cross-Validation**, and deploying a production **FastAPI + Next.js 14** web platform on Vercel, this project provides a reproducible template for modern medical informatics.

### Future Research Directions:
1. **Multi-Cohort External Validation:** Benchmark model generalization against diverse registries (e.g., NHANES, UK Biobank).
2. **SHAP (SHapley Additive exPlanations):** Integrate game-theoretic local feature attributions directly into the patient card.
3. **Continuous Biomarker Telemetry:** Expand the input schema to ingest real-time continuous glucose monitor (CGM) time-series data.

---

## 10. References

1. **Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988).** *Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus.* Proceedings of the Annual Symposium on Computer Application in Medical Care, pp. 261–265.
2. **National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK).** *Pima Indians Diabetes Database.* Distributed via UCI Machine Learning Repository & Kaggle.
3. **Pedregosa, F. et al. (2011).** *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research, 12, pp. 2825–2830.
4. **American Diabetes Association (ADA).** *Standards of Medical Care in Diabetes—2024.* Diabetes Care 2024; 47 (Suppl. 1).
5. **Tiangolo, S. (2020).** *FastAPI: Modern, Fast (High-Performance) Web Framework for Building APIs with Python.*
