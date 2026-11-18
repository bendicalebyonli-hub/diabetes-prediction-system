# Pima Indians Diabetes Dataset Documentation

## Source & Attribution
* **Original Source:** National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)
* **Distributor:** UCI Machine Learning Repository / Kaggle
* **Original Study Date:** 1990 (Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. "Using the ADAP Learning Algorithm to Forecast the Onset of Diabetes Mellitus")
* **Target Population:** Female patients $\ge 21$ years old of Pima Indian heritage residing near Phoenix, Arizona, USA.
* **Total Instances:** 768 records
* **Attributes:** 8 physiological/clinical predictor variables, 1 binary target outcome.

## Attribute Specifications
| # | Feature Name | Description | Clinical Unit | Expected Range | Physiological Zero? |
|---|---|---|---|---|---|
| 1 | `Pregnancies` | Number of times pregnant | Count | 0 - 17 | **Yes** (Nulliparous) |
| 2 | `Glucose` | Plasma glucose concentration at 2 hours in an oral glucose tolerance test | mg/dL | 44 - 199 | **No** (0 indicates missing measurement) |
| 3 | `BloodPressure` | Diastolic blood pressure | mm Hg | 24 - 122 | **No** (0 indicates missing measurement) |
| 4 | `SkinThickness` | Triceps skin fold thickness | mm | 7 - 99 | **No** (0 indicates missing measurement) |
| 5 | `Insulin` | 2-Hour serum insulin | $\mu$U/mL | 14 - 846 | **No** (0 indicates missing measurement) |
| 6 | `BMI` | Body mass index ($\text{weight in kg}/(\text{height in m})^2$) | $\text{kg/m}^2$ | 18.2 - 67.1 | **No** (0 indicates missing measurement) |
| 7 | `DiabetesPedigreeFunction` | Genetic risk scoring function based on ancestral family history | Score | 0.078 - 2.42 | **No** (Always positive) |
| 8 | `Age` | Patient age | Years | 21 - 81 | **No** ($\ge 21$ per study inclusion) |
| 9 | `Outcome` | Diabetes diagnosis within 5 years | Binary | 0 or 1 | Target: 0 = Negative, 1 = Positive |

## Clinical Demographics & Historical Limitations
1. **Demographic Specificity:** The dataset consists solely of adult females of Pima Indian heritage. Extrapolating these models to general diverse populations requires extreme caution due to distinct genetic, environmental, and socio-demographic factors.
2. **High Baseline Prevalence:** Pima Indians historically exhibited an unusually elevated rate of Type 2 diabetes (~35% in this cohort vs. ~10% in general adult populations), which shifts the prior probability distribution.
3. **Missing Data Encoding:** Biological measurements such as insulin, blood pressure, glucose, and skin thickness were recorded as `0` when tests were unperformed or data was missing. Treating these as numerical zeros corrupts statistical inference unless imputed.
4. **Academic Use:** This dataset is utilized strictly for pedagogical, statistical, and academic benchmarking purposes. It is **not** a validated clinical diagnostic system.
