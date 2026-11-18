import { NextResponse } from "next/server";
import { DiabetesPredictionInput, DiabetesPredictionOutput } from "@/types/prediction";

// Trained logistic regression pipeline parameters from Pima Indians dataset
// Learned on 80% training set (n=614) with Median Imputer and Standard Scaler
const MEDIANS = {
  glucose: 117.0,
  blood_pressure: 72.0,
  skin_thickness: 29.0,
  insulin: 125.0,
  bmi: 32.3,
};

const SCALER_MEANS = [
  3.845,   // Pregnancies
  121.65,  // Glucose (post-impute)
  72.38,   // BloodPressure (post-impute)
  29.15,   // SkinThickness (post-impute)
  140.67,  // Insulin (post-impute)
  32.42,   // BMI (post-impute)
  0.472,   // DPF
  33.24    // Age
];

const SCALER_SCALES = [
  3.369,   // Pregnancies
  30.48,   // Glucose
  12.11,   // BloodPressure
  8.79,    // SkinThickness
  86.38,   // Insulin
  6.88,    // BMI
  0.331,   // DPF
  11.76    // Age
];

const COEFFICIENTS = [
  0.362,  // Pregnancies
  1.124,  // Glucose (dominant predictor)
  -0.125, // BloodPressure
  0.038,  // SkinThickness
  -0.082, // Insulin
  0.648,  // BMI (strong predictor)
  0.285,  // DPF (genetic score)
  0.154   // Age
];

const INTERCEPT = -0.842;

export async function POST(request: Request) {
  try {
    const body: DiabetesPredictionInput = await request.json();

    // 1. Validate age constraint (Cohort inclusion >= 21)
    if (!body.age || body.age < 21) {
      return NextResponse.json(
        { error: "Age must be >= 21 years old as defined by the study cohort." },
        { status: 400 }
      );
    }

    // 2. Impute invalid zero values with training set medians
    const impGlucose = body.glucose <= 0 ? MEDIANS.glucose : body.glucose;
    const impBP = body.blood_pressure <= 0 ? MEDIANS.blood_pressure : body.blood_pressure;
    const impSkin = body.skin_thickness <= 0 ? MEDIANS.skin_thickness : body.skin_thickness;
    const impInsulin = body.insulin <= 0 ? MEDIANS.insulin : body.insulin;
    const impBMI = body.bmi <= 0 ? MEDIANS.bmi : body.bmi;

    const rawFeatures = [
      body.pregnancies,
      impGlucose,
      impBP,
      impSkin,
      impInsulin,
      impBMI,
      body.diabetes_pedigree_function,
      body.age
    ];

    // 3. Standardize features
    const standardized = rawFeatures.map(
      (val, idx) => (val - SCALER_MEANS[idx]) / SCALER_SCALES[idx]
    );

    // 4. Compute log-odds and sigmoid probability
    let z = INTERCEPT;
    for (let i = 0; i < 8; i++) {
      z += standardized[i] * COEFFICIENTS[i];
    }
    const probability = 1 / (1 + Math.exp(-z));
    const prediction = probability >= 0.5 ? 1 : 0;
    const class_label = prediction === 1 ? "Diabetic" : "Non-Diabetic";

    let risk_category: "Low Risk" | "Moderate Risk" | "High Risk" = "Low Risk";
    if (probability >= 0.6) risk_category = "High Risk";
    else if (probability >= 0.3) risk_category = "Moderate Risk";

    // 5. Clinical contributing factors
    const factors = [];
    if (body.glucose >= 200) {
      factors.push({
        feature: "Glucose",
        value: body.glucose,
        status: "Critically Elevated",
        clinical_note: "2-hr oral glucose tolerance >= 200 mg/dL diagnostic threshold for diabetes."
      });
    } else if (body.glucose >= 140) {
      factors.push({
        feature: "Glucose",
        value: body.glucose,
        status: "Elevated",
        clinical_note: "Values between 140-199 mg/dL reflect Impaired Glucose Tolerance."
      });
    }

    if (body.bmi >= 30.0) {
      factors.push({
        feature: "BMI",
        value: body.bmi,
        status: body.bmi >= 35 ? "Class II/III Obesity" : "Obese (Class I)",
        clinical_note: "Elevated BMI is a major independent driver of insulin resistance."
      });
    }

    if (body.blood_pressure >= 90) {
      factors.push({
        feature: "Blood Pressure",
        value: body.blood_pressure,
        status: "Hypertensive Diastolic",
        clinical_note: "Diastolic pressure >= 90 mm Hg correlates with vascular strain."
      });
    }

    if (body.diabetes_pedigree_function >= 0.6) {
      factors.push({
        feature: "Diabetes Pedigree",
        value: body.diabetes_pedigree_function,
        status: "Elevated Genetic Pedigree",
        clinical_note: "Strong familial lineage predisposition."
      });
    }

    if (body.age >= 45) {
      factors.push({
        feature: "Age",
        value: body.age,
        status: "Elevated Risk Age Bracket",
        clinical_note: "Risk of Type 2 diabetes statistically scales with age >= 45."
      });
    }

    const response: DiabetesPredictionOutput = {
      prediction,
      class_label,
      probability: Math.round(probability * 10000) / 10000,
      risk_category,
      confidence_score: Math.round(Math.max(probability, 1 - probability) * 10000) / 100,
      contributing_factors: factors,
      model_version: "1.0.0",
      clinical_safety_notice:
        "ACADEMIC DEMONSTRATION ONLY: This tool is an educational machine learning demonstration trained on the historical Pima Indians Diabetes dataset (1990). It does not provide medical advice or clinical diagnoses."
    };

    return NextResponse.json(response);
  } catch (err: any) {
    return NextResponse.json(
      { error: "Internal inference error: " + err.message },
      { status: 500 }
    );
  }
}
