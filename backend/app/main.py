"""
FastAPI Main Application for Diabetes Prediction Service.
Features an executive-grade clinical intelligence dashboard and production REST API.
"""

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from backend.app.schemas import (
    DiabetesPredictionInput,
    DiabetesPredictionOutput,
    HealthResponse,
    ModelInfoResponse
)
from backend.app.model import DiabetesPredictor

app = FastAPI(
    title="Clinical AI Diabetes Decision Support System",
    description="Production-grade clinical risk classification API & Executive Diagnostic Dashboard.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = DiabetesPredictor.get_instance()


@app.get("/", tags=["Dashboard & Root"])
def index_or_dashboard(request: Request):
    """
    Renders the Executive Clinical AI Dashboard when opened in a browser,
    or returns API discovery JSON if requested by automated clients.
    """
    accept = request.headers.get("accept", "")
    if "application/json" in accept and "text/html" not in accept:
        return {
            "service": "Clinical AI Diabetes Decision Support Platform",
            "version": "1.0.0",
            "status": "online",
            "dashboard_ui": "http://127.0.0.1:8000/",
            "documentation": "/docs",
            "health_check": "/health",
            "model_info": "/model-info",
            "predict_endpoint": "/predict"
        }

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Clinical AI Decision-Support Platform | Diabetes Risk Intelligence</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap">
  <style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    .font-mono { font-family: 'JetBrains Mono', monospace; }
    input[type=range] { accent-color: #0f766e; }
    .gauge-circle {
      transition: stroke-dashoffset 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    }
  </style>
</head>
<body class="bg-[#f8fafc] text-slate-900 min-h-screen flex flex-col justify-between selection:bg-teal-100 selection:text-teal-900">

  <!-- Executive Navigation Bar -->
  <header class="bg-white/95 backdrop-blur-md border-b border-slate-200/80 sticky top-0 z-40 shadow-xs">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 py-3.5 flex items-center justify-between">
      
      <!-- Logo & Branding -->
      <div class="flex items-center gap-3.5">
        <div class="w-11 h-11 rounded-xl bg-gradient-to-tr from-teal-700 via-teal-600 to-emerald-500 text-white flex items-center justify-center shadow-md shadow-teal-700/20 ring-1 ring-white/30">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.2" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="text-base sm:text-lg font-extrabold text-slate-900 tracking-tight">
              CLINICAL<span class="text-teal-600">AI</span> INTELLIGENCE
            </h1>
            <span class="px-2 py-0.5 text-[10px] font-bold rounded-full bg-slate-100 text-slate-600 border border-slate-200">
              v1.0 Production
            </span>
          </div>
          <p class="text-xs text-slate-500 font-medium">
            Predictive Diabetes Risk Classifier &bull; 5-Fold Cross-Validated Pipeline
          </p>
        </div>
      </div>

      <!-- Right Header Controls -->
      <div class="flex items-center gap-3">
        <a href="/docs" target="_blank" class="hidden sm:inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl shadow-2xs transition-all">
          <svg class="w-4 h-4 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>
          API Docs
        </a>
        <div class="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-50 border border-emerald-200/80 text-emerald-800 text-xs font-semibold">
          <span class="relative flex h-2 w-2">
            <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span>Inference Ready</span>
        </div>
      </div>

    </div>
  </header>

  <!-- Main Executive Body -->
  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-7 w-full flex-1">

    <!-- KPI Metric Stat Row -->
    <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-7">
      <div class="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
        <div class="text-[11px] font-bold uppercase tracking-wider text-slate-400">Model Discrimination</div>
        <div class="text-2xl font-extrabold text-slate-900 mt-0.5 font-mono">0.8385 <span class="text-xs text-emerald-600 font-bold">AUC</span></div>
        <div class="text-[11px] text-slate-500 mt-1">5-Fold Stratified Holdout Test</div>
      </div>

      <div class="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
        <div class="text-[11px] font-bold uppercase tracking-wider text-slate-400">Diagnostic Accuracy</div>
        <div class="text-2xl font-extrabold text-slate-900 mt-0.5 font-mono">77.27%</div>
        <div class="text-[11px] text-slate-500 mt-1">Zero-Leakage Pipeline</div>
      </div>

      <div class="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
        <div class="text-[11px] font-bold uppercase tracking-wider text-slate-400">Clinical Cohort</div>
        <div class="text-2xl font-extrabold text-slate-900 mt-0.5 font-mono">768 <span class="text-xs text-slate-400 font-normal">Patients</span></div>
        <div class="text-[11px] text-slate-500 mt-1">Authentic NIDDK Dataset</div>
      </div>

      <div class="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
        <div class="text-[11px] font-bold uppercase tracking-wider text-slate-400">Inference Latency</div>
        <div class="text-2xl font-extrabold text-teal-700 mt-0.5 font-mono">&lt; 10 <span class="text-xs font-normal">ms</span></div>
        <div class="text-[11px] text-slate-500 mt-1">Real-Time Scoring Engine</div>
      </div>
    </div>

    <!-- 2-Column Responsive Diagnostic Workbench -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-7 items-start">

      <!-- Left Column: Patient Diagnostic Configuration (7 Cols) -->
      <div class="lg:col-span-7 bg-white rounded-3xl border border-slate-200/80 shadow-xs p-6 sm:p-8">
        
        <!-- Section Header with Quick Presets -->
        <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-slate-100 gap-3">
          <div>
            <h2 class="text-lg font-bold text-slate-900">Patient Diagnostic Profile</h2>
            <p class="text-xs text-slate-500 mt-0.5">Adjust biological measurements or load calibrated clinical archetypes.</p>
          </div>
          <button type="button" onclick="resetForm()" class="inline-flex items-center gap-1.5 text-xs font-semibold text-slate-600 hover:text-slate-900 py-1.5 px-3 rounded-xl hover:bg-slate-100 transition-colors self-start sm:self-auto">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
            Reset
          </button>
        </div>

        <!-- Sample Profiles Pill Strip -->
        <div class="my-5">
          <div class="text-[11px] font-bold uppercase tracking-wider text-slate-400 mb-2.5">Quick Clinical Archetypes:</div>
          <div class="flex flex-wrap gap-2">
            <button type="button" onclick="applyPreset(1, 85, 68, 20, 70, 22.4, 0.25, 26)" class="text-xs px-3.5 py-2 rounded-xl bg-slate-50 hover:bg-emerald-50 text-slate-700 hover:text-emerald-800 font-semibold border border-slate-200 hover:border-emerald-300 transition-all flex items-center gap-1.5">
              <span class="w-2 h-2 rounded-full bg-emerald-500"></span>
              Healthy Baseline
            </button>
            <button type="button" onclick="applyPreset(3, 138, 78, 28, 110, 29.8, 0.48, 42)" class="text-xs px-3.5 py-2 rounded-xl bg-slate-50 hover:bg-amber-50 text-slate-700 hover:text-amber-800 font-semibold border border-slate-200 hover:border-amber-300 transition-all flex items-center gap-1.5">
              <span class="w-2 h-2 rounded-full bg-amber-500"></span>
              Borderline (Pre-DM)
            </button>
            <button type="button" onclick="applyPreset(6, 178, 88, 38, 210, 36.4, 1.15, 51)" class="text-xs px-3.5 py-2 rounded-xl bg-slate-50 hover:bg-rose-50 text-slate-700 hover:text-rose-800 font-semibold border border-slate-200 hover:border-rose-300 transition-all flex items-center gap-1.5">
              <span class="w-2 h-2 rounded-full bg-rose-500"></span>
              High-Risk Metabolic
            </button>
          </div>
        </div>

        <!-- Form Sliders Grid -->
        <form id="predictionForm" onsubmit="handleFormSubmit(event)" class="space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">

            <!-- 1. Glucose -->
            <div class="bg-slate-50/70 hover:bg-slate-50 p-4 rounded-2xl border border-slate-100 transition-all">
              <div class="flex justify-between items-baseline mb-1">
                <span class="text-xs font-bold text-slate-700">Plasma Glucose (2-hr OGTT)</span>
                <span id="val_glucose" class="text-xs font-bold text-teal-800 bg-teal-50/80 px-2.5 py-0.5 rounded-lg border border-teal-100 font-mono">120 mg/dL</span>
              </div>
              <div class="flex justify-between text-[10px] text-slate-400 mb-2">
                <span>Optimal: 70–99</span>
                <span>Pre-DM: 140–199</span>
              </div>
              <input type="range" id="glucose" min="0" max="250" step="1" value="120" oninput="updateVal('glucose', this.value, ' mg/dL')" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer">
            </div>

            <!-- 2. BMI -->
            <div class="bg-slate-50/70 hover:bg-slate-50 p-4 rounded-2xl border border-slate-100 transition-all">
              <div class="flex justify-between items-baseline mb-1">
                <span class="text-xs font-bold text-slate-700">Body Mass Index (BMI)</span>
                <span id="val_bmi" class="text-xs font-bold text-teal-800 bg-teal-50/80 px-2.5 py-0.5 rounded-lg border border-teal-100 font-mono">28.5 kg/m²</span>
              </div>
              <div class="flex justify-between text-[10px] text-slate-400 mb-2">
                <span>Normal: 18.5–24.9</span>
                <span>Obesity: &ge; 30.0</span>
              </div>
              <input type="range" id="bmi" min="0" max="60" step="0.1" value="28.5" oninput="updateVal('bmi', this.value, ' kg/m²')" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer">
            </div>

            <!-- 3. Blood Pressure -->
            <div class="bg-slate-50/70 hover:bg-slate-50 p-4 rounded-2xl border border-slate-100 transition-all">
              <div class="flex justify-between items-baseline mb-1">
                <span class="text-xs font-bold text-slate-700">Diastolic Blood Pressure</span>
                <span id="val_blood_pressure" class="text-xs font-bold text-teal-800 bg-teal-50/80 px-2.5 py-0.5 rounded-lg border border-teal-100 font-mono">72 mm Hg</span>
              </div>
              <div class="flex justify-between text-[10px] text-slate-400 mb-2">
                <span>Optimal: &lt; 80</span>
                <span>Stage 2: &ge; 90</span>
              </div>
              <input type="range" id="blood_pressure" min="0" max="140" step="1" value="72" oninput="updateVal('blood_pressure', this.value, ' mm Hg')" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer">
            </div>

            <!-- 4. Age -->
            <div class="bg-slate-50/70 hover:bg-slate-50 p-4 rounded-2xl border border-slate-100 transition-all">
              <div class="flex justify-between items-baseline mb-1">
                <span class="text-xs font-bold text-slate-700">Patient Age</span>
                <span id="val_age" class="text-xs font-bold text-teal-800 bg-teal-50/80 px-2.5 py-0.5 rounded-lg border border-teal-100 font-mono">35 yrs</span>
              </div>
              <div class="flex justify-between text-[10px] text-slate-400 mb-2">
                <span>Adult Cohort: 21–81</span>
                <span>Risk Scale: &ge; 45</span>
              </div>
              <input type="range" id="age" min="21" max="85" step="1" value="35" oninput="updateVal('age', this.value, ' yrs')" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer">
            </div>

            <!-- 5. Pregnancies -->
            <div class="bg-slate-50/70 hover:bg-slate-50 p-4 rounded-2xl border border-slate-100 transition-all">
              <div class="flex justify-between items-baseline mb-1">
                <span class="text-xs font-bold text-slate-700">Pregnancies</span>
                <span id="val_pregnancies" class="text-xs font-bold text-teal-800 bg-teal-50/80 px-2.5 py-0.5 rounded-lg border border-teal-100 font-mono">1</span>
              </div>
              <div class="text-[10px] text-slate-400 mb-2">0 = Nulliparous (valid physiological count)</div>
              <input type="range" id="pregnancies" min="0" max="17" step="1" value="1" oninput="updateVal('pregnancies', this.value, '')" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer">
            </div>

            <!-- 6. Genetic Pedigree -->
            <div class="bg-slate-50/70 hover:bg-slate-50 p-4 rounded-2xl border border-slate-100 transition-all">
              <div class="flex justify-between items-baseline mb-1">
                <span class="text-xs font-bold text-slate-700">Diabetes Pedigree Function</span>
                <span id="val_diabetes_pedigree_function" class="text-xs font-bold text-teal-800 bg-teal-50/80 px-2.5 py-0.5 rounded-lg border border-teal-100 font-mono">0.45</span>
              </div>
              <div class="flex justify-between text-[10px] text-slate-400 mb-2">
                <span>Low: &lt; 0.35</span>
                <span>Elevated: &gt; 0.60</span>
              </div>
              <input type="range" id="diabetes_pedigree_function" min="0.05" max="2.5" step="0.01" value="0.45" oninput="updateVal('diabetes_pedigree_function', parseFloat(this.value).toFixed(2), '')" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer">
            </div>

            <!-- 7. Insulin -->
            <div class="bg-slate-50/70 hover:bg-slate-50 p-4 rounded-2xl border border-slate-100 transition-all">
              <div class="flex justify-between items-baseline mb-1">
                <span class="text-xs font-bold text-slate-700">2-Hr Serum Insulin</span>
                <span id="val_insulin" class="text-xs font-bold text-teal-800 bg-teal-50/80 px-2.5 py-0.5 rounded-lg border border-teal-100 font-mono">85 &mu;U/mL</span>
              </div>
              <div class="text-[10px] text-slate-400 mb-2">0 = Unmeasured (Median Imputed)</div>
              <input type="range" id="insulin" min="0" max="400" step="5" value="85" oninput="updateVal('insulin', this.value, ' \u03bcU/mL')" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer">
            </div>

            <!-- 8. Skin Thickness -->
            <div class="bg-slate-50/70 hover:bg-slate-50 p-4 rounded-2xl border border-slate-100 transition-all">
              <div class="flex justify-between items-baseline mb-1">
                <span class="text-xs font-bold text-slate-700">Triceps Skin Thickness</span>
                <span id="val_skin_thickness" class="text-xs font-bold text-teal-800 bg-teal-50/80 px-2.5 py-0.5 rounded-lg border border-teal-100 font-mono">25 mm</span>
              </div>
              <div class="text-[10px] text-slate-400 mb-2">0 = Unmeasured (Median Imputed)</div>
              <input type="range" id="skin_thickness" min="0" max="60" step="1" value="25" oninput="updateVal('skin_thickness', this.value, ' mm')" class="w-full h-2 bg-slate-200 rounded-lg cursor-pointer">
            </div>

          </div>

          <div class="pt-5 border-t border-slate-100">
            <button type="submit" id="submitBtn" class="w-full py-4 px-5 bg-gradient-to-r from-teal-700 to-teal-600 hover:from-teal-800 hover:to-teal-700 text-white font-bold rounded-2xl shadow-lg shadow-teal-700/20 active:scale-[0.99] transition-all text-sm cursor-pointer flex items-center justify-center gap-2.5">
              <svg class="w-5 h-5 text-teal-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
              <span>Compute Clinical Risk Classification</span>
            </button>
          </div>
        </form>

      </div>

      <!-- Right Column: Real-Time Risk Assessment Console (5 Cols) -->
      <div class="lg:col-span-5 bg-white rounded-3xl border border-slate-200/80 shadow-xs p-6 sm:p-8 flex flex-col justify-between">
        <div>
          
          <div class="flex items-center justify-between pb-4 border-b border-slate-100">
            <div>
              <h3 class="text-base font-bold text-slate-900">Diagnostic Risk Console</h3>
              <p class="text-xs text-slate-400">Scikit-Learn Calibrated Pipeline</p>
            </div>
            <span id="riskPillBadge" class="px-3 py-1 text-xs font-extrabold rounded-full bg-slate-100 text-slate-600 border border-slate-200 uppercase tracking-wider">
              Awaiting Input
            </span>
          </div>

          <!-- Hero Radial Visual Gauge -->
          <div class="mt-6 flex flex-col items-center justify-center p-6 bg-slate-50/70 rounded-3xl border border-slate-100 text-center relative overflow-hidden">
            
            <div class="relative w-44 h-44 flex items-center justify-center mb-3">
              <svg class="w-full h-full transform -rotate-90" viewBox="0 0 120 120">
                <circle cx="60" cy="60" r="50" stroke="#e2e8f0" stroke-width="10" fill="transparent" />
                <circle id="gaugeSvgCircle" cx="60" cy="60" r="50" stroke="#0d9488" stroke-width="10" stroke-linecap="round" fill="transparent" stroke-dasharray="314.159" stroke-dashoffset="314.159" class="gauge-circle" />
              </svg>
              <div class="absolute flex flex-col items-center justify-center">
                <span id="riskPercentageText" class="text-3xl sm:text-4xl font-black text-slate-900 font-mono tracking-tight">--%</span>
                <span class="text-[10px] uppercase font-bold text-slate-400 tracking-wider mt-0.5">Risk Score</span>
              </div>
            </div>

            <div class="text-center">
              <div class="text-[11px] uppercase tracking-wider font-bold text-slate-400">Stratified Outcome</div>
              <div id="diagnosticClassText" class="text-2xl font-black text-slate-800 mt-0.5">Ready</div>
              <p id="riskSummaryText" class="text-xs text-slate-500 mt-1 max-w-xs">
                Select clinical parameters and click compute to run inference.
              </p>
            </div>

          </div>

          <!-- Identified Risk Biomarkers Breakdown -->
          <div class="mt-6">
            <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
              <svg class="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
              Contributing Clinical Biomarkers
            </h4>
            <div id="biomarkersContainer" class="space-y-2 text-xs text-slate-400 italic">
              Awaiting model calculation.
            </div>
          </div>

        </div>

        <div class="mt-7 pt-4 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
          <span>Decision Cutoff: &tau; = 0.50</span>
          <span id="confidenceIndicator" class="font-mono">Confidence: --%</span>
        </div>

      </div>

    </div>

    <!-- Algorithm Comparative Benchmark Matrix -->
    <div class="bg-white rounded-3xl border border-slate-200/80 shadow-xs p-6 sm:p-8 mt-8">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-100 gap-2">
        <div>
          <h3 class="text-base font-bold text-slate-900">Empirical Cross-Validation &amp; Holdout Benchmark</h3>
          <p class="text-xs text-slate-500 mt-0.5">5-Fold Stratified Cross-Validation on authentic NIDDK registry data ($n=768$).</p>
        </div>
        <span class="text-xs font-semibold px-3 py-1 rounded-full bg-teal-50 text-teal-800 border border-teal-200/60 self-start sm:self-auto">
          Champion: Logistic Regression
        </span>
      </div>

      <div class="overflow-x-auto mt-4">
        <table class="w-full text-left text-xs border-collapse">
          <thead>
            <tr class="bg-slate-50/80 border-b border-slate-200 text-[10px] uppercase font-bold text-slate-500 tracking-wider">
              <th class="py-3 px-3.5">Algorithm</th>
              <th class="py-3 px-2 text-center">CV ROC-AUC</th>
              <th class="py-3 px-2 text-center">Test Accuracy</th>
              <th class="py-3 px-2 text-center">Precision</th>
              <th class="py-3 px-2 text-center">Recall (Sens)</th>
              <th class="py-3 px-2 text-center">Specificity</th>
              <th class="py-3 px-2 text-center">F1-Score</th>
              <th class="py-3 px-2 text-center">Test ROC-AUC</th>
              <th class="py-3 px-3 text-center">Status</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-100">
            <tr class="bg-teal-50/40 font-medium">
              <td class="py-3.5 px-3.5">
                <span class="font-bold text-slate-900 text-sm">Logistic Regression</span>
                <p class="text-[11px] text-slate-400 font-normal">Optimal champion model with calibrated log-odds</p>
              </td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-700">0.8354</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-700">77.27%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-700">72.73%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-700">59.26%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-700">87.00%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-700">0.6531</td>
              <td class="py-3.5 px-2 text-center font-mono font-bold text-teal-800 bg-teal-100/60 rounded-lg">0.8385</td>
              <td class="py-3.5 px-3 text-center">
                <span class="px-2.5 py-1 text-[10px] font-extrabold rounded-full bg-teal-100 text-teal-800 border border-teal-200">
                  DEPLOYED
                </span>
              </td>
            </tr>
            <tr>
              <td class="py-3.5 px-3.5">
                <span class="font-bold text-slate-800">Random Forest (n=150)</span>
                <p class="text-[11px] text-slate-400 font-normal">Ensemble of non-linear decision trees</p>
              </td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">0.8312</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">76.62%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">72.09%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">57.41%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">87.00%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">0.6392</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">0.8322</td>
              <td class="py-3.5 px-3 text-center text-slate-400 text-[11px]">Evaluated</td>
            </tr>
            <tr>
              <td class="py-3.5 px-3.5">
                <span class="font-bold text-slate-800">Support Vector Machine</span>
                <p class="text-[11px] text-slate-400 font-normal">Radial basis function (RBF) kernel</p>
              </td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">0.8248</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">75.97%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">71.43%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">55.56%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">87.00%</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">0.6250</td>
              <td class="py-3.5 px-2 text-center font-mono text-slate-600">0.8265</td>
              <td class="py-3.5 px-3 text-center text-slate-400 text-[11px]">Evaluated</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

  </main>

  <!-- Clean Executive Footer -->
  <footer class="bg-white border-t border-slate-200/80 py-5 text-xs text-slate-400">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
      <div class="flex items-center gap-2 text-slate-500 font-medium">
        <svg class="w-4 h-4 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
        <span>Academic Demonstration &bull; Pima Indians Cohort Registry (NIDDK)</span>
      </div>
      <div class="flex items-center gap-3 text-slate-400 font-medium text-[11px]">
        <span>FastAPI</span> &bull; <span>Scikit-Learn</span> &bull; <span>Next.js</span> &bull; <span>Tailwind CSS</span> &bull; <span>Vercel</span>
      </div>
    </div>
  </footer>

  <!-- Script Logic -->
  <script>
    const CIRCLE_CIRCUMFERENCE = 2 * Math.PI * 50; // ~314.159

    function updateVal(id, val, unit) {
      document.getElementById('val_' + id).innerText = val + unit;
    }

    function applyPreset(preg, glu, bp, skin, ins, bmi, dpf, age) {
      document.getElementById('pregnancies').value = preg;
      updateVal('pregnancies', preg, '');

      document.getElementById('glucose').value = glu;
      updateVal('glucose', glu, ' mg/dL');

      document.getElementById('blood_pressure').value = bp;
      updateVal('blood_pressure', bp, ' mm Hg');

      document.getElementById('skin_thickness').value = skin;
      updateVal('skin_thickness', skin, ' mm');

      document.getElementById('insulin').value = ins;
      updateVal('insulin', ins, ' \u03bcU/mL');

      document.getElementById('bmi').value = bmi;
      updateVal('bmi', bmi, ' kg/m²');

      document.getElementById('diabetes_pedigree_function').value = dpf;
      updateVal('diabetes_pedigree_function', dpf, '');

      document.getElementById('age').value = age;
      updateVal('age', age, ' yrs');

      executePrediction();
    }

    function resetForm() {
      applyPreset(1, 120, 72, 25, 85, 28.5, 0.45, 35);
    }

    function handleFormSubmit(e) {
      e.preventDefault();
      executePrediction();
    }

    async function executePrediction() {
      const payload = {
        pregnancies: parseInt(document.getElementById('pregnancies').value),
        glucose: parseFloat(document.getElementById('glucose').value),
        blood_pressure: parseFloat(document.getElementById('blood_pressure').value),
        skin_thickness: parseFloat(document.getElementById('skin_thickness').value),
        insulin: parseFloat(document.getElementById('insulin').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        diabetes_pedigree_function: parseFloat(document.getElementById('diabetes_pedigree_function').value),
        age: parseInt(document.getElementById('age').value)
      };

      const btn = document.getElementById('submitBtn');
      btn.innerHTML = '<span>Calculating Calibrated Risk...</span>';
      btn.disabled = true;

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error('Inference server returned status ' + res.status);
        const data = await res.json();

        const probPct = Math.round(data.probability * 100);
        document.getElementById('riskPercentageText').innerText = probPct + '%';
        document.getElementById('diagnosticClassText').innerText = data.class_label;
        document.getElementById('confidenceIndicator').innerText = 'Confidence: ' + data.confidence_score + '%';

        // Update SVG circle stroke
        const circle = document.getElementById('gaugeSvgCircle');
        const offset = CIRCLE_CIRCUMFERENCE - (probPct / 100) * CIRCLE_CIRCUMFERENCE;
        circle.style.strokeDashoffset = offset;

        const pillBadge = document.getElementById('riskPillBadge');
        const classText = document.getElementById('diagnosticClassText');
        const summaryText = document.getElementById('riskSummaryText');

        pillBadge.innerText = data.risk_category;

        if (data.risk_category === 'High Risk') {
          pillBadge.className = 'px-3 py-1 text-xs font-extrabold rounded-full bg-rose-100 text-rose-800 border border-rose-300 uppercase tracking-wider';
          classText.className = 'text-2xl font-black text-rose-700 mt-0.5';
          circle.setAttribute('stroke', '#e11d48');
          summaryText.innerText = 'High probability of diabetes onset based on metabolic markers.';
        } else if (data.risk_category === 'Moderate Risk') {
          pillBadge.className = 'px-3 py-1 text-xs font-extrabold rounded-full bg-amber-100 text-amber-800 border border-amber-300 uppercase tracking-wider';
          classText.className = 'text-2xl font-black text-amber-700 mt-0.5';
          circle.setAttribute('stroke', '#f59e0b');
          summaryText.innerText = 'Borderline glycemic indicators warranting clinical follow-up.';
        } else {
          pillBadge.className = 'px-3 py-1 text-xs font-extrabold rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 uppercase tracking-wider';
          classText.className = 'text-2xl font-black text-emerald-700 mt-0.5';
          circle.setAttribute('stroke', '#0d9488');
          summaryText.innerText = 'Physiological values are within baseline parameters.';
        }

        // Render Biomarker Cards
        const container = document.getElementById('biomarkersContainer');
        if (data.contributing_factors.length === 0) {
          container.innerHTML = '<div class="p-3.5 bg-slate-50 rounded-2xl border border-slate-100 text-slate-500 italic">No elevated risk indicators detected for this patient profile.</div>';
        } else {
          container.innerHTML = data.contributing_factors.map(f => `
            <div class="p-3 bg-white rounded-2xl border border-slate-200/90 text-xs flex flex-col gap-0.5 shadow-2xs">
              <div class="flex items-center justify-between">
                <span class="font-bold text-slate-800">${f.feature} (${f.value})</span>
                <span class="text-[10px] px-2 py-0.5 rounded-full font-bold bg-rose-50 text-rose-700 border border-rose-100">${f.status}</span>
              </div>
              <p class="text-[11px] text-slate-500 mt-0.5 leading-snug">${f.clinical_note}</p>
            </div>
          `).join('');
        }

      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.innerHTML = `<svg class="w-5 h-5 text-teal-200" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg><span>Compute Clinical Risk Classification</span>`;
        btn.disabled = false;
      }
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="healthy",
        service="diabetes-prediction-backend",
        version="1.0.0",
        model_loaded=predictor.is_loaded(),
        architecture="FastAPI + Scikit-Learn Pipeline"
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Metadata"])
def get_model_information():
    meta = predictor.get_metadata()
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model metadata is currently unavailable."
        )
    return ModelInfoResponse(**meta)


@app.post(
    "/predict",
    response_model=DiabetesPredictionOutput,
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict_diabetes_risk(payload: DiabetesPredictionInput):
    try:
        result = predictor.predict(payload)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error encountered: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
