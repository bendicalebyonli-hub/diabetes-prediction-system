"use client";

import React, { useState } from "react";
import { ClinicalDisclaimer } from "@/components/ClinicalDisclaimer";
import { PredictionForm } from "@/components/PredictionForm";
import { ResultCard } from "@/components/ResultCard";
import { ModelMetricsCard } from "@/components/ModelMetricsCard";
import { DatasetInfoModal } from "@/components/DatasetInfoModal";
import {
  DiabetesPredictionInput,
  DiabetesPredictionOutput,
} from "@/types/prediction";
import { HeartPulse, BookOpen, ShieldCheck, Activity, Award, Users, Zap } from "lucide-react";

export default function Home() {
  const [result, setResult] = useState<DiabetesPredictionOutput | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false);
  const [apiSource, setApiSource] = useState<string>("FastAPI Online");

  const handlePredict = async (formData: DiabetesPredictionInput) => {
    setIsLoading(true);

    const backendUrl =
      process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3000);

    try {
      // 1. Attempt connection to FastAPI backend service
      const res = await fetch(`${backendUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data: DiabetesPredictionOutput = await res.json();
        setResult(data);
        setApiSource("FastAPI Microservice (Port 8000)");
        setIsLoading(false);
        return;
      }
    } catch {
      clearTimeout(timeoutId);
      // Backend unreachable or timed out; fall back to Next.js serverless route
    }

    try {
      // 2. Seamless serverless fallback for local & Vercel deployment
      const fallbackRes = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });

      if (!fallbackRes.ok) {
        throw new Error("Prediction request failed");
      }

      const data: DiabetesPredictionOutput = await fallbackRes.json();
      setResult(data);
      setApiSource("Next.js Serverless Edge Pipeline (Vercel Mode)");
    } catch (err: any) {
      alert("Error calculating prediction: " + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col justify-between bg-[#f8fafc]">
      {/* Executive Header */}
      <header className="bg-white/95 backdrop-blur-md border-b border-slate-200 sticky top-0 z-30 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 py-3.5 flex items-center justify-between">
          <div className="flex items-center gap-3.5">
            <div className="w-11 h-11 rounded-xl bg-gradient-to-tr from-teal-700 via-teal-600 to-emerald-500 text-white flex items-center justify-center shadow-md shadow-teal-700/20">
              <HeartPulse className="w-6 h-6" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base sm:text-lg font-extrabold text-slate-900 tracking-tight">
                  CLINICAL<span className="text-teal-600">AI</span> INTELLIGENCE
                </h1>
                <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-slate-100 text-slate-600 border border-slate-200">
                  v1.0 Production
                </span>
              </div>
              <p className="text-xs text-slate-500 font-medium hidden sm:block">
                Predictive Diabetes Risk Classifier &bull; 5-Fold Cross-Validated Pipeline
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsModalOpen(true)}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-xl shadow-2xs transition-all"
            >
              <BookOpen className="w-3.5 h-3.5 text-teal-600" />
              <span>Dataset Provenance</span>
            </button>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-semibold">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
              <span>{apiSource}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-7 w-full flex-1">
        
        {/* KPI Metric Strip */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-7">
          <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
            <div className="flex items-center justify-between text-slate-400 mb-1">
              <span className="text-[11px] font-bold uppercase tracking-wider">Model Discrimination</span>
              <Award className="w-4 h-4 text-teal-600" />
            </div>
            <div className="text-2xl font-extrabold text-slate-900 font-mono">
              0.8385 <span className="text-xs text-emerald-600 font-bold">AUC</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-1">5-Fold Stratified Holdout Test</div>
          </div>

          <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
            <div className="flex items-center justify-between text-slate-400 mb-1">
              <span className="text-[11px] font-bold uppercase tracking-wider">Diagnostic Accuracy</span>
              <Activity className="w-4 h-4 text-emerald-600" />
            </div>
            <div className="text-2xl font-extrabold text-slate-900 font-mono">
              77.27%
            </div>
            <div className="text-[11px] text-slate-500 mt-1">Zero-Leakage Pipeline</div>
          </div>

          <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
            <div className="flex items-center justify-between text-slate-400 mb-1">
              <span className="text-[11px] font-bold uppercase tracking-wider">Clinical Cohort</span>
              <Users className="w-4 h-4 text-blue-600" />
            </div>
            <div className="text-2xl font-extrabold text-slate-900 font-mono">
              768 <span className="text-xs text-slate-400 font-normal">Patients</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-1">Authentic NIDDK Dataset</div>
          </div>

          <div className="bg-white p-4 rounded-2xl border border-slate-200/80 shadow-2xs">
            <div className="flex items-center justify-between text-slate-400 mb-1">
              <span className="text-[11px] font-bold uppercase tracking-wider">Inference Latency</span>
              <Zap className="w-4 h-4 text-amber-500" />
            </div>
            <div className="text-2xl font-extrabold text-teal-700 font-mono">
              &lt; 10 <span className="text-xs font-normal">ms</span>
            </div>
            <div className="text-[11px] text-slate-500 mt-1">Real-Time Scoring Engine</div>
          </div>
        </div>

        <ClinicalDisclaimer />

        {/* 2-Column Responsive Diagnostic Workbench */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-7 items-start">
          {/* Left Column: Input Form (7 cols on desktop) */}
          <div className="lg:col-span-7">
            <PredictionForm onSubmit={handlePredict} isLoading={isLoading} />
          </div>

          {/* Right Column: Dynamic Result & Factor Breakdown (5 cols on desktop) */}
          <div className="lg:col-span-5 lg:sticky lg:top-24">
            <ResultCard result={result} isLoading={isLoading} />
          </div>
        </div>

        {/* Benchmark Evaluation, Confusion Matrix & Preprocessing Section */}
        <ModelMetricsCard />
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12 text-xs text-slate-400">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2 text-slate-500 font-medium">
            <ShieldCheck className="w-4 h-4 text-teal-600" />
            <span>
              Academic Machine Learning Project &bull; Pima Indians Cohort Registry (NIDDK)
            </span>
          </div>
          <div className="flex flex-wrap items-center gap-3 text-slate-400 font-medium text-[11px]">
            <span>FastAPI</span>
            <span>&bull;</span>
            <span>Scikit-Learn</span>
            <span>&bull;</span>
            <span>Next.js 14</span>
            <span>&bull;</span>
            <span>Tailwind CSS</span>
            <span>&bull;</span>
            <span>Vercel</span>
          </div>
        </div>
      </footer>

      {/* Provenance & Limitations Modal */}
      <DatasetInfoModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  );
}
