"use client";

import React from "react";
import { DiabetesPredictionOutput } from "@/types/prediction";
import { ShieldCheck, AlertCircle, CheckCircle, Info, HeartPulse } from "lucide-react";

interface ResultCardProps {
  result: DiabetesPredictionOutput | null;
  isLoading: boolean;
}

export const ResultCard: React.FC<ResultCardProps> = ({ result, isLoading }) => {
  if (isLoading) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-8 text-center animate-pulse flex flex-col items-center justify-center min-h-[380px]">
        <div className="w-16 h-16 bg-slate-100 rounded-full mb-4 animate-spin border-4 border-teal-500 border-t-transparent" />
        <h3 className="text-base font-semibold text-slate-700">
          Executing Scikit-Learn Inference Pipeline
        </h3>
        <p className="text-xs text-slate-400 mt-1 max-w-xs">
          Imputing biological zero values via median transformer and applying standardized weights...
        </p>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-8 text-center flex flex-col items-center justify-center min-h-[380px]">
        <div className="w-14 h-14 bg-teal-50 text-teal-600 rounded-2xl flex items-center justify-center mb-3">
          <HeartPulse className="w-7 h-7" />
        </div>
        <h3 className="text-base font-bold text-slate-800">
          Awaiting Diagnostic Input
        </h3>
        <p className="text-xs text-slate-500 mt-1.5 max-w-sm leading-relaxed">
          Configure the patient clinical variables on the left panel or click an academic preset, then press <strong>Compute</strong> to evaluate diabetes onset risk.
        </p>
      </div>
    );
  }

  const isDiabetic = result.prediction === 1;
  const probPercent = Math.round(result.probability * 100);

  // Dynamic color palette based on risk category
  const theme = {
    "Low Risk": {
      bg: "bg-emerald-50",
      border: "border-emerald-200",
      text: "text-emerald-900",
      badgeBg: "bg-emerald-100 text-emerald-800",
      barColor: "bg-emerald-500",
      ringColor: "stroke-emerald-500",
      icon: <CheckCircle className="w-6 h-6 text-emerald-600" />,
    },
    "Moderate Risk": {
      bg: "bg-amber-50",
      border: "border-amber-200",
      text: "text-amber-900",
      badgeBg: "bg-amber-100 text-amber-800",
      barColor: "bg-amber-500",
      ringColor: "stroke-amber-500",
      icon: <AlertCircle className="w-6 h-6 text-amber-600" />,
    },
    "High Risk": {
      bg: "bg-rose-50",
      border: "border-rose-200",
      text: "text-rose-900",
      badgeBg: "bg-rose-100 text-rose-800",
      barColor: "bg-rose-500",
      ringColor: "stroke-rose-500",
      icon: <AlertCircle className="w-6 h-6 text-rose-600" />,
    },
  }[result.risk_category];

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 sm:p-7 flex flex-col justify-between transition-all">
      <div>
        {/* Header Badge */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-teal-600" />
            <h2 className="text-base font-bold text-slate-900">
              Model Diagnostic Assessment
            </h2>
          </div>
          <span className="text-[11px] font-semibold text-slate-400">
            Pipeline v{result.model_version}
          </span>
        </div>

        {/* Prediction Outcome Card */}
        <div className={`mt-5 p-5 rounded-2xl border ${theme.bg} ${theme.border} transition-all`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {theme.icon}
              <div>
                <span className="text-[11px] uppercase tracking-wider font-bold text-slate-500 block">
                  Predicted Classification
                </span>
                <span className={`text-xl sm:text-2xl font-black ${theme.text}`}>
                  {result.class_label}
                </span>
              </div>
            </div>
            <span className={`px-3 py-1 text-xs font-bold rounded-full ${theme.badgeBg}`}>
              {result.risk_category}
            </span>
          </div>

          {/* Probability Gauge Bar */}
          <div className="mt-5">
            <div className="flex justify-between items-center text-xs font-medium mb-1.5">
              <span className="text-slate-600">Calculated Onset Probability:</span>
              <span className="font-bold text-slate-900 text-sm">{probPercent}%</span>
            </div>
            <div className="w-full h-3 bg-slate-200/80 rounded-full overflow-hidden p-0.5">
              <div
                className={`h-full rounded-full transition-all duration-700 ${theme.barColor}`}
                style={{ width: `${Math.min(100, Math.max(5, probPercent))}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-400 mt-1">
              <span>0% (Low)</span>
              <span>50% (Threshold)</span>
              <span>100% (High)</span>
            </div>
          </div>
        </div>

        {/* Contributing Risk Indicators */}
        <div className="mt-6">
          <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 mb-3 flex items-center gap-1.5">
            <Info className="w-4 h-4 text-slate-400" />
            Physiological Risk Factor Analysis
          </h4>

          {result.contributing_factors.length === 0 ? (
            <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-100 text-xs text-slate-500 italic">
              No high-risk physiological anomalies detected for this profile.
            </div>
          ) : (
            <div className="space-y-2">
              {result.contributing_factors.map((factor, idx) => (
                <div
                  key={idx}
                  className="p-3 bg-slate-50 rounded-xl border border-slate-100 text-xs flex flex-col gap-0.5"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-slate-800">
                      {factor.feature} ({factor.value})
                    </span>
                    <span className="text-[10px] px-2 py-0.5 rounded font-medium bg-rose-50 text-rose-700 border border-rose-100">
                      {factor.status}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-500 mt-0.5 leading-snug">
                    {factor.clinical_note}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Safety Notice Footer */}
      <div className="mt-6 pt-4 border-t border-slate-100 text-[11px] text-slate-400 flex items-center justify-between">
        <span>Decision Threshold: &tau; = 0.50</span>
        <span>Confidence: {result.confidence_score}%</span>
      </div>
    </div>
  );
};
