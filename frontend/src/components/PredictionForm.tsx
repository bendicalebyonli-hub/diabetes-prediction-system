"use client";

import React, { useState } from "react";
import { DiabetesPredictionInput } from "@/types/prediction";
import { Activity, RotateCcw, Sparkles } from "lucide-react";

interface PredictionFormProps {
  onSubmit: (data: DiabetesPredictionInput) => void;
  isLoading: boolean;
}

const DEFAULT_VALUES: DiabetesPredictionInput = {
  pregnancies: 1,
  glucose: 120.0,
  blood_pressure: 72.0,
  skin_thickness: 25.0,
  insulin: 85.0,
  bmi: 28.5,
  diabetes_pedigree_function: 0.45,
  age: 35,
};

const PRESETS = [
  {
    name: "Healthy Baseline (Low Risk)",
    data: {
      pregnancies: 1,
      glucose: 85.0,
      blood_pressure: 68.0,
      skin_thickness: 20.0,
      insulin: 70.0,
      bmi: 22.4,
      diabetes_pedigree_function: 0.25,
      age: 26,
    },
  },
  {
    name: "Borderline / Pre-Diabetes",
    data: {
      pregnancies: 3,
      glucose: 135.0,
      blood_pressure: 78.0,
      skin_thickness: 28.0,
      insulin: 110.0,
      bmi: 29.8,
      diabetes_pedigree_function: 0.48,
      age: 42,
    },
  },
  {
    name: "High-Risk Metabolic Profile",
    data: {
      pregnancies: 6,
      glucose: 175.0,
      blood_pressure: 88.0,
      skin_thickness: 38.0,
      insulin: 210.0,
      bmi: 36.4,
      diabetes_pedigree_function: 1.15,
      age: 51,
    },
  },
];

export const PredictionForm: React.FC<PredictionFormProps> = ({
  onSubmit,
  isLoading,
}) => {
  const [formData, setFormData] = useState<DiabetesPredictionInput>(DEFAULT_VALUES);

  const handleChange = (field: keyof DiabetesPredictionInput, value: number) => {
    setFormData((prev) => ({
      ...prev,
      [field]: isNaN(value) ? 0 : value,
    }));
  };

  const handleApplyPreset = (presetData: DiabetesPredictionInput) => {
    setFormData(presetData);
  };

  const handleReset = () => {
    setFormData(DEFAULT_VALUES);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="bg-white rounded-2xl border border-slate-200/80 shadow-sm p-6 sm:p-7 transition-all"
    >
      {/* Header & Quick Profiles */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-5 border-b border-slate-100 gap-3">
        <div>
          <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <Activity className="w-5 h-5 text-teal-600" />
            Patient Clinical Measurements
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Enter 8 diagnostic parameters derived from the oral glucose tolerance study.
          </p>
        </div>
        <button
          type="button"
          onClick={handleReset}
          className="inline-flex items-center gap-1.5 text-xs text-slate-500 hover:text-slate-800 self-start sm:self-auto py-1 px-2.5 rounded-lg hover:bg-slate-100 transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Reset Form
        </button>
      </div>

      {/* Preset Buttons */}
      <div className="my-5">
        <label className="text-xs font-semibold uppercase tracking-wider text-slate-400 block mb-2">
          Load Academic Sample Profile:
        </label>
        <div className="flex flex-wrap gap-2">
          {PRESETS.map((p, idx) => (
            <button
              key={idx}
              type="button"
              onClick={() => handleApplyPreset(p.data)}
              className="text-xs px-3 py-1.5 rounded-lg bg-teal-50 hover:bg-teal-100 text-teal-800 font-medium border border-teal-200/60 transition-colors flex items-center gap-1.5"
            >
              <Sparkles className="w-3 h-3 text-teal-600" />
              {p.name}
            </button>
          ))}
        </div>
      </div>

      {/* Input Fields Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 text-slate-800">
        {/* 1. Glucose */}
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <div className="flex justify-between items-baseline mb-1">
            <label className="text-xs font-semibold text-slate-700">
              Plasma Glucose (2-hr OGTT)
            </label>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-100">
              {formData.glucose} mg/dL
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mb-2.5 flex justify-between">
            <span>Normal: 70–99 mg/dL</span>
            <span>Pre-DM: 140–199</span>
          </div>
          <input
            type="range"
            min="0"
            max="250"
            step="1"
            value={formData.glucose}
            onChange={(e) => handleChange("glucose", parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg cursor-pointer"
          />
        </div>

        {/* 2. BMI */}
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <div className="flex justify-between items-baseline mb-1">
            <label className="text-xs font-semibold text-slate-700">
              Body Mass Index (BMI)
            </label>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-100">
              {formData.bmi.toFixed(1)} kg/m²
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mb-2.5 flex justify-between">
            <span>Normal: 18.5–24.9</span>
            <span>Obese: $\ge$ 30.0</span>
          </div>
          <input
            type="range"
            min="0"
            max="60"
            step="0.1"
            value={formData.bmi}
            onChange={(e) => handleChange("bmi", parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg cursor-pointer"
          />
        </div>

        {/* 3. Blood Pressure */}
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <div className="flex justify-between items-baseline mb-1">
            <label className="text-xs font-semibold text-slate-700">
              Diastolic Blood Pressure
            </label>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-100">
              {formData.blood_pressure} mm Hg
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mb-2.5 flex justify-between">
            <span>Optimal: &lt; 80 mm Hg</span>
            <span>Stage 2: $\ge$ 90</span>
          </div>
          <input
            type="range"
            min="0"
            max="140"
            step="1"
            value={formData.blood_pressure}
            onChange={(e) => handleChange("blood_pressure", parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg cursor-pointer"
          />
        </div>

        {/* 4. Age */}
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <div className="flex justify-between items-baseline mb-1">
            <label className="text-xs font-semibold text-slate-700">
              Patient Age
            </label>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-100">
              {formData.age} years
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mb-2.5 flex justify-between">
            <span>Cohort Range: 21–81 yrs</span>
            <span>Adult Females</span>
          </div>
          <input
            type="range"
            min="21"
            max="85"
            step="1"
            value={formData.age}
            onChange={(e) => handleChange("age", parseInt(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg cursor-pointer"
          />
        </div>

        {/* 5. Pregnancies */}
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <div className="flex justify-between items-baseline mb-1">
            <label className="text-xs font-semibold text-slate-700">
              Number of Pregnancies
            </label>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-100">
              {formData.pregnancies}
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mb-2.5">
            0 = Nulliparous (valid zero measurement)
          </div>
          <input
            type="range"
            min="0"
            max="17"
            step="1"
            value={formData.pregnancies}
            onChange={(e) => handleChange("pregnancies", parseInt(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg cursor-pointer"
          />
        </div>

        {/* 6. Diabetes Pedigree Function */}
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <div className="flex justify-between items-baseline mb-1">
            <label className="text-xs font-semibold text-slate-700">
              Diabetes Pedigree Function
            </label>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-100">
              {formData.diabetes_pedigree_function.toFixed(2)}
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mb-2.5 flex justify-between">
            <span>Low: &lt; 0.35</span>
            <span>High: &gt; 0.60</span>
          </div>
          <input
            type="range"
            min="0.05"
            max="2.5"
            step="0.01"
            value={formData.diabetes_pedigree_function}
            onChange={(e) => handleChange("diabetes_pedigree_function", parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg cursor-pointer"
          />
        </div>

        {/* 7. Serum Insulin */}
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <div className="flex justify-between items-baseline mb-1">
            <label className="text-xs font-semibold text-slate-700">
              2-Hr Serum Insulin
            </label>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-100">
              {formData.insulin} $\mu$U/mL
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mb-2.5 flex justify-between">
            <span>Normal: 16–166</span>
            <span>0 = Unmeasured (Imputed)</span>
          </div>
          <input
            type="range"
            min="0"
            max="400"
            step="5"
            value={formData.insulin}
            onChange={(e) => handleChange("insulin", parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg cursor-pointer"
          />
        </div>

        {/* 8. Skin Thickness */}
        <div className="bg-slate-50/70 p-4 rounded-xl border border-slate-100">
          <div className="flex justify-between items-baseline mb-1">
            <label className="text-xs font-semibold text-slate-700">
              Triceps Skin Fold Thickness
            </label>
            <span className="text-xs font-semibold text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-100">
              {formData.skin_thickness} mm
            </span>
          </div>
          <div className="text-[11px] text-slate-400 mb-2.5 flex justify-between">
            <span>Normal: 10–30 mm</span>
            <span>0 = Unmeasured (Imputed)</span>
          </div>
          <input
            type="range"
            min="0"
            max="60"
            step="1"
            value={formData.skin_thickness}
            onChange={(e) => handleChange("skin_thickness", parseFloat(e.target.value))}
            className="w-full h-2 bg-slate-200 rounded-lg cursor-pointer"
          />
        </div>
      </div>

      {/* Submit Button */}
      <div className="mt-7 pt-5 border-t border-slate-100">
        <button
          type="submit"
          disabled={isLoading}
          className="w-full py-3.5 px-4 bg-teal-600 hover:bg-teal-700 disabled:bg-teal-300 text-white font-semibold rounded-xl shadow-md shadow-teal-700/10 hover:shadow-lg hover:shadow-teal-700/20 active:scale-[0.99] transition-all flex items-center justify-center gap-2 text-sm sm:text-base cursor-pointer"
        >
          {isLoading ? (
            <>
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              Calculating Calibrated Risk Score...
            </>
          ) : (
            <>
              <Activity className="w-5 h-5" />
              Compute Diabetes Classification & Probability
            </>
          )}
        </button>
      </div>
    </form>
  );
};
