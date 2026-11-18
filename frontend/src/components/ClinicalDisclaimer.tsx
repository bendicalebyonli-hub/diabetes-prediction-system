"use client";

import React from "react";
import { ShieldCheck } from "lucide-react";

export const ClinicalDisclaimer: React.FC = () => {
  return (
    <div className="flex items-center justify-between px-4 py-2.5 bg-slate-100/70 border border-slate-200/80 rounded-xl mb-6 text-xs text-slate-500">
      <div className="flex items-center gap-2">
        <ShieldCheck className="w-4 h-4 text-teal-600 shrink-0" />
        <span>
          <strong>Clinical Decision Support Engine:</strong> Powered by 5-Fold Cross-Validated Scikit-Learn Pipeline &bull; NIDDK Cohort Registry
        </span>
      </div>
      <span className="hidden sm:inline-block font-mono text-[10px] uppercase font-bold text-slate-400">
        Academic System v1.0
      </span>
    </div>
  );
};
