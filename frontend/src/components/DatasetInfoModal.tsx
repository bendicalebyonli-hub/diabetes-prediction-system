"use client";

import React from "react";
import { X, BookOpen } from "lucide-react";

interface DatasetInfoModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const DatasetInfoModal: React.FC<DatasetInfoModalProps> = ({
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto shadow-2xl border border-slate-200 p-6 sm:p-7 relative text-slate-800">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-5 right-5 p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-2.5 pb-4 border-b border-slate-100">
          <BookOpen className="w-6 h-6 text-teal-600" />
          <div>
            <h3 className="text-lg font-bold text-slate-900">
              Pima Indians Diabetes Dataset Provenance
            </h3>
            <p className="text-xs text-slate-500">
              Historical Context, Original Study &amp; Clinical Limitations
            </p>
          </div>
        </div>

        {/* Content Body */}
        <div className="mt-5 space-y-4 text-xs sm:text-sm text-slate-600 leading-relaxed">
          <section>
            <h4 className="font-bold text-slate-900 mb-1">1. Historical Origin</h4>
            <p>
              The dataset originated from the <strong>National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK)</strong> and was first analyzed in 1990 by <em>Smith et al.</em> using the ADAP associative neural learning architecture to forecast diabetes mellitus onset.
            </p>
          </section>

          <section>
            <h4 className="font-bold text-slate-900 mb-1">2. Cohort Inclusion Criteria</h4>
            <ul className="list-disc list-inside space-y-1 text-slate-700">
              <li>Demographic: Females of Pima Indian ancestry</li>
              <li>Age Threshold: At least 21 years of age</li>
              <li>Location: Gila River Indian Community near Phoenix, Arizona</li>
              <li>Diagnostic Standard: World Health Organization (WHO) 2-hour post-load plasma glucose criteria (&ge; 200 mg/dL)</li>
            </ul>
          </section>

          <section>
            <h4 className="font-bold text-slate-900 mb-1">3. The Physiological Zero Issue</h4>
            <p>
              Several biological indicators were cataloged with numerical values of <code>0</code>:
            </p>
            <ul className="list-disc list-inside space-y-1 mt-1 text-slate-700">
              <li><strong>Glucose (5 records with 0):</strong> Human life is incompatible with 0 mg/dL glucose (hypoglycemic shock occurs below 40 mg/dL).</li>
              <li><strong>Blood Pressure (35 records with 0):</strong> True diastolic 0 indicates cardiac arrest or missing test.</li>
              <li><strong>Skin Thickness (227 records with 0):</strong> Triceps skin fold thickness is never 0 mm.</li>
              <li><strong>Insulin (374 records with 0):</strong> 48.7% of subjects had unrecorded 2-hr serum insulin levels.</li>
              <li><strong>BMI (11 records with 0):</strong> Physically impossible zero measurement.</li>
            </ul>
            <p className="mt-2 text-teal-800 bg-teal-50 p-2.5 rounded-lg border border-teal-100 text-xs">
              <strong>Engineering Solution:</strong> Our ML pipeline masks these biologically impossible zeros to <code>NaN</code> and applies robust median imputation learned solely from the training partition.
            </p>
          </section>

          <section>
            <h4 className="font-bold text-slate-900 mb-1">4. Ethical &amp; Clinical Disclaimers</h4>
            <div className="p-3 bg-amber-50 rounded-xl border border-amber-200 text-xs text-amber-900 space-y-1">
              <p>
                <strong>Academic Scope:</strong> This machine learning prototype is engineered for higher education coursework, algorithmic benchmarking, and software architecture demonstration.
              </p>
              <p>
                It is <strong>not clinically certified</strong>, has not undergone clinical trials, and must never be utilized to diagnose, monitor, or treat human patients.
              </p>
            </div>
          </section>
        </div>

        {/* Footer */}
        <div className="mt-6 pt-4 border-t border-slate-100 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white font-semibold rounded-xl text-xs transition-colors"
          >
            Close Overview
          </button>
        </div>
      </div>
    </div>
  );
};
