import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Pima Indians Diabetes Prediction System | Academic ML System",
  description:
    "Design and Implementation of a Machine Learning-Based Diabetes Prediction System Using the Pima Indians Diabetes Dataset. Built with Scikit-learn, FastAPI, Next.js, and Tailwind CSS.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased bg-slate-50 text-slate-900 selection:bg-teal-100 selection:text-teal-900">
        {children}
      </body>
    </html>
  );
}
