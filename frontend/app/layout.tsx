import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "XAI Network Intrusion Detection",
  description:
    "Detect suspicious network traffic, understand why the model flagged it, and compare full and lightweight detection models.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen grid-bg">
        <Navbar />
        <main className="max-w-6xl mx-auto px-4 sm:px-6 py-10">{children}</main>
        <footer className="max-w-6xl mx-auto px-4 sm:px-6 py-8 text-xs text-muted border-t border-border mt-12">
          IEEE Research Project · UNSW-NB15 · XGBoost + SHAP · Not a production IDS.
        </footer>
      </body>
    </html>
  );
}
