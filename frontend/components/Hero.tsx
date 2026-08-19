import Link from "next/link";

export default function Hero() {
  return (
    <section className="text-center pt-8 pb-12">
      <div className="inline-block px-3 py-1 mb-4 rounded-full text-xs font-mono border border-border text-muted">
        IEEE Research · UNSW-NB15 · XGBoost + SHAP
      </div>
      <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
        Explainable AI <br />
        <span className="gradient-text">Network Intrusion Detection</span>
      </h1>
      <p className="mt-5 text-muted max-w-2xl mx-auto text-sm sm:text-base">
        Detect suspicious network traffic, understand why the model flagged it,
        and compare full and lightweight detection models.
      </p>
      <div className="mt-7 flex flex-col sm:flex-row gap-3 justify-center">
        <Link
          href="/analyzer"
          className="px-5 py-2.5 rounded-lg bg-accent text-bg font-semibold text-sm hover:opacity-90"
        >
          Analyze Traffic →
        </Link>
        <Link
          href="/research"
          className="px-5 py-2.5 rounded-lg border border-border text-sm hover:border-accent"
        >
          Explore Research
        </Link>
      </div>

      <div className="mt-12 flex flex-wrap items-center justify-center gap-2 font-mono text-xs text-muted">
        <span className="px-3 py-1.5 border border-border rounded">Traffic</span>
        <span>→</span>
        <span className="px-3 py-1.5 border border-accent/40 text-accent rounded">
          XGBoost
        </span>
        <span>→</span>
        <span className="px-3 py-1.5 border border-border rounded">Prediction</span>
        <span>→</span>
        <span className="px-3 py-1.5 border border-accent2/40 text-accent2 rounded">
          SHAP
        </span>
        <span>→</span>
        <span className="px-3 py-1.5 border border-border rounded">Explanation</span>
      </div>
    </section>
  );
}
