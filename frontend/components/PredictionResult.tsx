import type { SinglePredictResponse } from "@/lib/api";
import ShapChart from "./ShapChart";

export default function PredictionResult({ result }: { result: SinglePredictResponse }) {
  const isAttack = result.label.toLowerCase() !== "normal";
  const pct = (result.probability * 100).toFixed(2);

  return (
    <div className="glass rounded-xl p-6 space-y-5">
      <div
        className={`rounded-lg p-4 text-center ${
          isAttack
            ? "bg-danger/10 border border-danger/40"
            : "bg-accent2/10 border border-accent2/40"
        }`}
      >
        <div className="text-3xl mb-1">{isAttack ? "🚨" : "✓"}</div>
        <div
          className={`font-mono text-xl font-bold ${
            isAttack ? "text-danger" : "text-accent2"
          }`}
        >
          {isAttack ? "ATTACK DETECTED" : "NORMAL TRAFFIC"}
        </div>
        <div className="text-xs text-muted mt-1">
          Class label: <span className="font-mono">{result.label}</span> (id={result.prediction})
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="glass rounded-lg p-4">
          <div className="text-[11px] uppercase tracking-wider text-muted">
            Confidence
          </div>
          <div className="text-2xl font-mono text-accent">{pct}%</div>
          <div className="text-xs text-muted mt-1">probability of attack</div>
        </div>
        <div className="glass rounded-lg p-4">
          <div className="text-[11px] uppercase tracking-wider text-muted">Model</div>
          <div className="text-2xl font-mono">
            {result.model === "full" ? "Full XGBoost" : "Lightweight XGBoost"}
          </div>
          <div className="text-xs text-muted mt-1">
            {result.explanation.length} feature contributions
          </div>
        </div>
      </div>

      <div>
        <h4 className="text-sm font-semibold mb-2 text-muted uppercase tracking-wider">
          SHAP Model Contribution
        </h4>
        <ShapChart items={result.explanation} />
      </div>
    </div>
  );
}
