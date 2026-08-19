"use client";

import { useEffect, useState } from "react";
import Hero from "@/components/Hero";
import ModelCard from "@/components/ModelCard";
import LoadingState from "@/components/LoadingState";
import ErrorMessage from "@/components/ErrorMessage";
import { getModelInfo, type ModelInfoResponse } from "@/lib/api";

export default function HomePage() {
  const [info, setInfo] = useState<ModelInfoResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getModelInfo()
      .then((m) => !cancelled && setInfo(m))
      .catch((e) => !cancelled && setError(e.message ?? "Could not reach backend."));
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-14">
      <Hero />

      <section>
        <h2 className="text-lg font-semibold mb-4">Models</h2>
        {error && <ErrorMessage message={error} />}
        {!error && !info && <LoadingState label="Loading model information…" />}
        {info && (
          <div className="grid sm:grid-cols-2 gap-4">
            <ModelCard
              title="Full XGBoost"
              subtitle="39 raw network-flow features"
              info={info.full_model}
              highlight
            />
            <ModelCard
              title="Lightweight XGBoost"
              subtitle="Top-10 SHAP-selected features"
              info={info.lightweight_model}
            />
          </div>
        )}
        {info && info.top_features?.length > 0 && (
          <div className="mt-4 glass rounded-lg p-4">
            <div className="text-[11px] uppercase tracking-wider text-muted mb-2">
              SHAP Top Features (Lightweight Model)
            </div>
            <div className="flex flex-wrap gap-2 font-mono text-xs">
              {info.top_features.map((f) => (
                <span
                  key={f}
                  className="px-2 py-1 border border-accent/30 text-accent rounded"
                >
                  {f.replace("num__", "")}
                </span>
              ))}
            </div>
          </div>
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-4">Why it matters</h2>
        <div className="grid sm:grid-cols-3 gap-4">
          <Pillar
            title="Detection"
            body="Identifies potentially malicious network traffic using an XGBoost classifier trained on UNSW-NB15."
          />
          <Pillar
            title="Explainability"
            body="SHAP values reveal which network-flow features contributed most to each prediction."
          />
          <Pillar
            title="Lightweight Detection"
            body="SHAP-selected features reduce input representation while retaining comparable predictive performance."
          />
        </div>
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-4">Model comparison</h2>
        <div className="glass rounded-lg p-5 text-sm text-slate-300">
          The lightweight model uses SHAP-selected features and aims to reduce
          the input representation while retaining comparable predictive
          performance to the full XGBoost baseline. Both numbers above come
          from the live FastAPI backend.
        </div>
      </section>
    </div>
  );
}

function Pillar({ title, body }: { title: string; body: string }) {
  return (
    <div className="glass rounded-xl p-5">
      <h3 className="font-semibold mb-2 text-accent">{title}</h3>
      <p className="text-sm text-slate-300">{body}</p>
    </div>
  );
}
