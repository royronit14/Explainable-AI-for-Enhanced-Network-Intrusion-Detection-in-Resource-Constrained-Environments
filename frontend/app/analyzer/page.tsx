"use client";

import { useState } from "react";
import {
  DEMO_SAMPLE,
  predictBatch,
  predictSingle,
  RAW_FEATURES,
  TOP10_RAW,
  type ApiError,
  type BatchPredictResponse,
  type ModelName,
  type SinglePredictResponse,
} from "@/lib/api";
import PredictionResult from "@/components/PredictionResult";
import BatchResults from "@/components/BatchResults";
import LoadingState from "@/components/LoadingState";
import ErrorMessage from "@/components/ErrorMessage";

type Mode = "single" | "batch";

export default function AnalyzerPage() {
  const [mode, setMode] = useState<Mode>("single");
  const [model, setModel] = useState<ModelName>("full");
  const [values, setValues] = useState<Record<string, string>>(() => {
    const init: Record<string, string> = {};
    for (const f of RAW_FEATURES) init[f] = "";
    return init;
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [single, setSingle] = useState<SinglePredictResponse | null>(null);
  const [batch, setBatch] = useState<BatchPredictResponse | null>(null);
  const [showAll, setShowAll] = useState(false);

  function setVal(k: string, v: string) {
    setValues((p) => ({ ...p, [k]: v }));
  }

  function fillDemo() {
    const next: Record<string, string> = {};
    for (const f of RAW_FEATURES) next[f] = String(DEMO_SAMPLE[f] ?? 0);
    setValues(next);
  }

  function clearAll() {
    setValues((p) => {
      const n: Record<string, string> = {};
      for (const k of Object.keys(p)) n[k] = "";
      return n;
    });
  }

  async function runSingle() {
    setError(null);
    setSingle(null);
    setBusy(true);
    try {
      // Both 'full' and 'light' require the full 39 raw features because the
      // deployed lightweight model was trained on a slice of the 39-feature
      // transformed matrix. The SHAP explanation in the response highlights
      // the top 10 contributions.
      const required = RAW_FEATURES;
      const features: Record<string, number> = {};
      for (const f of required) {
        const raw = values[f];
        if (raw === "" || raw === undefined) {
          throw new Error(`Feature "${f}" is required.`);
        }
        const n = Number(raw);
        if (Number.isNaN(n)) throw new Error(`Feature "${f}" must be numeric.`);
        features[f] = n;
      }
      const res = await predictSingle(features, model);
      setSingle(res);
    } catch (e) {
      const err = e as ApiError | Error;
      setError(("message" in err ? err.message : "Prediction failed."));
    } finally {
      setBusy(false);
    }
  }

  async function runBatch(file: File | null) {
    if (!file) return;
    setError(null);
    setBatch(null);
    setBusy(true);
    try {
      const res = await predictBatch(file, model);
      setBatch(res);
    } catch (e) {
      const err = e as ApiError | Error;
      setError("message" in err ? err.message : "Batch prediction failed.");
    } finally {
      setBusy(false);
    }
  }

  // The lightweight model is a slice of the full transformed matrix, so the
  // form always shows every raw input. SHAP still highlights the top 10.
  const requiredForModel = RAW_FEATURES;
  const extraFullFeatures = RAW_FEATURES.filter((f) => !TOP10_RAW.includes(f));

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-3xl font-bold">Traffic Analyzer</h1>
        <p className="text-muted text-sm mt-1">
          Submit a single network flow or upload a CSV. The backend reuses the
          trained XGBoost + SHAP pipeline.
        </p>
      </header>

      <div className="flex gap-2">
        <Tab active={mode === "single"} onClick={() => setMode("single")}>
          Single Flow
        </Tab>
        <Tab active={mode === "batch"} onClick={() => setMode("batch")}>
          Batch CSV
        </Tab>
      </div>

      <div className="glass rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center gap-6">
          <label className="flex items-center gap-2 text-sm">
            <input
              type="radio"
              name="model"
              value="full"
              checked={model === "full"}
              onChange={() => setModel("full")}
            />
            <span>
              Full XGBoost <span className="text-muted text-xs">(39 features)</span>
            </span>
          </label>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="radio"
              name="model"
              value="light"
              checked={model === "light"}
              onChange={() => setModel("light")}
            />
            <span>
              Lightweight{" "}
              <span className="text-muted text-xs">(top-10 SHAP highlights; full 39-feature input still required)</span>
            </span>
          </label>
        </div>

        {mode === "single" ? (
          <SingleForm
            required={requiredForModel}
            extras={model === "full" ? extraFullFeatures : []}
            showAll={showAll}
            setShowAll={setShowAll}
            values={values}
            setVal={setVal}
            onSubmit={runSingle}
            onDemo={fillDemo}
            onClear={clearAll}
            busy={busy}
          />
        ) : (
          <BatchForm
            onSubmit={runBatch}
            busy={busy}
            model={model}
          />
        )}
      </div>

      {busy && <LoadingState label="Analyzing network flow…" />}
      {error && <ErrorMessage title="Analysis failed" message={error} />}
      {!busy && !error && single && <PredictionResult result={single} />}
      {!busy && !error && batch && <BatchResults result={batch} />}
    </div>
  );
}

function Tab({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-4 py-1.5 rounded text-sm font-mono border ${
        active
          ? "border-accent text-accent bg-accent/5"
          : "border-border text-muted hover:text-slate-200"
      }`}
    >
      {children}
    </button>
  );
}

function SingleForm({
  required,
  extras,
  showAll,
  setShowAll,
  values,
  setVal,
  onSubmit,
  onDemo,
  onClear,
  busy,
}: {
  required: string[];
  extras: string[];
  showAll: boolean;
  setShowAll: (v: boolean) => void;
  values: Record<string, string>;
  setVal: (k: string, v: string) => void;
  onSubmit: () => void;
  onDemo: () => void;
  onClear: () => void;
  busy: boolean;
}) {
  const showExtraSection = extras.length > 0 && showAll;
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs text-muted">
          Required features: <span className="font-mono">{required.length}</span>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={onDemo}
            className="text-xs border border-border rounded px-3 py-1.5 hover:border-accent"
          >
            Try Demo
          </button>
          <button
            type="button"
            onClick={onClear}
            className="text-xs border border-border rounded px-3 py-1.5 hover:border-danger text-muted"
          >
            Clear
          </button>
        </div>
      </div>

      <div className="grid sm:grid-cols-3 md:grid-cols-4 gap-3">
        {required.map((f) => (
          <div key={f}>
            <label htmlFor={`f-${f}`} title={f}>{f}</label>
            <input
              id={`f-${f}`}
              type="number"
              step="any"
              value={values[f] ?? ""}
              onChange={(e) => setVal(f, e.target.value)}
              placeholder="0"
            />
          </div>
        ))}
      </div>

      {extras.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setShowAll(!showAll)}
            className="text-xs text-accent hover:underline"
          >
            {showAll ? "Hide" : "Show"} remaining {extras.length} full-model
            features
          </button>
          {showExtraSection && (
            <div className="mt-3 grid sm:grid-cols-3 md:grid-cols-4 gap-3 border-t border-border pt-3">
              {extras.map((f) => (
                <div key={f}>
                  <label htmlFor={`f-${f}`}>{f}</label>
                  <input
                    id={`f-${f}`}
                    type="number"
                    step="any"
                    value={values[f] ?? ""}
                    onChange={(e) => setVal(f, e.target.value)}
                    placeholder="0"
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="flex justify-end">
        <button
          type="button"
          disabled={busy}
          onClick={onSubmit}
          className="px-5 py-2.5 rounded-lg bg-accent text-bg font-semibold text-sm disabled:opacity-50"
        >
          {busy ? "Analyzing…" : "Analyze Traffic"}
        </button>
      </div>
    </div>
  );
}

function BatchForm({
  onSubmit,
  busy,
  model,
}: {
  onSubmit: (file: File | null) => void;
  busy: boolean;
  model: ModelName;
}) {
  const [file, setFile] = useState<File | null>(null);
  return (
    <div className="space-y-4">
      <p className="text-sm text-muted">
        Upload a CSV with the same raw feature columns used by the model
        (UNSW-NB15-style network flows). The first row may be a header.
      </p>
      <input
        type="file"
        accept=".csv,text/csv"
        disabled={busy}
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />
      <div className="flex justify-end">
        <button
          type="button"
          disabled={busy || !file}
          onClick={() => onSubmit(file)}
          className="px-5 py-2.5 rounded-lg bg-accent text-bg font-semibold text-sm disabled:opacity-50"
        >
          {busy ? "Uploading…" : "Analyze CSV"}
        </button>
      </div>
      <div className="text-xs text-muted">
        Currently using the <span className="font-mono">{model}</span> model.
      </div>
    </div>
  );
}
