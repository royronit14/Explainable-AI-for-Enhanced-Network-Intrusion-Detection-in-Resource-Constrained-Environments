import { useState } from "react";
import type { BatchPredictResponse } from "@/lib/api";

interface Props {
  result: BatchPredictResponse;
}

const PAGE_SIZE = 25;

export default function BatchResults({ result }: Props) {
  const [page, setPage] = useState(0);
  const totalPages = Math.ceil(result.results.length / PAGE_SIZE);
  const slice = result.results.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);
  const attackRate = result.total_rows
    ? ((result.attack / result.total_rows) * 100).toFixed(2)
    : "0.00";

  function downloadCsv() {
    const header = ["prediction", "label", "probability"].join(",");
    const lines = result.results.map((r) =>
      [r.prediction, JSON.stringify(r.label), r.probability.toFixed(6)].join(","),
    );
    const blob = new Blob([header + "\n" + lines.join("\n")], {
      type: "text/csv",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "nids_predictions.csv";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-4">
        <Stat label="Total Flows" value={result.total_rows} />
        <Stat label="Normal" value={result.normal} tone="normal" />
        <Stat label="Attacks" value={result.attack} tone="attack" />
      </div>

      <div className="glass rounded-lg p-4 flex items-center justify-between">
        <div className="text-sm">
          Attack rate:&nbsp;
          <span className="font-mono text-accent">{attackRate}%</span>
        </div>
        <button
          onClick={downloadCsv}
          className="text-xs font-mono border border-border rounded px-3 py-1.5 hover:border-accent"
        >
          Download CSV
        </button>
      </div>

      <div className="glass rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-panel text-muted text-xs uppercase">
            <tr>
              <th className="text-left px-4 py-2">#</th>
              <th className="text-left px-4 py-2">Prediction</th>
              <th className="text-left px-4 py-2">Label</th>
              <th className="text-right px-4 py-2">Confidence</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((row, i) => {
              const isAttack = row.label.toLowerCase() !== "normal";
              return (
                <tr key={page * PAGE_SIZE + i} className="border-t border-border">
                  <td className="px-4 py-2 font-mono text-muted">
                    {page * PAGE_SIZE + i + 1}
                  </td>
                  <td className="px-4 py-2 font-mono">{row.prediction}</td>
                  <td
                    className={`px-4 py-2 ${
                      isAttack ? "text-danger" : "text-accent2"
                    }`}
                  >
                    {row.label}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    {(row.probability * 100).toFixed(2)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between text-xs text-muted">
          <button
            disabled={page === 0}
            onClick={() => setPage((p) => Math.max(0, p - 1))}
            className="border border-border rounded px-3 py-1 disabled:opacity-40"
          >
            Prev
          </button>
          <span>
            Page {page + 1} of {totalPages}
          </span>
          <button
            disabled={page >= totalPages - 1}
            onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            className="border border-border rounded px-3 py-1 disabled:opacity-40"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

function Stat({
  label,
  value,
  tone,
}: {
  label: string;
  value: number;
  tone?: "normal" | "attack";
}) {
  const color =
    tone === "attack" ? "text-danger" : tone === "normal" ? "text-accent2" : "text-slate-200";
  return (
    <div className="glass rounded-lg p-4">
      <div className="text-[11px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`text-2xl font-mono ${color}`}>{value.toLocaleString()}</div>
    </div>
  );
}
