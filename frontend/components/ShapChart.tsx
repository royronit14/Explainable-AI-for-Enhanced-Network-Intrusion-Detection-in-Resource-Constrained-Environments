import type { ShapItem } from "@/lib/api";

interface Props {
  items: ShapItem[];
  /** How many top items to render. */
  limit?: number;
}

export default function ShapChart({ items, limit = 12 }: Props) {
  if (!items?.length) {
    return <div className="text-sm text-muted">No explanation data.</div>;
  }
  const visible = items.slice(0, limit);
  const max = Math.max(...visible.map((i) => Math.abs(i.shap_value)), 1e-9);

  return (
    <div className="space-y-2">
      {visible.map((it) => {
        const pct = (Math.abs(it.shap_value) / max) * 100;
        const positive = it.shap_value >= 0;
        return (
          <div key={it.feature} className="grid grid-cols-[180px_1fr_80px] items-center gap-2">
            <div
              className="font-mono text-xs truncate text-slate-300"
              title={it.feature}
            >
              {stripPrefix(it.feature)}
            </div>
            <div className="h-5 bg-panel rounded relative overflow-hidden">
              <div
                className={`h-full ${positive ? "bg-accent" : "bg-danger"}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <div
              className={`text-xs font-mono text-right ${
                positive ? "text-accent" : "text-danger"
              }`}
            >
              {it.shap_value.toFixed(3)}
            </div>
          </div>
        );
      })}
      <div className="text-xs text-muted pt-1">
        Bars show model contribution. Cyan = pushes toward predicted class,
        red = pushes away. Magnitudes are SHAP values, not causal effects.
      </div>
    </div>
  );
}

function stripPrefix(name: string): string {
  // "num__sttl" -> "sttl"; pass through anything else.
  const idx = name.indexOf("__");
  return idx >= 0 ? name.slice(idx + 2) : name;
}
