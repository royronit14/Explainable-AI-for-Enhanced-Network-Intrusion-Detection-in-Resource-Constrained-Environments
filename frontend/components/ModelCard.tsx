import type { ModelSubInfo } from "@/lib/api";

interface Props {
  title: string;
  subtitle: string;
  info: ModelSubInfo;
  highlight?: boolean;
}

export default function ModelCard({ title, subtitle, info, highlight }: Props) {
  return (
    <div
      className={`glass rounded-xl p-6 ${
        highlight ? "border-accent/40 shadow-[0_0_0_1px_rgba(34,211,238,0.25)]" : ""
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <h3 className="font-semibold">{title}</h3>
        {highlight && (
          <span className="text-[10px] uppercase tracking-wider text-accent font-mono">
            default
          </span>
        )}
      </div>
      <div className="text-xs text-muted mb-4">{subtitle}</div>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-muted">
            Accuracy
          </div>
          <div className="text-2xl font-mono text-accent2">
            {(info.accuracy * 100).toFixed(2)}%
          </div>
        </div>
        <div>
          <div className="text-[11px] uppercase tracking-wider text-muted">
            Features
          </div>
          <div className="text-2xl font-mono">{info.features}</div>
        </div>
      </div>
    </div>
  );
}
