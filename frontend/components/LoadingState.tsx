export default function LoadingState({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="glass rounded-lg p-6 flex items-center gap-3 text-sm text-muted">
      <span className="w-3 h-3 rounded-full bg-accent animate-pulse" />
      {label}
    </div>
  );
}
