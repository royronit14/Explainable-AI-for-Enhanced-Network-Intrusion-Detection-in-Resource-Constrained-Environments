interface Props {
  title?: string;
  message: string;
}

export default function ErrorMessage({ title = "Something went wrong", message }: Props) {
  return (
    <div
      role="alert"
      className="glass rounded-lg p-4 border-danger/40 bg-danger/5 text-sm"
    >
      <div className="font-semibold text-danger mb-1">{title}</div>
      <div className="text-slate-300">{message}</div>
    </div>
  );
}
