export default function ResearchPage() {
  return (
    <div className="space-y-12 max-w-3xl">
      <header>
        <h1 className="text-3xl font-bold">Research</h1>
        <p className="text-muted text-sm mt-1">
          A summary of the published approach behind this system.
        </p>
      </header>

      <Section title="Problem">
        <p>
          Modern networks generate vast amounts of flow data. Identifying
          malicious traffic reliably — and explaining <em>why</em> a model
          flagged it — is critical for security analysts and for deploying
          detection models in resource-constrained environments.
        </p>
      </Section>

      <Section title="Approach">
        <ol className="list-decimal pl-5 space-y-2">
          <li>UNSW-NB15 network-flow dataset.</li>
          <li>Preprocessing: numeric scaling + categorical encoding.</li>
          <li>Train XGBoost as the full-detection baseline.</li>
          <li>Apply SHAP to rank features by their model contribution.</li>
          <li>Retrain a lightweight XGBoost on the Top-10 SHAP features.</li>
        </ol>
      </Section>

      <Section title="Results">
        <div className="grid sm:grid-cols-2 gap-3">
          <Stat label="Full XGBoost" value="97.64% accuracy · 39 features" />
          <Stat label="Lightweight XGBoost" value="97.30% accuracy · 10 features" />
        </div>
        <p className="mt-3 text-sm text-muted">
          Numbers come from the published notebook and the live FastAPI
          backend (<code>/model-info</code>).
        </p>
      </Section>

      <Section title="Explainability">
        <p>
          SHAP (SHapley Additive exPlanations) attributes each prediction to
          individual feature contributions. The system uses a local SHAP
          explanation per prediction so analysts can see which features
          pushed the model toward its decision.
        </p>
      </Section>

      <Section title="Lightweight model">
        <p>
          Reducing the input to the most informative features lowers the
          computational cost of feature collection and inference — useful for
          resource-constrained environments such as edge gateways. The
          lightweight XGBoost retains most of the full-model accuracy.
        </p>
      </Section>

      <Section title="Limitations">
        <ul className="list-disc pl-5 space-y-1">
          <li>
            Trained and evaluated on UNSW-NB15 only. Real-world traffic may
            shift the feature distribution.
          </li>
          <li>
            No real-time packet capture, blocking, or SOC workflow is
            implemented.
          </li>
          <li>
            SHAP explanations describe model contribution, not causality.
          </li>
          <li>
            This is a research artifact, not a production intrusion-prevention
            system.
          </li>
        </ul>
      </Section>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section>
      <h2 className="text-lg font-semibold mb-2 text-accent">{title}</h2>
      <div className="text-sm text-slate-300 space-y-2 leading-relaxed">
        {children}
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="glass rounded-lg p-4">
      <div className="text-[11px] uppercase tracking-wider text-muted">
        {label}
      </div>
      <div className="text-base font-mono text-accent2">{value}</div>
    </div>
  );
}
