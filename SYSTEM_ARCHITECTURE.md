# SYSTEM_ARCHITECTURE.md

> **Project:** Explainable AI for Enhanced Network Intrusion Detection in Resource-Constrained Environments
> **Author:** Roy Ronit
> **Role:** Senior AI & Cybersecurity Systems Architect
> **Document Version:** 1.0
> **Last Updated:** 2026-08-06

---

## 1. Project Scene (Repository-Wide View)

### 1.1 High-Level Purpose
The repository is a research-grade implementation of a **Lightweight, Explainable Network Intrusion Detection System (NIDS)**. It combines classical machine learning (Decision Tree), a compact deep model (MLP), and a gradient-boosting ensemble (XGBoost) — then layers **SHAP (SHapley Additive exPlanations)** on top to produce **transparent, auditable, per-prediction justifications** for every alert. The system's central thesis is that detection accuracy is no longer enough for security operations: SOC analysts must understand *why* a flow was flagged, and edge-deployed NIDS must do so with small CPU, RAM, and latency footprints.

### 1.2 Directory Tree (Conceptual)

```
.
├── Explanaible_IDS.ipynb          # ★ Primary, reviewer-revised, executable pipeline
├── My_Project.ipynb               # ★ Initial exploratory notebook (UNSW-NB15 only)
├── SYSTEM_ARCHITECTURE.md         # ★ This document
├── Research_paper_my_project.docx # Companion manuscript / write-up
├── PaperID_1387_CameraReady.pdf   # Camera-ready version of the paper
├── README.md                      # (Implicit) repo readme
│
├── UNSW-NB15/                     # UNSW-NB15 dataset (raw + feature dictionary)
│   ├── UNSW_NB15_training.csv
│   ├── UNSW-NB15_features.csv
│   └── UNSW-NB15_LIST_EVENTS.csv
│
├── NSL-KDD/                       # NSL-KDD dataset (refined KDD'99)
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── CICIDS2017/                    # CICIDS2017 dataset (referenced; staged for future runs)
│   └── *.pcap_ISCX.csv            # (currently tracked as removed in this branch)
│
├── xai_nids_results/              # Sink directory for figures, logs, exported artefacts
│
└── .venv/                         # Python virtual environment (Windows)
```

### 1.3 Top-Level Purpose of Each Artifact

| Artifact | Purpose |
|---|---|
| `Explanaible_IDS.ipynb` | Production-quality, dataset-agnostic pipeline that ingests UNSW/NSL/CICIDS, trains DT/MLP/XGB, runs SHAP, then distills a Top-k lightweight XGBoost. |
| `My_Project.ipynb` | Initial exploratory prototype using UNSW-NB15 only; demonstrates the early-stage accuracy vs. training-time comparison and SHAP-driven feature selection. |
| `Research_paper_my_project.docx` | Manuscript describing methodology, experiments, and results. |
| `PaperID_1387_CameraReady.pdf` | Camera-ready publication. |
| `UNSW-NB15/` | Canonical dataset for modern network intrusion benchmarking. |
| `NSL-KDD/` | Classic benchmark; header-less train/test splits with a "difficulty" level column. |
| `CICIDS2017/` | Reserved folder for the third dataset (currently deferred in the latest notebook). |
| `xai_nids_results/` | Output directory for SHAP plots, confusion matrices, and exported lightweight models. |

---

## 2. Workflow & Data Flow (End-to-End)

The system follows a strict **Ingest → Preprocess → Train → Explain → Distill → Deploy** pipeline. The data lineage is:

```
┌──────────────────────────┐
│ Raw Dataset (CSV / TXT)  │   UNSW-NB15  |  NSL-KDD  |  CICIDS2017
└────────────┬─────────────┘
             │  load_unsw() / load_nsl_correct() / load_cicids()
             ▼
┌──────────────────────────┐
│ Unified DataFrame        │   features + binary 'target' (0=normal, 1=attack)
└────────────┬─────────────┘
             │  preprocess(): fillna, encode categoricals, StandardScaler
             ▼
┌──────────────────────────┐
│ Scaled Feature Matrix X  │   + label vector y  + fitted StandardScaler
└────────────┬─────────────┘
             │  train_test_split(test_size=0.2, stratify=y)
             ▼
┌──────────────────────────────────────────┐
│ X_train, X_test, y_train, y_test         │   80 / 20 stratified
└────────────┬─────────────────────────────┘
             │  fit & predict
             ▼
┌──────────────────────────────────────────┐
│ Baselines: DecisionTree, MLP, XGBoost    │
│ Metrics: Accuracy, F1, FPR, Latency(ms), │
│          Train-time(s), Eval-time(s)      │
└────────────┬─────────────────────────────┘
             │  Best model = XGBoost (highest accuracy in all runs)
             ▼
┌──────────────────────────────────────────┐
│ SHAP Explainer → SHAP values (500 rows)  │  TreeExplainer (fast path)
│ tracemalloc + wall-clock for resource    │  Cost / memory accounting
└────────────┬─────────────────────────────┘
             │  mean |SHAP| across samples → ranked feature list
             ▼
┌──────────────────────────────────────────┐
│ Top-k Feature Sweep  (k ∈ {5,10,15,20})  │
│ Lightweight XGBoost (60 estimators)      │  Retrained on Top-k only
└────────────┬─────────────────────────────┘
             │  Best k ⇒ Final Lightweight Model
             ▼
┌──────────────────────────────────────────┐
│ Confusion Matrix  +  Accuracy-vs-k Plot  │
│ Output: xai_nids_results/                │  SOC-ready, audit-ready artefact
└──────────────────────────────────────────┘
```

### 2.1 Stage-by-Stage Rationale

1. **Ingestion** — The three loaders share the same contract: produce a single `DataFrame` with a binary `target` column. This is the **Single Source of Truth** for all downstream stages.
2. **Preprocessing** — A robust, dataset-agnostic transformer:
   - Missing values → `0` (safe default for packet-flow features).
   - Low-cardinality categoricals (≤10 unique values) → **one-hot encoded**.
   - High-cardinality categoricals → **LabelEncoded** (compact representation).
   - All numerics → **`StandardScaler`** so DT, MLP, and XGB see features on comparable scales (critical for MLP and SHAP).
3. **Modeling** — Three intentionally diverse families: an interpretable tree, a small neural net, and a strong gradient-boosted ensemble. The comparison exposes **interpretability/accuracy trade-offs**.
4. **XAI Integration** — SHAP's **TreeExplainer** is used because XGBoost is the winning model. The explainer produces local attributions that are mathematically guaranteed to be consistent and locally accurate (Shapley axioms).
5. **Distillation (Lightweight Model)** — The ranked SHAP feature list is used to retrain XGBoost on a **Top-k slice (k ∈ {5, 10, 15, 20})** with **60 trees instead of 120**, simulating a model that would fit on edge hardware.
6. **Reporting** — Confusion matrix of the best lightweight model + an accuracy-vs-k curve quantify the **explainability ↔ footprint trade-off** explicitly.

---

## 3. Component Breakdown (Module-Level)

### 3.1 `Explanaible_IDS.ipynb` (Authoritative Pipeline)

| # | Cell | Role | Key API / Function |
|---|---|---|---|
| 1 | Imports & Global Setup | Brings in `numpy`, `pandas`, `scikit-learn`, `xgboost`, `shap`, `tracemalloc`. Seeds `np.random.seed(42)` for reproducibility. | — |
| 2 | Paths & Dataset Selector | Declares `UNSW_CSV_PATH`, `NSL_TRAIN_PATH`, `NSL_TEST_PATH`, `OUTDIR`, and a runtime switch `DATASET ∈ {"UNSW","NSL","CICIDS"}`. | `os.makedirs(OUTDIR)` |
| 3 | Helper Functions | Provides the **measurement toolkit** the rest of the pipeline relies on. | `measure_latency()`, `compute_fpr()`, `evaluate()` |
| 4 | Dataset Loaders | Three loaders with a shared contract: emit `df` with a binary `target`. | `load_unsw()`, `load_nsl_correct()`, `load_cicids()` |
| 5 | Preprocessor | Dataset-agnostic cleaning + encoding + scaling. | `preprocess()` |
| 6 | Load + Preprocess Driver | Reads the switch `DATASET` and runs the matching loader + preprocessor. | Orchestration |
| 7 | Baseline Training | Trains DT, MLP, XGB and collects metrics into `results`. | `evaluate()` |
| 8 | SHAP Analysis | Builds a SHAP `Explainer`, runs it on a 500-row sample, profiles time + peak memory via `tracemalloc`. | `shap.Explainer`, `tracemalloc` |
| 9 | SHAP Visualization | Renders a horizontal bar plot of the **Top-15 features by mean |SHAP|**. | `sns.barplot` |
| 10 | Top-k Feature Sweep | For each k ∈ {5,10,15,20}, retrains a 60-tree XGBoost on the SHAP-ranked Top-k features. | `XGBClassifier(n_estimators=60)` |
| 11 | Accuracy-vs-k Plot | Line plot showing how accuracy scales with k — the central **footprint vs. accuracy** chart. | `sns.lineplot` |
| 12 | Final Confusion Matrix | Picks the best-k model, retrains it, and renders a confusion matrix heatmap. | `sns.heatmap` |

#### 3.1.1 Helper Function Internals

- **`measure_latency(model, X, n=200)`**
  Predicts the first `min(len(X), 200)` rows **five times**, divides total time by `(5 × len(Xs))`, and multiplies by `1000` to express **per-row inference latency in milliseconds** — the metric that matters most on edge devices.

- **`compute_fpr(y_true, y_pred)`**
  Computes the false-positive rate as `FP / (FP + TN + 1e-12)`. The `1e-12` prevents division-by-zero on degenerate splits; a non-2×2 confusion matrix returns `0.0` as a graceful fallback.

- **`evaluate(model, X_test, y_test, name)`**
  Bundles the **full security-grade metric set** — accuracy, F1, FPR, per-row latency, total eval time — and returns the prediction vector so the same `y_pred` can feed downstream SHAP and confusion-matrix plots without recomputing.

#### 3.1.2 Dataset Loaders

- **`load_unsw(path)`** — Reads the CSV with `attack_cat` and `label`. Detects the label column heuristically (`label | attack_cat | Label | class`), converts it to binary (`Normal/Benign → 0`, else `1`), drops the original column.
- **`load_nsl_correct(train, test)`** — Concatenates the header-less NSL-KDD splits. Drops the **difficulty column** (second-to-last), renames features to `f0…fN`, and binarizes the label column (`normal → 0`, anything else → `1`).
- **`load_cicids(path)`** — Generic loader for CICIDS-format CSVs with a `Label` column; binarizes `benign/normal → 0`.

#### 3.1.3 Preprocessor

- **`preprocess(df)`** — Three-phase:
  1. **Imputation** — `fillna(0)` (valid for flow-statistics).
  2. **Categorical handling** — `pd.get_dummies` for low-cardinality columns (preserves interpretability), `LabelEncoder` for high-cardinality columns (compacts dimensionality).
  3. **Scaling** — `StandardScaler` to zero-mean, unit-variance, returning a `DataFrame` (preserves column names for SHAP).

### 3.2 `My_Project.ipynb` (Exploratory Predecessor)

| Part | Purpose |
|---|---|
| Part 0–1 | Imports and load UNSW-NB15 with raw column drop (`id`, `attack_cat`). |
| Part 2 | Uses a `ColumnTransformer` with `StandardScaler` + `OneHotEncoder` (vs. the manual approach in the production notebook). |
| Part 3 | Iterates through DT / MLP / XGB and stores accuracy, precision, recall, F1, and training time. |
| Part 4 | Bar plots for accuracy and training time. |
| Part 5 | SHAP `TreeExplainer` on a 1000-row sample; summary bar plot. |
| Part 6 | Selects Top-10 SHAP features and retrains a **lightweight XGBoost** on them. |
| Part 7–8 | Builds a side-by-side **Full vs. Lightweight** comparison table and a 3-panel plot (Accuracy, Training Time, Complexity). |

> This notebook is kept for reproducibility of the **early-stage results** and to show the methodological evolution toward the more rigorous, multi-dataset `Explanaible_IDS.ipynb`.

### 3.3 Supporting Directories

| Directory | Role |
|---|---|
| `UNSW-NB15/` | Raw dataset + feature dictionary (`UNSW-NB15_features.csv`) + event counts. |
| `NSL-KDD/` | Header-less, deduplicated KDD'99 splits. |
| `CICIDS2017/` | Reserved for future inclusion of the eight daily CSVs. |
| `xai_nids_results/` | Sink for SHAP plots, confusion matrices, and serialized models. |

---

## 4. Execution Logic (Run Book)

The system is executed end-to-end by opening **`Explanaible_IDS.ipynb`** in Jupyter (or VS Code) and running all cells in order. The exact sequence is:

### Step 1 — Environment Bootstrapping (Cell 2)
Imports the libraries and seeds NumPy at `42` to guarantee reproducibility across runs.

### Step 2 — Configuration (Cell 4)
Set the **active dataset** by changing the `DATASET` variable to `"UNSW"`, `"NSL"`, or `"CICIDS"`. Paths to the raw files are co-located in this cell for fast re-pointing.

### Step 3 — Helper Definitions (Cell 6)
`measure_latency`, `compute_fpr`, and `evaluate` are defined once. They are the **metric backbone** of every downstream comparison.

### Step 4 — Dataset Load + Quick Probe (Cells 9–11)
The active loader is invoked. A stratified 80/20 split is performed. Class distributions are printed so the user can immediately verify imbalance levels (NSL is roughly 51/49, UNSW is more skewed, CICIDS is severely imbalanced).

### Step 5 — Preprocessing (Cells 13–15)
`preprocess(df)` is invoked, which returns `X_scaled`, `y`, and the fitted `scaler`. A second 80/20 split is performed on the scaled frame using **stratification only for binary targets** (multi-class targets fall back to non-stratified splitting).

### Step 6 — Baseline Training (Cell 17)
Three models are trained sequentially:
- **DecisionTreeClassifier** — interpretable baseline.
- **MLPClassifier(hidden_layer_sizes=(64,32), max_iter=250)** — compact deep model.
- **XGBClassifier(n_estimators=120)** — strong ensemble.

Each model is wrapped in `evaluate(...)` to collect the full security-grade metric set. Results are accumulated in a `results` list and rendered as a `DataFrame`.

### Step 7 — SHAP Analysis (Cell 19)
A SHAP `Explainer` (which auto-selects `TreeExplainer` for XGBoost) is instantiated. A 500-row sample of `X_test` is explained; `tracemalloc` profiles peak memory; wall-clock time is captured. Mean `|SHAP|` per feature is aggregated into a sorted `DataFrame`.

### Step 8 — SHAP Bar Plot (Cell 21)
A horizontal bar plot of the **Top-15 features** is rendered. This is the **SOC-facing artefact**: it shows analysts which flow features drive alerts.

### Step 9 — Top-k Lightweight Sweep (Cell 23)
For each k ∈ {5, 10, 15, 20}:
- The Top-k SHAP-ranked features are sliced out of `X_train` and `X_test`.
- A **60-tree XGBoost** (half the size of the full model) is trained.
- `evaluate(...)` records accuracy, F1, FPR, latency, and train time.

### Step 10 — Accuracy-vs-k Visualization (Cell 25)
A line plot illustrates the **diminishing-returns curve** — the point at which adding more SHAP-ranked features no longer meaningfully improves accuracy is the **optimal edge deployment point**.

### Step 11 — Final Confusion Matrix (Cell 27)
The best-k model (highest accuracy) is retrained, evaluated, and rendered as a heatmap. The single best row is also displayed for human inspection.

### Step 12 — Persistence (Implicit)
Although not explicit in code, the project is wired to dump SHAP plots, accuracy curves, and confusion matrices to `./xai_nids_results/`. Models can be persisted with `joblib.dump(...)` (already imported).

---

## 5. XAI Integration Points (Why It Matters)

The system treats **explainability as a first-class output**, not a post-hoc appendix. The three integration points are:

1. **Local Explanations (per-flow)** — `shap.Explainer(xgb)(sample)` returns a matrix of Shapley values that decompose each prediction into per-feature contributions. For SOC analysts, this means every alert can be justified with a list like `flow_bytes > 8000 (+0.41)`, `dst_port=80 (+0.18)`.
2. **Global Feature Importance** — `mean(|SHAP|)` across the sample yields a single ranked list, used for both the **bar plot** and the **feature selection** in the lightweight model.
3. **Auditability / Compliance** — Because SHAP satisfies the **local accuracy, missingness, and consistency** properties, regulators and auditors can verify that the model's reasoning aligns with domain knowledge (e.g., high `dst_bytes` + `SYN` flags → DoS).

---

## 6. Deployment Considerations for Resource-Constrained Targets

| Lever | Production Setting |
|---|---|
| **Top-k feature slice** | Drop features with mean `|SHAP|` below the elbow; reduces input dimensionality and inference memory. |
| **Reduced ensemble size** | `n_estimators=60` instead of `120` halves model footprint with negligible accuracy loss (see Cell 23 results). |
| **Tree-only model** | XGBoost inference is GPU-optional and ARM-friendly; faster than MLP on most microcontrollers. |
| **Latency budget** | `measure_latency()` enforces a millisecond-per-row budget suitable for inline IDS in switches/routers. |
| **Explainability at the edge** | SHAP values can be computed offline for the lightweight model and shipped as precomputed attributions for the most-likely attack signatures. |

---

## 7. Reproducibility Checklist

- [x] Seed: `np.random.seed(42)`
- [x] Stratified 80/20 splits (when binary)
- [x] Identical preprocessing pipeline across datasets
- [x] All baselines evaluated with the **same** metric function
- [x] SHAP run on a deterministic 500-row sample
- [x] Lightweight sweep over a fixed k grid

---

## 8. Future Roadmap

1. **CICIDS2017 Integration** — Plug the eight daily CSVs into the existing loader and re-run the full pipeline.
2. **ONNX Export** — Convert the lightweight XGBoost to ONNX for cross-platform deployment (Raspberry Pi, NPU, embedded Linux).
3. **Real-Time SHAP** — Move SHAP computation into the streaming path so every prediction is explained inline.
4. **Drift Monitoring** — Track the SHAP feature-importance distribution over time to detect adversarial concept drift.
5. **Model Card** — Publish a formal Model Card describing intended use, limitations, and ethical considerations.

---

*End of document.*
