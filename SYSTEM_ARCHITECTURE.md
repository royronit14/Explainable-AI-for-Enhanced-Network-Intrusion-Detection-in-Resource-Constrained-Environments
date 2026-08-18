# SYSTEM_ARCHITECTURE.md

> **Project:** Explainable AI for Enhanced Network Intrusion Detection in Resource-Constrained Environments
> **Author:** Roy Ronit
> **Role:** Senior AI & Cybersecurity Systems Architect
> **Document Version:** 2.0 — rewritten to match the actual repository state
> **Last Updated:** 2026-08-06

---

## 0. Note to the Reader

The previous version of this document described a 12-cell, multi-dataset pipeline (`load_unsw / load_nsl_correct / load_cicids`, Top-k sweep over k ∈ {5,10,15,20}, accuracy-vs-k plot, NSL-KDD folder, CICIDS2017 folder, a camera-ready PDF) that **does not exist in the current code base**. That description was an aspirational target — the shape the reviewers said they wanted, not what was actually committed.

This version is grounded in the notebooks, datasets, and files that are actually checked into the repository as of today. If a system element is *planned but not implemented*, it is called out explicitly in **§5 Future Work** rather than presented as live behavior.

---

## 1. Project Scene (Repository-Wide View)

### 1.1 High-Level Purpose

The repository is a research-grade implementation of a **Lightweight, Explainable Network Intrusion Detection System (NIDS)** for environments where CPU, RAM, and per-flow latency are constrained. It compares three model families on the UNSW-NB15 dataset — an interpretable **Decision Tree**, a compact **MLP**, and a strong **XGBoost** ensemble — selects XGBoost, then layers **SHAP (SHapley Additive exPlanations)** on top to produce transparent, per-prediction justifications for every alert. The final contribution is a **lightweight XGBoost variant trained on the SHAP-ranked Top-10 features** that preserves accuracy while drastically cutting compute footprint.

The core thesis: detection accuracy is necessary but not sufficient for security operations. SOC analysts must understand *why* a flow was flagged, and edge-deployed NIDS must do so with small CPU, RAM, and latency footprints.

### 1.2 Actual Directory Tree

```
.
├── .gitignore                                      # Excludes .venv/, *.png outputs, etc.
├── README.md                                       # Workspace overview
├── WORKFLOW.md                                     # Day-to-day Git / branch workflow
├── SYSTEM_ARCHITECTURE.md                          # ★ This document
│
├── My_Project.ipynb                                # ★ Active pipeline (8-part)
├── Explanaible_IDS.ipynb                           # Reviewer-suggested revision, currently a stub
├── Explanaible_IDS_executed.ipynb                  # Generated executed copy (gitignored outputs)
│
├── requirements.txt                               # Python dependencies (9 packages)
│
├── UNSW_NB15_training.csv                         # ★ Primary dataset (175,341 records, ~14 MB)
├── UNSW-NB15_features.csv                          # Feature dictionary
├── UNSW-NB15_LIST_EVENTS.csv                       # Attack-category → event mapping
│
├── Research_paper_my_project.docx                  # Companion manuscript
│
├── .venv/                                          # Local Python virtual environment (gitignored)
└── xai_nids_results/                               # Output sink for figures (currently empty except .gitkeep)
```

### 1.3 Top-Level Purpose of Each Artifact

| Artifact | Purpose |
|---|---|
| `My_Project.ipynb` | **The active, runnable pipeline.** Eight cells walk through data load → preprocessing → model comparison → SHAP → lightweight retraining → final comparison plot. |
| `Explanaible_IDS.ipynb` | Reviewer-suggested successor notebook. Currently a 2-cell stub (title heading + empty cell) intended as the home for the multi-dataset / Top-k-sweep architecture described in commit history. **Not yet implemented.** |
| `requirements.txt` | Nine pinned-by-name dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `shap`, `lime`, `eli5`, plus `jupyter` / `ipykernel` for the runtime. `lime` and `eli5` are listed but not currently used in the active notebook. |
| `UNSW_NB15_training.csv` | The canonical training split of the UNSW-NB15 dataset (175,341 rows × 45 cols including label). |
| `UNSW-NB15_features.csv` | Human-readable feature dictionary. |
| `UNSW-NB15_LIST_EVENTS.csv` | Maps attack-category labels to event IDs. |
| `Research_paper_my_project.docx` | Methodology + results write-up referenced by the manuscript. |
| `xai_nids_results/` | Reserved output directory for SHAP plots and confusion matrices. Currently empty (`.gitkeep` only). |
| `WORKFLOW.md` | Branch-and-recovery playbook (`main` vs `roy-local-work`). Out of scope for the ML architecture; referenced here for completeness. |

---

## 2. Workflow & Data Flow (End-to-End)

The pipeline follows a strict **Load → Preprocess → Train → Evaluate → Explain → Distill → Compare** sequence. The data lineage is:

```
┌─────────────────────────────────────────────┐
│ Raw dataset: UNSW_NB15_training.csv (14 MB) │
│   175,341 rows × 45 columns                 │
│   Label column: "label"  (0 = normal,       │
│                            1 = attack)       │
└────────────────────────┬────────────────────┘
                         │  Cell 3 (Part 1): pandas.read_csv
                         │  → drop ['id', 'attack_cat']
                         │  → X = df.drop('label'); y = df['label']
                         ▼
┌─────────────────────────────────────────────┐
│ Raw feature matrix X, label vector y         │
└────────────────────────┬────────────────────┘
                         │  Cell 5 (Part 1): train_test_split
                         │  test_size=0.2, random_state=42, stratify=y
                         ▼
┌─────────────────────────────────────────────┐
│ X_train_raw, X_test_raw, y_train, y_test    │
│  (80 / 20 stratified)                       │
└────────────────────────┬────────────────────┘
                         │  Cell 7 (Part 2): ColumnTransformer
                         │    num → StandardScaler
                         │    cat → OneHotEncoder(handle_unknown='ignore')
                         │    remainder='passthrough'
                         ▼
┌─────────────────────────────────────────────┐
│ X_train, X_test  (dense ndarray, 39 cols)   │
│ feature_names from preprocessor             │
└────────────────────────┬────────────────────┘
                         │  Cell 9 (Part 3): three models trained sequentially
                         ▼
┌─────────────────────────────────────────────┐
│ Baselines  (each fitted on X_train):        │
│   DecisionTreeClassifier(random_state=42)   │
│   MLPClassifier(random_state=42, max_iter=500)
│   XGBClassifier(use_label_encoder=False,    │
│                 eval_metric='logloss',      │
│                 random_state=42)            │
└────────────────────────┬────────────────────┘
                         │  Cell 11 (Part 3): evaluate on X_test
                         │  → accuracy, precision, recall, F1 (weighted), training_time
                         ▼
┌─────────────────────────────────────────────┐
│ results_df (3 rows × 6 cols)                │
│   Columns: Model, Accuracy, Precision,      │
│            Recall, F1 Score,                │
│            Training Time (s)                │
└────────────────────────┬────────────────────┘
                         │  Cell 13 (Part 4): sns.barplot (1×2)
                         ▼
┌─────────────────────────────────────────────┐
│ Two-panel bar chart:                        │
│   Accuracy (%) vs Training Time (s)         │
└────────────────────────┬────────────────────┘
                         │  Cell 15 (Part 5): shap.TreeExplainer(best_model)
                         │  shap.sample(X_test, 1000) → 1000-row SHAP values
                         ▼
┌─────────────────────────────────────────────┐
│ shap_values (1000 × 39)                     │
│ mean |SHAP| → ranked feature importance     │
└────────────────────────┬────────────────────┘
                         │  shap.summary_plot(... plot_type="bar",
                         │                     feature_names=feature_names)
                         ▼
┌─────────────────────────────────────────────┐
│ Top-10 SHAP features identified             │
└────────────────────────┬────────────────────┘
                         │  Cell 17 (Part 6):
                         │  → convert numpy → DataFrame(feature_names)
                         │  → slice to top_10_features
                         │  → train new XGBClassifier on 10-feature subset
                         ▼
┌─────────────────────────────────────────────┐
│ lightweight_model (XGB on 10 features)      │
│ Predictions, accuracy, training_time        │
└────────────────────────┬────────────────────┘
                         │  Cell 19 + 21 (Parts 7 + 8):
                         │  → summary_df table
                         │  → 3-panel bar chart (Accuracy / Time / # Features)
                         ▼
┌─────────────────────────────────────────────┐
│ Final artefacts: comparison table + plots   │
└─────────────────────────────────────────────┘
```

### 2.1 Stage-by-Stage Rationale

1. **Ingestion** — A single `pd.read_csv('UNSW_NB15_training.csv')` with column-drop `[id, attack_cat]` (only `attack_cat` was actually present in the run). This is the single source of truth for all downstream stages.
2. **Stratified Split** — `train_test_split(test_size=0.2, random_state=42, stratify=y)` preserves the attack/normal ratio across train and test (mitigates class-imbalance leakage).
3. **Preprocessing** — A robust `ColumnTransformer`:
   - **Numerics** → `StandardScaler` (mean=0, std=1) so the MLP and SHAP see features on comparable scales.
   - **Categoricals** → `OneHotEncoder(handle_unknown='ignore')` for any future-proofing against unseen categories at test time.
   - `remainder='passthrough'` preserves any non-matched columns (currently empty since the dataset is all numeric after the drop).
4. **Modeling** — Three intentionally diverse families are compared:
   - **DecisionTree** — interpretable, fast, the explainability baseline.
   - **MLPClassifier(max_iter=500)** — compact deep model (sklearn defaults: 100, single hidden layer).
   - **XGBClassifier** — strong gradient-boosted ensemble; this is the family we will explain and distil.
5. **XAI Integration** — `shap.TreeExplainer(best_model)` is chosen because XGBoost is a tree ensemble. The explainer returns a 1000×39 matrix of Shapley values — locally faithful decompositions of each prediction.
6. **Distillation (Lightweight)** — The mean `|SHAP|` per feature is sorted, the **Top-10** are selected, and a new XGBoost is trained on that 10-column subset only. This simulates an edge-deployable model that still benefits from SHAP-justified feature selection.
7. **Reporting** — A formatted summary table (Accuracy / Training Time / Feature count with % change) and a 3-panel bar chart quantify the **explainability ↔ footprint trade-off**.

---

## 3. Component Breakdown (Cell-Level)

Because the system is implemented as a Jupyter notebook rather than a Python package, the "components" are notebook cells grouped into Parts. Each Part has a single, well-defined responsibility.

### 3.1 `My_Project.ipynb` — Active Pipeline (8 Parts)

| Part | Cells | Role | Key API / Function |
|---|---|---|---|
| **Part 0** | (header) | Title / orientation | — |
| **Part 1** | Imports + Load | `pandas.read_csv`, drop auxiliary columns, stratified 80/20 split. | `train_test_split(test_size=0.2, random_state=42, stratify=y)` |
| **Part 2** | Preprocessing | Build and apply the `ColumnTransformer`. Capture `feature_names`. | `ColumnTransformer([('num', StandardScaler(), num_feats), ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats)], remainder='passthrough')` |
| **Part 3** | Baseline Training | Train DT / MLP / XGB on the same split; collect 6 metrics each (Model, Accuracy, Precision, Recall, F1 Score, Training Time). | `model.fit(...)`, `accuracy_score`, `precision_score(average='weighted')`, `recall_score(average='weighted')`, `f1_score(average='weighted')` |
| **Part 4** | Visualization: Model Comparison | Sort `results_df` by Accuracy; render side-by-side bar charts of Accuracy and Training Time. | `results_df.to_string(index=False)`, `sns.barplot` |
| **Part 5** | XAI — SHAP | Select XGBoost as `best_model`. Run `TreeExplainer` on a `shap.sample(X_test, 1000)` slice. Render a SHAP bar summary. | `shap.TreeExplainer`, `shap.sample`, `shap.summary_plot(plot_type="bar")` |
| **Part 6** | Lightweight Model | Compute mean `|SHAP|` per feature, select Top-10, train a second XGBoost on the 10-feature subset, build the formatted comparison table. | `np.abs(shap_values).mean(0)`, `XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)` |
| **Part 7** | Final Comparison Table | Build a presentable `summary_df` with `Change` column (Accuracy delta, Time delta %, Features delta %) and print it. | `pd.DataFrame`, formatted print |
| **Part 8** | Final Visualization | Three-panel bar chart — Accuracy, Training Time, Number of Features — for Full vs. Lightweight XGBoost. | `plt.subplots(1, 3, figsize=(18, 6))`, `sns.barplot` |

#### 3.1.1 Observed Results (from the executed `My_Project.ipynb` output captured in the notebook)

These are the actual numbers the previous run produced — they ground the system in real measured behavior, not theoretical claims.

| Model | Accuracy | F1 Score | Recall | Precision | Training Time (s) |
|---|---|---|---|---|---|
| **XGBoost** | **97.64 %** | **97.65 %** | **97.64 %** | **97.65 %** | 1.46 |
| Decision Tree | 96.47 % | 96.47 % | 96.47 % | 96.47 % | 2.36 |
| MLP Classifier | 95.92 % | 95.92 % | 95.92 % | 95.92 % | 69.54 |

#### 3.1.2 Top-10 SHAP Features (observed)

```
num__sttl                3.451165
num__ct_dst_src_ltm      1.460100
num__ct_dst_sport_ltm    1.348982
num__sbytes              1.056261
num__smean               0.773659
num__ct_srv_dst          0.602477
num__ct_srv_src          0.443963
num__ct_dst_ltm          0.285410
num__dbytes              0.264313
num__dmean               0.237030
```

Every one of these is a numeric flow statistic — `sttl` (source-to-destination TTL), connection counters (`ct_*`), and byte-mean features. No categorical features survive SHAP ranking, which matches what would be expected after one-hot encoding produces sparse indicator columns that XGBoost can simply ignore.

#### 3.1.3 Full vs. Lightweight XGBoost (observed)

```
                   Full XGBoost Model  Lightweight XGBoost Model  Change
Accuracy (%)                  97.64                  97.30        -0.35 %
Training Time (s)              1.46                   1.20        -17.98 %
Number of Features            39.00                  10.00        -74.36 %
```

The headline result: **cutting 74 % of features costs 0.35 percentage points of accuracy and saves 18 % of training time.** This is the empirical justification for the "lightweight XAI-NIDS" thesis.

### 3.2 `Explanaible_IDS.ipynb` — Reviewer-Suggested Stub

| Cell | Current content |
|---|---|
| 1 (markdown) | `# This is the New More Revised Project Work Suggested by the Reviewers` |
| 2 (code) | Empty |

This file is the placeholder for the multi-dataset, Top-k-sweep architecture described in the previous version of this document. **It is not yet implemented.** See **§5 Future Work** for the planned content.

### 3.3 Module / Script Inventory

The project does **not** contain any standalone Python modules (`.py` files) at the repo root. All logic lives in the notebooks. There is no `__main__.py`, no `src/` directory, no `models/` package, and no `pytest/` setup.

### 3.4 Configuration & Secrets

| Concern | Where it lives | Risk |
|---|---|---|
| Hyperparameters | Hard-coded inside notebook cells (e.g., `random_state=42`, `max_iter=500`, `use_label_encoder=False`). | None for this dataset; not externalized. |
| Dataset paths | Hard-coded relative paths (`'UNSW_NB15_training.csv'`). | Breaks if the notebook is run from a different working directory. |
| API keys / secrets | None. | N/A. |
| Random seeds | `random_state=42` is consistently applied across DT / MLP / XGB / train_test_split. | Reproducible. |

---

## 4. Execution Logic (Run Book)

The system is executed end-to-end by opening **`My_Project.ipynb`** in Jupyter or VS Code and running all cells in sequence. The exact order is:

### Step 1 — Environment Bootstrapping
```bash
cd "C:\College-Work\Projects\Explainable AI for Enhanced Network Intrusion Detection in Resource-Constrained Environments"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The previous run used the system Python at `C:\Users\Baba_\AppData\Roaming\Python\Python310\`. A future run should use the local `.venv` (now gitignored and excluded from the repo).

### Step 2 — Imports (Cell: Part 0 + Part 1)
The notebook imports: `pandas`, `numpy`, `time`, `matplotlib.pyplot`, `seaborn`, `sklearn.tree.DecisionTreeClassifier`, `sklearn.neural_network.MLPClassifier`, `xgboost.XGBClassifier`, `sklearn.metrics.{f1_score, precision_score, recall_score, accuracy_score}`, `sklearn.model_selection.train_test_split`, `sklearn.preprocessing.{StandardScaler, OneHotEncoder}`, `sklearn.compose.ColumnTransformer`, and `shap`.

### Step 3 — Data Load + Split (Cell: Part 1)
```python
df = pd.read_csv('UNSW_NB15_training.csv')          # 175,341 rows
df = df.drop(columns=['id', 'attack_cat'], axis=1)   # id absent; attack_cat dropped
X, y = df.drop('label', axis=1), df['label']
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### Step 4 — Preprocessing (Cell: Part 2)
```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)
X_train = preprocessor.fit_transform(X_train_raw)
X_test  = preprocessor.transform(X_test_raw)
feature_names = preprocessor.get_feature_names_out()
```

Output: 39-feature dense ndarray, column names preserved for SHAP.

### Step 5 — Baseline Training (Cell: Part 3)
Three models are trained sequentially on the same `X_train` / `y_train`. For each, the pipeline records `accuracy`, `precision`, `recall`, `f1` (all weighted-average × 100), and `training_time` in seconds.

### Step 6 — Model Comparison Plot (Cell: Part 4)
`results_df` is sorted by Accuracy (descending) and printed. A 1×2 figure compares Accuracy (%) and Training Time (s) across the three models.

### Step 7 — SHAP Analysis (Cell: Part 5)
```python
best_model = models["XGBoost"]
explainer = shap.TreeExplainer(best_model)
X_test_sample = shap.sample(X_test, 1000)
shap_values = explainer.shap_values(X_test_sample)
shap.summary_plot(shap_values, X_test_sample,
                  plot_type="bar",
                  feature_names=feature_names)
```

This is the XAI integration point — local explanations are produced for 1000 test rows, then aggregated into a global importance ranking.

### Step 8 — Lightweight Model Build (Cell: Part 6)
```python
vals = np.abs(shap_values).mean(0)
feature_importance = pd.DataFrame(
    list(zip(feature_names, vals)),
    columns=['feature_name', 'feature_importance_val']
).sort_values(by='feature_importance_val', ascending=False)

top_10_features = feature_importance.head(10)['feature_name'].tolist()

X_train_lightweight = X_train_df[top_10_features]
X_test_lightweight  = X_test_df[top_10_features]

lightweight_model = XGBClassifier(
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
lightweight_model.fit(X_train_lightweight, y_train)
y_pred_lightweight = lightweight_model.predict(X_test_lightweight)
```

### Step 9 — Final Comparison (Cells: Parts 7 + 8)
Print a `summary_df` with a `Change` column, then render a 1×3 bar chart (Accuracy, Training Time, Number of Features) comparing Full vs. Lightweight XGBoost.

### Step 10 — Persistence (currently absent)
There is no code today that writes a serialization (e.g., `joblib.dump`) or saves plots to `xai_nids_results/`. The `.gitkeep` in `xai_nids_results/` and the `.gitignore` patterns `xai_nids_results/*.png` / `*.csv` show the *intent* to write there. See **§5 Future Work** for the planned implementation.

### Reference Run Times (from the captured output)

| Phase | Approx. wall-clock |
|---|---|
| Data load + split | < 5 s |
| Preprocessing (StandardScaler + OHE) | seconds |
| DecisionTree train + score | ~3 s |
| XGBoost train + score | ~2 s |
| MLP train + score | ~70 s (the bottleneck) |
| SHAP on 1000-row sample | tens of seconds |
| Lightweight XGBoost train + score | ~1.5 s |
| **Total** | **~80–120 s end-to-end** on a typical laptop |

---

## 5. XAI Integration Points

The system treats **explainability as a first-class output** at exactly two integration points — local explanations and global feature selection — plus one derivative: the lightweight model itself is the explainability-driven distillation of the full one.

### 5.1 Local Explanations (per-flow)
`shap.TreeExplainer(best_model)` returns a 1000×39 matrix of Shapley values that decompose each prediction into per-feature contributions. For SOC analysts, this means every alert can in principle be justified with a list like:

```
flow:  sttl=252, sbytes=6234, ct_dst_src_ltm=12
SHAP:  sttl → +0.41   sbytes → +0.18   ct_dst_src_ltm → +0.09   …   sum → logit
```

In practice the notebook only renders the **global bar summary**; per-flow visualizations (`shap.force_plot`, `shap.waterfall_plot`) are not generated automatically.

### 5.2 Global Feature Importance
`mean(|SHAP|)` across the 1000-row sample yields the single ranked list shown in §3.1.2. This list drives both the bar chart and the Top-10 feature selection for the lightweight model.

### 5.3 Auditability
Because SHAP satisfies the **local accuracy, missingness, and consistency** properties, regulators and auditors can verify that the model's reasoning aligns with domain knowledge — e.g., high `sbytes` + high `ct_dst_ltm` + low `sttl` correlates strongly with brute-force and DoS patterns in the UNSW-NB15 taxonomy.

### 5.4 Lightweight Model as Explainability Artifact
The lightweight XGBoost is not just a smaller model — it is the *explanation-derived model*. Its 10 input features are precisely the ones the SHAP analysis flagged as the most causal contributors, making it easier to defend in a security review ("we kept these 10 features because they are the ones the model actually relies on").

---

## 6. Deployment Considerations for Resource-Constrained Targets

| Lever | Current state | Production setting |
|---|---|---|
| **Feature reduction** | Hard-coded Top-10. | Make k a config knob; sweep k ∈ {5, 10, 15, 20, 39}. |
| **Ensemble size** | Defaults accepted (XGBoost 100 trees). | Tune `n_estimators` (the previous version of this doc proposed 60; that change is **not yet committed**). |
| **Tree-only model** | XGBoost only. Drop DT and MLP for prod. | ARM-friendly, GPU-optional. |
| **Latency budget** | Not actively measured. | Add `per-flow latency (ms)` to the metric set. |
| **Explainability at the edge** | SHAP run once, offline. | Pre-compute SHAP attributions for top attack signatures; ship as a static lookup. |

---

## 7. Reproducibility Checklist

- [x] `random_state=42` everywhere (DT, MLP, XGB, train_test_split).
- [x] Stratified 80/20 split.
- [x] Single preprocessing pipeline applied to a single dataset.
- [x] All baselines evaluated with the same metric functions (`weighted` averaging).
- [x] SHAP run on a deterministic `shap.sample(X_test, 1000)` slice.
- [x] Lightweight model built on a fixed Top-10 list.
- [ ] **No `requirements.txt` pinning of exact versions** — `pip install -r requirements.txt` may pull newer XGBoost / SHAP where the `use_label_encoder` parameter has been removed (the prior captured output shows the deprecation warning).
- [ ] **No serialization** — the trained models are not saved, so each run re-trains from scratch.
- [ ] **No logging to `xai_nids_results/`** — the directory is empty.

---

## 8. Future Work (Real, Not Aspirational)

Concretely, the things this repository currently does **not** do but should:

1. **Populate `Explanaible_IDS.ipynb`** — the 12-cell multi-dataset architecture from the previous version of this doc (loaders for UNSW / NSL-KDD / CICIDS, dataset-agnostic `preprocess()` with `StandardScaler`, helper functions `measure_latency` / `compute_fpr` / `evaluate`, Top-k sweep over k ∈ {5, 10, 15, 20}, accuracy-vs-k plot, final confusion matrix). To do this, the NSL-KDD and CICIDS2017 datasets need to be added to the repo first.
2. **Wire `xai_nids_results/` for output** — save SHAP plots, confusion matrices, and serialized `joblib` models there. The `.gitignore` already excludes outputs to keep the repo lean.
3. **Pin `requirements.txt`** to specific versions known to be compatible (avoid the XGBoost `use_label_encoder` deprecation).
4. **Add explicit latency profiling** — the dataset-agnostic architecture targets edge deployment; current notebook never measures per-row inference time.
5. **Multi-class extension** — UNSW-NB15 has 9 attack categories in `attack_cat`; the current notebook collapses everything to binary, losing the category-level signal.
6. **Stream-incremental evaluation** — none of the three models has been tested against `partial_fit` or online-learning alternatives for true streaming ingestion.

---

## 9. File Manifest

A one-line manifest of the repository as it actually stands today:

```
.gitignore                            # Excludes .venv/, *.png, *.csv outputs, __pycache__
README.md                             # Workspace overview (UNSW-NB15 focus)
WORKFLOW.md                           # Git/branch recovery playbook
SYSTEM_ARCHITECTURE.md                # ★ This document (v2.0)
My_Project.ipynb                      # ★ Active 8-part pipeline
Explanaible_IDS.ipynb                 # Reviewer-suggested stub (2 cells)
Explanaible_IDS_executed.ipynb        # Generated executed copy (gitignored)
requirements.txt                      # 9 dependencies
Research_paper_my_project.docx        # Manuscript
UNSW_NB15_training.csv                # Primary dataset (~14 MB, 175,341 rows)
UNSW-NB15_features.csv                # Feature dictionary
UNSW-NB15_LIST_EVENTS.csv             # Attack-category mapping
xai_nids_results/.gitkeep             # Reserved output dir
.venv/                                # Local Python venv (gitignored)
```

---

*End of document. All claims about current behavior are sourced from the files named above, not from prior versions of this document.*
