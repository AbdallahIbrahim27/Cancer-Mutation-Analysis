# 🧬 Cancer Type Classifier from Somatic Mutation Profiles

> A machine learning pipeline that predicts cancer type from patient genomic mutation data using XGBoost, Random Forest, and SHAP explainability — built on TCGA data from the GDC Portal.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Description](#2-dataset-description)
3. [Model Explanation — Why XGBoost?](#3-model-explanation--why-xgboost)
4. [Code Functionality](#4-code-functionality)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Solution Impact](#6-solution-impact)
7. [Output Files Reference](#7-output-files-reference)
8. [Quick Reference / Configuration](#8-quick-reference--configuration)

---

## 1. Project Overview

### Problem Statement

Cancer is not a single disease — it is a family of over 100 distinct diseases, each driven by unique patterns of somatic mutations in the genome. Accurately identifying the cancer type is the critical first step in determining the appropriate treatment strategy. However, conventional histopathological diagnosis can be ambiguous, particularly for **metastatic tumors of unknown primary origin (TUPO)**.

This project addresses the problem as a **supervised multi-class classification task**: given the binary somatic mutation profile of a patient (which genes are mutated), train a model to predict the primary cancer type.

> **Core question:** *Can we look at a patient's DNA mutations and automatically identify which type of cancer they have?*

### Objectives

- Parse and harmonise hundreds of Masked Somatic Mutation MAF files from the GDC portal into a single structured feature matrix
- Engineer a binary **patient × gene** feature space augmented with Tumour Mutation Burden (TMB)
- Train and rigorously compare two ensemble classifiers — **XGBoost** and **Random Forest** — selecting the best by macro F1-score
- Provide biologically meaningful model explainability using **SHAP** (SHapley Additive exPlanations) to reveal which genes drive each cancer type's prediction
- Produce reproducible, cached pipeline artifacts for downstream clinical research or academic reporting

### Scientific Significance

Beyond classification accuracy, this project contributes to the field of **computational oncology** by identifying putative driver genes per cancer type via SHAP. The gene-level interpretability output can be cross-referenced with known oncogenes and tumour suppressor genes (e.g., TP53, KRAS, BRCA1/2), bridging the gap between a data-driven ML model and established molecular biology.

---

## 2. Dataset Description

### 2.1 Source

All data is sourced from the **Genomic Data Commons (GDC) portal** maintained by the National Cancer Institute (NCI). The pipeline consumes **Masked Somatic Mutation files in MAF format** (`.maf.gz`), derived from TCGA (The Cancer Genome Atlas) projects spanning multiple cancer cohorts.

### 2.2 Data Format

| Component | Description |
|-----------|-------------|
| **MAF Files** | Tab-separated files documenting somatic variants per patient. Each row is one mutation event. |
| **Key Columns Used** | `Tumor_Sample_Barcode` (patient ID), `Hugo_Symbol` (gene name), `Variant_Classification` (mutation type) |
| **Sample Sheet** | GDC-provided TSV mapping File ID (UUID) → Project ID (e.g., `TCGA-BRCA`). Used to attach cancer type labels. |
| **File Structure** | Each UUID folder contains one `.maf.gz` file, nested under `maf_files/` |
| **Scale** | Typically 150+ MAF files covering thousands of patients across 10–33 cancer types |

### 2.3 Mutation Types Retained

Only **non-silent (protein-altering)** mutations are kept. Silent/synonymous mutations are excluded as they are unlikely to be oncogenic drivers.

| Mutation Type | Biological Significance |
|---------------|------------------------|
| `Missense_Mutation` | Changes one amino acid — most common cancer driver mutation type |
| `Nonsense_Mutation` | Introduces a premature stop codon, truncating the protein |
| `Frame_Shift_Del / Ins` | Shifts reading frame — often catastrophic for protein function |
| `Splice_Site` | Disrupts mRNA splicing, causing exon skipping or intron retention |
| `In_Frame_Del / Ins` | Small in-frame changes — may alter protein conformation |
| `Translation_Start_Site` | Prevents normal protein initiation |
| `Nonstop_Mutation` | Eliminates the stop codon, extending the protein abnormally |

### 2.4 Preprocessing Steps

**Step 1 — Label Extraction**
The GDC sample sheet maps each file UUID to its TCGA Project ID (e.g., `TCGA-BRCA` → `BRCA`). This becomes the classification target.

**Step 2 — Silent Mutation Filtering**
Synonymous mutations that do not change the amino acid sequence are excluded, as they are unlikely to be oncogenic drivers.

**Step 3 — Deduplication**
Multiple mutation events in the same gene for the same patient are collapsed to a single binary indicator (mutated = `1`, not mutated = `0`). This prevents high-TMB patients from dominating the feature space.

**Step 4 — TMB Feature**
Tumour Mutation Burden is computed as the count of uniquely mutated genes per patient *before* frequency filtering. It is appended as an additional numerical feature.

**Step 5 — Frequency Filtering**
Genes mutated in fewer than **2%** of patients across the entire cohort are removed. This eliminates extremely rare passenger mutations that add noise without predictive signal.

**Step 6 — Caching**
The resulting feature matrix (`X`) and labels (`y`) are saved to CSV. Subsequent pipeline runs skip re-parsing, making iteration significantly faster.

### 2.5 Final Feature Matrix

| Property | Value |
|----------|-------|
| **Rows** | One row per unique patient (`Tumor_Sample_Barcode`) |
| **Columns** | One binary column per gene passing frequency filter, plus TMB |
| **Typical Dimensions** | ~3,000–5,000 patients × ~500–2,000 genes (varies by cohort) |
| **Data Type** | `int8` binary (0/1) for gene columns; `int` for TMB |
| **Target Variable** | Cancer type abbreviation (e.g., BRCA, LUAD, COAD, GBM, ...) |

---

## 3. Model Explanation — Why XGBoost?

XGBoost (eXtreme Gradient Boosting) is the primary model because it consistently outperforms other algorithms on **tabular, high-dimensional, sparse binary data** — exactly the structure of our mutation matrix.

### A) Handles High-Dimensional Sparse Inputs Natively

Our feature matrix has potentially thousands of binary gene columns, the vast majority of which are `0` for any given patient (most genes are not mutated). XGBoost's internal tree construction is optimised for sparse inputs using a **sparsity-aware split finding algorithm** that efficiently skips missing and zero values — avoiding the curse of dimensionality that plagues distance-based methods like k-NN or SVM.

### B) Captures Non-Linear Gene Interactions

Cancer is driven by *combinations* of mutations — a TP53 mutation means something different in the presence of KRAS than in isolation. Decision trees inherently capture feature interactions at each split. The boosting ensemble amplifies this by iteratively focusing on misclassified samples, allowing the model to learn subtle **co-mutation patterns** that a linear classifier would miss entirely.

### C) Robustness to Overfitting

Three complementary regularisation mechanisms are employed:

- **Subsampling** (`subsample=0.8`): Each tree is trained on a random 80% of patients, introducing variance that prevents memorisation
- **Column subsampling** (`colsample_bytree=0.8`): Each tree sees only 80% of genes, analogous to feature bagging in Random Forest
- **Shrinkage** (`learning_rate=0.05`): A low learning rate means each tree contributes only a small correction, producing a smoother, more generalisable decision boundary

### D) Efficient Multi-class Support

XGBoost implements native **softmax multi-class classification** using the `mlogloss` evaluation metric. This is superior to one-vs-rest wrappers because the model jointly optimises the probability distribution across all cancer classes simultaneously, improving calibration and boundary clarity for overlapping types.

### E) Built-in SHAP Compatibility

XGBoost is natively supported by the **SHAP TreeExplainer**, which computes exact Shapley values in polynomial time. This makes the explainability step both accurate and computationally feasible at scale.

### Model Configuration

```python
XGBClassifier(
    n_estimators     = 500,        # 500 boosting rounds
    max_depth        = 6,          # tree depth — controls complexity
    learning_rate    = 0.05,       # shrinkage factor
    subsample        = 0.8,        # row subsampling per tree
    colsample_bytree = 0.8,        # column subsampling per tree
    eval_metric      = 'mlogloss', # multi-class log loss
    random_state     = 42,         # reproducibility
    n_jobs           = -1          # use all CPU cores
)
```

### Baseline: Random Forest

A Random Forest with 300 trees and `class_weight='balanced'` is trained in parallel as a baseline. The balanced class weighting is important because TCGA cohorts are not perfectly balanced — some cancer types (BRCA, LUAD) have hundreds of samples while others have fewer than 100. The final model is selected by **macro F1-score**, which treats all classes equally regardless of size.

---

## 4. Code Functionality

### 4.1 Pipeline Architecture

| Step | Function | Description |
|------|----------|-------------|
| **Step 0** | Config block | Sets all paths, hyperparameters, and thresholds in one place |
| **Step 1** | `load_sample_sheet()` + `parse_all_mafs()` | Reads and combines all `.maf.gz` files into a unified long-format DataFrame |
| **Step 2** | `build_feature_matrix()` | Pivots long-format mutations into a binary patient × gene matrix with TMB |
| **Step 3** | `train_and_evaluate()` | Splits data, trains XGBoost and RF, evaluates, generates confusion matrix |
| **Step 4** | `run_shap_analysis()` | Computes and visualises SHAP values globally and per cancer type |
| **Step 5** | `save_outputs()` | Writes predictions CSV and ranked feature importance CSV |
| **`main()`** | Orchestrator | Runs all steps with intelligent caching to skip re-parsing |

### 4.2 Installation

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn tqdm
```

### 4.3 Folder Structure

```
project/
├── cancer_classifier_pipeline.py
├── data/
│   ├── maf_files/
│   │   ├── <uuid-1>/
│   │   │   └── *.maf.gz
│   │   └── <uuid-2>/
│   │       └── *.maf.gz
│   ├── gdc_sample_sheet.tsv
│   ├── tcga_feature_matrix.csv   ← auto-generated cache
│   └── tcga_labels.csv           ← auto-generated cache
└── outputs/
    ├── confusion_matrix.png
    ├── shap_global_importance.png
    ├── shap_per_class.png
    ├── predictions.csv
    └── feature_importance.csv
```

### 4.4 Running the Pipeline

```bash
python cancer_classifier_pipeline.py
```

> **Note:** On the first run, the pipeline parses all MAF files and caches the feature matrix (this may take 10–30 minutes for large cohorts). Subsequent runs load the cached CSV directly, reducing iteration time to under 2 minutes. To force re-parsing, delete `data/tcga_feature_matrix.csv` and `data/tcga_labels.csv`.

### 4.5 Data Flow

```
Raw .maf.gz files + sample_sheet.tsv
        │
        ▼
Long-format DataFrame (patient, gene, cancer_type)
        │
        ▼
Binary patient × gene matrix + TMB column
        │
        ▼
Frequency-filtered matrix (≥2% mutation rate)
        │
        ▼
80% Train set ──── 20% Test set (stratified)
        │                   │
        ▼                   ▼
Fitted XGBoost         Predictions
    Model          Accuracy / F1 / Confusion Matrix
        │                   │
        └──────────┬─────────┘
                   ▼
            SHAP Values
     Global + Per-class Gene Plots
                   │
                   ▼
    predictions.csv + feature_importance.csv
```

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

| Metric | Definition & Rationale |
|--------|----------------------|
| **Macro F1-Score** | Harmonic mean of precision and recall, averaged equally across all classes. Chosen as the primary metric because it penalises poor performance on rare cancer types equally to common ones. |
| **Accuracy** | Fraction of correctly classified samples. Reported for reference but can be misleading with class imbalance. |
| **Per-class Precision** | Of all patients predicted as cancer type C, what fraction truly had type C? High precision = few false alarms. |
| **Per-class Recall** | Of all patients who truly had cancer type C, what fraction did the model correctly identify? High recall = few missed cases. |

### 5.2 The Confusion Matrix — In Depth

The confusion matrix is the most comprehensive single evaluation artifact for a multi-class classifier. It is a **C × C grid** where C is the number of cancer types. Each cell `(i, j)` counts the number of patients whose true cancer type was `i` but were predicted to be type `j`.

#### Reading the Matrix

| Cell Position | Meaning |
|---------------|---------|
| **Diagonal** `(i = i)` | **True Positives** — true type and predicted type match. A bright diagonal in the heatmap indicates a well-performing model. |
| **Off-diagonal row** `(i, j≠i)` | **False Negatives** for class `i` — the model failed to identify the correct type. In clinical terms: a **missed diagnosis**. |
| **Off-diagonal column** `(i≠j, j)` | **False Positives** for class `j` — the model incorrectly predicted type `j`. Can lead to unnecessary treatments. |
| **Row sums** | Total true samples per cancer type (actual class distribution in the test set) |
| **Column sums** | Total predictions per class — reveals if the model is biased toward predicting certain types |

#### Deriving Per-class Metrics

```
For cancer type C:

  TP  = confusion_matrix[C, C]
  FP  = sum(confusion_matrix[:, C]) - TP   # all predicted C, minus true C
  FN  = sum(confusion_matrix[C, :]) - TP   # all true C, minus correctly predicted
  TN  = total_samples - TP - FP - FN

  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  F1-Score  = 2 * Precision * Recall / (Precision + Recall)
```

#### Interpreting Common Confusion Patterns

- **Clean diagonal block** → The model is an excellent discriminator; each cancer type forms a distinct cluster in mutation space
- **Off-diagonal cluster between specific pairs** (e.g., LUAD ↔ LUSC) → These types share similar mutational profiles — a biologically meaningful finding, not just a modelling failure
- **Scattered misclassifications for one class** → That cancer type likely has too few training samples; consider oversampling
- **Systematic bias toward one predicted class** → Inspect column sums; the model may need `class_weight` balancing or threshold adjustment

### 5.3 SHAP as a Complementary Metric

Beyond accuracy metrics, SHAP provides a **game-theoretic measure** of each gene's contribution to each prediction. A SHAP value of `+0.3` for TP53 in a prediction of GBM means that the presence of a TP53 mutation increased the log-odds of predicting GBM by 0.3 units, holding all other features constant.

- **Global SHAP plot** — Ranks genes by mean absolute SHAP value across all samples and classes; reveals universally important mutation markers
- **Per-class SHAP plots** — For each cancer type, shows the top 10 genes most responsible for predictions of that type

---

## 6. Solution Impact

### 6.1 Technical Achievements

- **End-to-end automation** — From raw `.maf.gz` files to trained model, plots, and predictions in a single script execution
- **Scalable ingestion** — The recursive glob + tqdm parsing loop handles any number of MAF files (tested on 150+) with robust error handling for missing sample sheet entries or malformed files
- **Intelligent caching** — Once the feature matrix is built, all subsequent runs skip directly to model training (30 min → under 2 min)
- **Robust SHAP normalisation** — Custom shape-normalisation handles different SHAP output formats across library versions (list of arrays, 3D tensor, 2D matrix) without silent failures
- **Imbalance-aware evaluation** — Macro F1-score and per-class classification report are both reported, not just accuracy

### 6.2 Clinical & Research Applications

| Application | How This Pipeline Enables It |
|-------------|------------------------------|
| **Tumour of Unknown Primary (TUP)** | A patient presents with metastatic cancer but the primary site is unknown. Running their mutation profile through this classifier can suggest the most likely primary cancer type to guide biopsy and treatment. |
| **Biomarker Discovery** | The SHAP per-class gene rankings serve as a hypothesis generator — flagging candidate driver genes for wet-lab validation in each cancer type. |
| **Treatment Stratification** | Cancer types identified via mutation patterns may respond differently to targeted therapies (e.g., EGFR inhibitors in LUAD). This pipeline enables genomics-informed treatment matching. |
| **Cohort Quality Control** | Mislabelled samples in large genomic databases can be detected: if a sample labelled as BRCA is consistently classified as OV by the model, it warrants re-inspection. |
| **Academic Research** | The SHAP outputs and confusion matrix are directly publishable figures demonstrating model interpretability alongside predictive performance. |

### 6.3 Limitations

- **Binary mutation features** — The model uses only *whether* a gene is mutated, not the specific amino acid change, functional impact score, or copy number variation
- **No temporal information** — Somatic mutations accumulate over time, but the model treats each patient as a static snapshot
- **Class imbalance** — Some TCGA cancer types are underrepresented, which may inflate metrics for large classes at the expense of small ones

### 6.4 Future Directions

- **Survival prediction** — Augment mutation features with clinical metadata (age, stage) to build a Cox proportional hazards or DeepSurv model
- **Copy Number Variation (CNV)** — Integrate CNV profiles alongside mutations for a richer feature space
- **Pathway-level features** — Aggregate gene mutations by biological pathway (e.g., DNA repair, cell cycle), reducing dimensionality while improving interpretability
- **Cross-validation** — Replace single train/test split with 5-fold stratified cross-validation for more robust performance estimates
- **Hyperparameter tuning** — Apply Optuna or `BayesSearchCV` to optimise XGBoost hyperparameters systematically
- **Pan-cancer multi-omics** — Extend to RNA-seq expression, methylation profiles, or protein expression from TCGA for a multi-modal classifier

---

## 7. Output Files Reference

| File | Description |
|------|-------------|
| `outputs/confusion_matrix.png` | Heatmap showing prediction accuracy per cancer type. Diagonal = correct predictions. Colour intensity proportional to count. |
| `outputs/shap_global_importance.png` | Horizontal bar chart of top 25 genes ranked by mean absolute SHAP value across all cancer types. |
| `outputs/shap_per_class.png` | Grid of bar charts — one per cancer type — showing the top 10 most influential genes for predicting that specific type. |
| `outputs/predictions.csv` | Test set results: `true_label`, `predicted_label`, `correct` (bool), and probability score for each cancer class. |
| `outputs/feature_importance.csv` | All genes ranked by global SHAP importance score — useful for downstream biological analysis. |
| `data/tcga_feature_matrix.csv` | Cached binary patient × gene matrix. Delete to force re-parsing from raw MAF files. |
| `data/tcga_labels.csv` | Cached cancer type labels aligned to the feature matrix row index. |

---

## 8. Quick Reference / Configuration

All key parameters are defined at the top of `cancer_classifier_pipeline.py`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `MAF_DIR` | `./data/maf_files` | Root folder containing UUID subfolders with `.maf.gz` files |
| `SAMPLE_SHEET` | `./data/gdc_sample_sheet.tsv` | Downloaded from GDC Cart → Sample Sheet |
| `MIN_MUTATION_FREQ` | `0.02` | Genes must be mutated in ≥2% of samples to be included |
| `TEST_SIZE` | `0.20` | 80/20 train/test split |
| `RANDOM_STATE` | `42` | Set for reproducibility across all random operations |
| `n_estimators` (XGB) | `500` | Increase for potentially better performance, at the cost of training time |
| `learning_rate` (XGB) | `0.05` | Lower values generalise better but require more trees |
| `n_samples` (SHAP) | `500` | Test samples used for SHAP computation — increase for more stable estimates |

---

*M.Sc. Computer Science — AI Specialization | Cairo University — Faculty of Computers and Artificial Intelligence*
