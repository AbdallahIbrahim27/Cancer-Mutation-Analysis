"""
=============================================================================
 Cancer Type Classifier from Mutation Profiles — Complete Pipeline
 (Class-Imbalance Edition)
=============================================================================
 Data source : GDC Portal (Masked Somatic Mutation MAF files)
 Task        : Multi-class classification — predict cancer type from
               somatic mutation profile
 Models      : XGBoost (primary) + Random Forest (baseline)
 Imbalance   : SMOTE oversampling + scale_pos_weight + class_weight +
               macro-F1 optimisation + per-class threshold tuning
 Outputs     : trained model, confusion matrix, SHAP plots, predictions CSV
=============================================================================

 WHY THIS MATTERS — confusion matrix from the ORIGINAL run showed:
   • BRCA  → 129 correct  (dominant class, well-represented)
   • UVM   →   1 correct  (rare class, severely under-learned)
   • LIHC  →  33 correct  (14 samples bled into BRCA — shared mutation patterns)
   The root cause: class imbalance causing the model to optimise accuracy
   on large classes at the expense of small ones.

 WHAT CHANGED vs. ORIGINAL:
   1. SMOTE oversampling        — synthesises minority-class training samples
   2. scale_pos_weight          — XGBoost class weights inversely proportional
                                  to class frequency
   3. RandomForest class_weight — 'balanced_subsample' (more aggressive)
   4. Macro F1 as the sole      — never lets a small class be ignored
      selection criterion
   5. Per-class threshold tuning — post-hoc Youden's J threshold per class
                                   to maximise per-class recall
   6. Stratified split          — already present, kept and documented

 FOLDER STRUCTURE EXPECTED:
   data/
   ├── maf_files/
   │   ├── <uuid-1>/
   │   │   └── *.maf.gz
   │   └── ...
   └── gdc_sample_sheet.tsv

 INSTALL:
   pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
               tqdm imbalanced-learn
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report,
    confusion_matrix,
    roc_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
MAF_DIR          = './data/maf_files'
SAMPLE_SHEET     = './data/gdc_sample_sheet.tsv'
OUTPUT_DIR       = './outputs'
CLEANED_MATRIX   = './data/tcga_feature_matrix.csv'
CLEANED_LABELS   = './data/tcga_labels.csv'

MIN_MUTATION_FREQ = 0.02   # keep genes mutated in >= 2% of samples
TEST_SIZE         = 0.20   # 80 / 20 stratified split
RANDOM_STATE      = 42

# SMOTE: minimum samples a class needs before SMOTE is applied to it.
# Classes below this threshold will be synthesised up to it.
SMOTE_MIN_SAMPLES = 50

# MAF variant classifications to keep (exclude silent / synonymous)
KEEP_CLASSIFICATIONS = {
    'Missense_Mutation',
    'Nonsense_Mutation',
    'Frame_Shift_Del',
    'Frame_Shift_Ins',
    'Splice_Site',
    'In_Frame_Del',
    'In_Frame_Ins',
    'Translation_Start_Site',
    'Nonstop_Mutation'
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print(" CANCER TYPE CLASSIFIER — IMBALANCE-AWARE PIPELINE")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2. STEP 1 — PARSE MAF FILES  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def load_sample_sheet(path: str) -> dict:
    """
    Reads the GDC sample sheet.
    Returns { file_uuid -> cancer_type_short }  e.g. { '3b4c81f3-...' -> 'BRCA' }
    """
    ss = pd.read_csv(path, sep='\t')
    ss.columns = ss.columns.str.strip()

    id_col, project_col = 'File ID', 'Project ID'
    if id_col not in ss.columns or project_col not in ss.columns:
        raise ValueError(
            f"Expected columns '{id_col}' and '{project_col}' in sample sheet.\n"
            f"Found: {list(ss.columns)}"
        )

    mapping = {
        str(row[id_col]).strip(): str(row[project_col]).strip().replace('TCGA-', '')
        for _, row in ss.iterrows()
    }
    print(f"[Sample sheet] Loaded {len(mapping)} file → cancer type mappings")
    return mapping


def parse_single_maf(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            filepath, sep='\t', comment='#',
            usecols=['Tumor_Sample_Barcode', 'Hugo_Symbol', 'Variant_Classification'],
            low_memory=False, compression='gzip'
        )
        df = df[df['Variant_Classification'].isin(KEEP_CLASSIFICATIONS)]
        return df[['Tumor_Sample_Barcode', 'Hugo_Symbol']].copy()
    except Exception as e:
        print(f"  [WARN] Failed to parse {os.path.basename(filepath)}: {e}")
        return pd.DataFrame()


def parse_all_mafs(maf_dir: str, id_to_cancer: dict) -> pd.DataFrame:
    pattern   = os.path.join(maf_dir, '**', '*.maf.gz')
    all_files = glob.glob(pattern, recursive=True)

    if not all_files:
        raise FileNotFoundError(
            f"No .maf.gz files found under '{maf_dir}'.\n"
            "Make sure you extracted the GDC tar.gz into that folder."
        )

    print(f"\n[Step 1] Found {len(all_files)} MAF files — parsing...")
    records, skipped = [], 0

    for fpath in tqdm(all_files, desc='Parsing MAFs'):
        uuid        = os.path.basename(os.path.dirname(fpath))
        cancer_type = id_to_cancer.get(uuid)
        if cancer_type is None:
            skipped += 1
            continue
        df = parse_single_maf(fpath)
        if df.empty:
            skipped += 1
            continue
        df['cancer_type'] = cancer_type
        records.append(df)

    if not records:
        raise ValueError(
            "No MAF files were successfully parsed. "
            "Check that your sample sheet UUIDs match the folder names."
        )
    if skipped:
        print(f"  [WARN] Skipped {skipped} files (no sample sheet match or parse error)")

    combined = pd.concat(records, ignore_index=True)
    print(f"  Total mutation records : {len(combined):>10,}")
    print(f"  Unique patients        : {combined['Tumor_Sample_Barcode'].nunique():>10,}")
    print(f"  Unique genes           : {combined['Hugo_Symbol'].nunique():>10,}")
    print(f"  Cancer types found     : {sorted(combined['cancer_type'].unique())}")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 3. STEP 2 — BUILD BINARY FEATURE MATRIX  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(long_df: pd.DataFrame,
                         min_freq: float = MIN_MUTATION_FREQ) -> tuple:
    print(f"\n[Step 2] Building feature matrix (frequency filter >= {min_freq*100:.0f}%)...")

    deduped = (
        long_df[['Tumor_Sample_Barcode', 'Hugo_Symbol', 'cancer_type']]
        .drop_duplicates(subset=['Tumor_Sample_Barcode', 'Hugo_Symbol'])
    )

    tmb    = deduped.groupby('Tumor_Sample_Barcode')['Hugo_Symbol'].count()
    tmb.name = 'TMB'
    labels = (
        deduped.drop_duplicates('Tumor_Sample_Barcode')
               .set_index('Tumor_Sample_Barcode')['cancer_type']
    )

    matrix = deduped.pivot_table(
        index='Tumor_Sample_Barcode', columns='Hugo_Symbol',
        aggfunc='size', fill_value=0
    ).clip(upper=1).astype(np.int8)

    print(f"  Raw matrix shape       : {matrix.shape}")

    freq  = matrix.mean(axis=0)
    keep  = freq[freq >= min_freq].index
    matrix = matrix[keep]
    print(f"  After frequency filter : {matrix.shape}")

    matrix['TMB'] = tmb.reindex(matrix.index).fillna(0).astype(int)
    y = labels.reindex(matrix.index)

    print(f"  Final feature count    : {matrix.shape[1]}")
    print(f"  Patients               : {matrix.shape[0]}")
    print(f"\n  Label distribution:")
    for ct, count in y.value_counts().items():
        bar = '█' * (count // 20)
        print(f"    {ct:<8} {count:>5}  {bar}")

    return matrix, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. STEP 3 — IMBALANCE-AWARE TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(y: pd.Series):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def compute_class_weights(y_enc: np.ndarray, n_classes: int) -> dict:
    """
    Inverse-frequency weights: w_c = N / (n_classes * n_c)
    Returns dict  { class_index -> weight }
    """
    counts  = np.bincount(y_enc, minlength=n_classes).astype(float)
    n_total = len(y_enc)
    weights = {i: n_total / (n_classes * max(counts[i], 1))
               for i in range(n_classes)}
    return weights


def apply_smote(X_train: np.ndarray,
                y_train: np.ndarray,
                min_samples: int = SMOTE_MIN_SAMPLES) -> tuple:
    """
    Applies SMOTE only to classes with fewer than `min_samples` training
    samples, bringing them up to that floor.  Classes already above the
    floor are untouched, so the dominant classes are NOT over-sampled.

    SMOTE synthesises new samples by interpolating between k nearest
    neighbours in feature space — it does NOT simply duplicate rows.
    """
    counts = Counter(y_train)
    print(f"\n  [SMOTE] Class counts before resampling:")
    for cls, cnt in sorted(counts.items()):
        flag = " ← will be oversampled" if cnt < min_samples else ""
        print(f"    Class {cls:>2}: {cnt:>4} samples{flag}")

    # Build sampling_strategy: only raise classes below the floor
    strategy = {
        cls: max(cnt, min_samples)
        for cls, cnt in counts.items()
    }

    # SMOTE requires k_neighbors < n_samples_in_class.
    # For very small classes use k=1 to avoid errors.
    min_class_size = min(counts.values())
    k = max(1, min(5, min_class_size - 1))

    smote     = SMOTE(sampling_strategy=strategy,
                      k_neighbors=k,
                      random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"\n  [SMOTE] Class counts after resampling:")
    for cls, cnt in sorted(Counter(y_res).items()):
        print(f"    Class {cls:>2}: {cnt:>4} samples")
    print(f"  [SMOTE] Total training samples: {len(X_res):,}")

    return X_res, y_res


def tune_thresholds(model, X_val: np.ndarray,
                    y_val: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Per-class threshold tuning using Youden's J statistic on a validation
    set.  For each class c the threshold t* maximises:

        J(t) = TPR(t) - FPR(t)   (Youden's J, equivalent to balanced accuracy)

    This is particularly important for rare classes where the default 0.5
    threshold is too conservative — the model needs to be given a lower
    bar to fire on minority classes.

    Returns array of shape (n_classes,) with optimal thresholds.
    """
    proba      = model.predict_proba(X_val)   # (n_samples, n_classes)
    thresholds = np.full(n_classes, 0.5)

    for c in range(n_classes):
        # One-vs-rest: class c vs all others
        y_bin = (y_val == c).astype(int)

        if y_bin.sum() == 0:
            continue   # class not in val set — keep default

        fpr, tpr, thresh = roc_curve(y_bin, proba[:, c])

        # Youden's J: maximise TPR - FPR
        j_scores = tpr - fpr
        best_idx  = np.argmax(j_scores)

        # Guard against edge-case thresholds at the extremes
        t_star = float(thresh[best_idx])
        t_star = np.clip(t_star, 0.05, 0.95)

        thresholds[c] = t_star

    return thresholds


def predict_with_thresholds(proba: np.ndarray,
                             thresholds: np.ndarray) -> np.ndarray:
    """
    Applies per-class thresholds to a probability matrix.
    For each sample, selects the class whose probability exceeds its
    threshold by the greatest margin.  Falls back to argmax if no
    class clears its threshold.
    """
    margins = proba - thresholds[np.newaxis, :]   # (N, C)
    preds   = np.where(
        margins.max(axis=1) >= 0,
        margins.argmax(axis=1),
        proba.argmax(axis=1)          # fallback: standard argmax
    )
    return preds


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """
    Imbalance-aware training pipeline:
      1. Stratified 80/20 split  →  prevents leakage of SMOTE into test set
      2. SMOTE on train only     →  synthesise minority classes
      3. XGBoost with inverse-frequency class weights
      4. Random Forest with balanced_subsample weights
      5. Per-class threshold tuning on a held-out validation slice
      6. Final evaluation on the untouched test set
    """
    print("\n[Step 3] Imbalance-aware model training...")

    feature_names        = list(X.columns)
    X_arr                = X.values
    y_enc, le            = encode_labels(y)
    n_classes            = len(le.classes_)

    # ── 3a. Stratified split: train+val / test ───────────────────────────────
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_arr, y_enc,
        test_size=TEST_SIZE,
        stratify=y_enc,
        random_state=RANDOM_STATE
    )

    # Further split train_val → train / val  (val used only for threshold tuning)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.15,          # ~15% of total data
        stratify=y_train_val,
        random_state=RANDOM_STATE
    )

    print(f"\n  Data splits:")
    print(f"    Train      : {len(X_train):>5} samples")
    print(f"    Validation : {len(X_val):>5} samples  (threshold tuning only)")
    print(f"    Test       : {len(X_test):>5} samples  (final evaluation)")

    # ── 3b. SMOTE on training set only ──────────────────────────────────────
    X_train_res, y_train_res = apply_smote(X_train, y_train, SMOTE_MIN_SAMPLES)

    # ── 3c. Compute inverse-frequency class weights ──────────────────────────
    class_weights = compute_class_weights(y_train_res, n_classes)
    sample_weight = np.array([class_weights[c] for c in y_train_res])

    print(f"\n  Class weights (inverse frequency):")
    for i, ct in enumerate(le.classes_):
        print(f"    {ct:<8} w = {class_weights[i]:.3f}")

    results = {}

    # ── Model A: XGBoost ─────────────────────────────────────────────────────
    print("\n  Training XGBoost (with sample weights)...")
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,          # ← prevents overfitting on tiny SMOTE classes
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(
        X_train_res, y_train_res,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Tune thresholds on validation set
    xgb_thresholds = tune_thresholds(xgb, X_val, y_val, n_classes)
    xgb_proba      = xgb.predict_proba(X_test)
    xgb_preds      = predict_with_thresholds(xgb_proba, xgb_thresholds)

    results['XGBoost'] = {
        'model':      xgb,
        'preds':      xgb_preds,
        'proba':      xgb_proba,
        'thresholds': xgb_thresholds,
        'acc':        accuracy_score(y_test, xgb_preds),
        'f1':         f1_score(y_test, xgb_preds, average='macro')
    }

    # ── Model B: Random Forest ───────────────────────────────────────────────
    print("  Training Random Forest (balanced_subsample)...")
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        class_weight='balanced_subsample',   # recomputes weights per tree bootstrap
        min_samples_leaf=2,                  # avoids pure leaf overfitting
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train_res, y_train_res,
           sample_weight=sample_weight)

    rf_thresholds = tune_thresholds(rf, X_val, y_val, n_classes)
    rf_proba      = rf.predict_proba(X_test)
    rf_preds      = predict_with_thresholds(rf_proba, rf_thresholds)

    results['RandomForest'] = {
        'model':      rf,
        'preds':      rf_preds,
        'proba':      rf_proba,
        'thresholds': rf_thresholds,
        'acc':        accuracy_score(y_test, rf_preds),
        'f1':         f1_score(y_test, rf_preds, average='macro')
    }

    # ── Comparison ───────────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10}")
    print("─" * 45)
    for name, res in results.items():
        print(f"  {name:<20} {res['acc']:>10.4f} {res['f1']:>10.4f}")
    print("─" * 45)

    best_name = max(results, key=lambda k: results[k]['f1'])
    best      = results[best_name]
    print(f"\n  Best model : {best_name}")

    print(f"\n  Per-class thresholds ({best_name}):")
    for i, (ct, t) in enumerate(zip(le.classes_, best['thresholds'])):
        print(f"    {ct:<8} threshold = {t:.3f}")

    print(f"\n  Classification report ({best_name}):")
    print(classification_report(y_test, best['preds'], target_names=le.classes_))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    _save_confusion_matrix(y_test, best['preds'], le, best_name)

    # ── Class distribution plot ───────────────────────────────────────────────
    _save_class_distribution(y_train, y_train_res, le)

    return (best['model'], le, feature_names, X_test, y_test,
            best_name, best['thresholds'])


def _save_confusion_matrix(y_test, preds, le, model_name):
    cm  = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=le.classes_, yticklabels=le.classes_,
        cmap='Blues', ax=ax, linewidths=0.3
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix — {model_name} (Imbalance-Corrected)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def _save_class_distribution(y_before, y_after, le):
    """Saves a before/after class distribution bar chart."""
    classes   = np.arange(len(le.classes_))
    before    = np.bincount(y_before, minlength=len(le.classes_))
    after     = np.bincount(y_after,  minlength=len(le.classes_))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    for ax, counts, title, color in zip(
        axes,
        [before, after],
        ['Before SMOTE (original training set)', 'After SMOTE (resampled training set)'],
        ['#C0392B', '#2E86C1']
    ):
        ax.bar(le.classes_, counts, color=color, edgecolor='white', linewidth=0.5)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Cancer Type')
        ax.set_ylabel('Number of Samples')
        ax.tick_params(axis='x', rotation=45)
        for i, v in enumerate(counts):
            ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=8)

    plt.suptitle('Class Distribution: Before vs After SMOTE Resampling', fontsize=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'class_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. STEP 4 — SHAP EXPLAINABILITY  (unchanged logic, updated signature)
# ─────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(model, X_test, feature_names, le,
                      model_name, n_samples=500):
    print(f"\n[Step 4] Running SHAP analysis (n={n_samples} test samples)...")

    idx    = np.random.choice(len(X_test), size=min(n_samples, len(X_test)),
                              replace=False)
    X_sub  = X_test[idx]

    explainer   = shap.TreeExplainer(model,
                                     feature_perturbation='interventional')
    shap_values = explainer.shap_values(X_sub, check_additivity=False)

    n_features = len(feature_names)
    n_classes  = len(le.classes_)

    # Normalise to (C, N, F)
    if isinstance(shap_values, list):
        sv_3d = np.stack(shap_values, axis=0)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            sv_3d = (shap_values.transpose(2, 0, 1)
                     if shap_values.shape[2] == n_classes
                     else shap_values)
        elif shap_values.ndim == 2:
            sv_3d = (shap_values.T[:, np.newaxis, :]
                     if shap_values.shape == (n_features, n_classes)
                     else shap_values[np.newaxis, ...])
        else:
            sv_3d = shap_values[np.newaxis, np.newaxis, ...]
    else:
        sv_3d = np.array(shap_values)[np.newaxis, ...]

    print(f"  SHAP array shape: {sv_3d.shape}")

    # Global summary
    mean_abs_shap    = np.abs(sv_3d).mean(axis=(0, 1)).flatten()[:n_features]
    feature_importance = pd.Series(mean_abs_shap,
                                   index=feature_names).sort_values(ascending=False)
    top25 = feature_importance.head(25)

    fig, ax = plt.subplots(figsize=(10, 8))
    top25[::-1].plot(kind='barh', ax=ax, color='#378ADD')
    ax.set_xlabel('Mean |SHAP value| across all cancer types', fontsize=11)
    ax.set_title('Top 25 genes by global SHAP importance', fontsize=13)
    ax.axvline(0, color='black', linewidth=0.5)
    plt.tight_layout()
    global_path = os.path.join(OUTPUT_DIR, 'shap_global_importance.png')
    plt.savefig(global_path, dpi=150)
    plt.close()
    print(f"  Saved → {global_path}")

    # Per-class plots
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = np.array(axes).flatten()

    for i, cancer_type in enumerate(le.classes_):
        sv = sv_3d[i]
        class_importance = pd.Series(
            np.abs(sv).mean(axis=0), index=feature_names
        ).sort_values(ascending=False).head(10)

        axes[i].barh(range(len(class_importance)),
                     class_importance.values[::-1], color='#D85A30')
        axes[i].set_yticks(range(len(class_importance)))
        axes[i].set_yticklabels(class_importance.index[::-1], fontsize=9)
        axes[i].set_title(cancer_type, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Mean |SHAP|', fontsize=8)

    for j in range(n_classes, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Top 10 driver genes per cancer type (SHAP)', fontsize=14, y=1.01)
    plt.tight_layout()
    per_class_path = os.path.join(OUTPUT_DIR, 'shap_per_class.png')
    plt.savefig(per_class_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved → {per_class_path}")

    return feature_importance


# ─────────────────────────────────────────────────────────────────────────────
# 6. STEP 5 — SAVE PREDICTIONS + FEATURE IMPORTANCE CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_outputs(model, le, X_test, y_test, feature_names,
                 feature_importance, thresholds):
    print("\n[Step 5] Saving outputs...")

    proba = model.predict_proba(X_test)
    preds = predict_with_thresholds(proba, thresholds)

    pred_df = pd.DataFrame(proba, columns=[f'prob_{c}' for c in le.classes_])
    pred_df['true_label']      = le.inverse_transform(y_test)
    pred_df['predicted_label'] = le.inverse_transform(preds)
    pred_df['correct']         = pred_df['true_label'] == pred_df['predicted_label']

    pred_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved → {pred_path}")

    fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    feature_importance.reset_index() \
        .rename(columns={'index': 'gene', 0: 'shap_importance'}) \
        .to_csv(fi_path, index=False)
    print(f"  Saved → {fi_path}")

    thresh_df = pd.DataFrame({
        'cancer_type': le.classes_,
        'threshold':   thresholds
    })
    thresh_path = os.path.join(OUTPUT_DIR, 'per_class_thresholds.csv')
    thresh_df.to_csv(thresh_path, index=False)
    print(f"  Saved → {thresh_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load or build feature matrix
    if os.path.exists(CLEANED_MATRIX) and os.path.exists(CLEANED_LABELS):
        print("\n[Cache] Found pre-built feature matrix — skipping MAF parsing.")
        X = pd.read_csv(CLEANED_MATRIX, index_col=0)
        y = pd.read_csv(CLEANED_LABELS, index_col=0).squeeze()
        print(f"  Loaded: {X.shape[0]} patients × {X.shape[1]} features")
        print(f"  Cancer types: {sorted(y.unique())}")
    else:
        if not os.path.exists(SAMPLE_SHEET):
            raise FileNotFoundError(
                f"Sample sheet not found at '{SAMPLE_SHEET}'.\n"
                "Download from GDC portal → Cart → Download → Sample Sheet"
            )
        id_to_cancer = load_sample_sheet(SAMPLE_SHEET)
        long_df      = parse_all_mafs(MAF_DIR, id_to_cancer)
        X, y         = build_feature_matrix(long_df, MIN_MUTATION_FREQ)
        X.to_csv(CLEANED_MATRIX)
        y.to_csv(CLEANED_LABELS, header=True)
        print(f"\n  [Cache] Saved matrix to {CLEANED_MATRIX}")

    # Train
    (model, le, feature_names,
     X_test, y_test, model_name, thresholds) = train_and_evaluate(X, y)

    # Explain
    feature_importance = run_shap_analysis(
        model, X_test, feature_names, le, model_name
    )

    # Save
    save_outputs(model, le, X_test, y_test, feature_names,
                 feature_importance, thresholds)

    print("\n" + "=" * 65)
    print(" PIPELINE COMPLETE")
    print("=" * 65)
    print("  outputs/confusion_matrix.png       — per-class accuracy heatmap")
    print("  outputs/class_distribution.png     — before/after SMOTE chart")
    print("  outputs/shap_global_importance.png — top genes across all types")
    print("  outputs/shap_per_class.png         — top genes per cancer type")
    print("  outputs/predictions.csv            — test set predictions")
    print("  outputs/feature_importance.csv     — ranked gene importances")
    print("  outputs/per_class_thresholds.csv   — tuned decision thresholds")
    print("=" * 65)

    print("\n  Top 15 most important genes (global SHAP):")
    print(feature_importance.head(15).to_string())


if __name__ == '__main__':
    main()