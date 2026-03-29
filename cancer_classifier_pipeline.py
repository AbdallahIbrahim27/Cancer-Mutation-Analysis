"""
=============================================================================
 Cancer Type Classifier from Mutation Profiles — Complete Pipeline
=============================================================================
 Data source : GDC Portal (Masked Somatic Mutation MAF files)
 Task        : Multi-class classification — predict cancer type from
               somatic mutation profile
 Models      : XGBoost (primary) + Random Forest (baseline)
 Outputs     : trained model, confusion matrix, SHAP plots, predictions CSV
=============================================================================

 FOLDER STRUCTURE EXPECTED:
   data/
   ├── maf_files/
   │   ├── <uuid-1>/
   │   │   └── *.maf.gz
   │   ├── <uuid-2>/
   │   │   └── *.maf.gz
   │   └── ...
   └── gdc_sample_sheet.tsv

 RUN:
   pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn tqdm
   python cancer_classifier_pipeline.py
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

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import shap

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION  — edit these paths to match your setup
# ─────────────────────────────────────────────────────────────────────────────
MAF_DIR          = './data/maf_files'
SAMPLE_SHEET     = './data/gdc_sample_sheet.tsv'
OUTPUT_DIR       = './outputs'
CLEANED_MATRIX   = './data/tcga_feature_matrix.csv'
CLEANED_LABELS   = './data/tcga_labels.csv'

MIN_MUTATION_FREQ = 0.02   # keep genes mutated in >= 2% of samples
TEST_SIZE         = 0.20   # 80/20 train-test split
RANDOM_STATE      = 42

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
print(" CANCER TYPE CLASSIFIER — COMPLETE PIPELINE")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 2. STEP 1 — PARSE MAF FILES
# ─────────────────────────────────────────────────────────────────────────────
def load_sample_sheet(path: str) -> dict:
    """
    Reads the GDC sample sheet and returns a dict:
        { file_uuid -> cancer_type_short }
    e.g. { '3b4c81f3-...' -> 'BRCA' }
    """
    ss = pd.read_csv(path, sep='\t')

    # GDC sample sheet columns:
    # 'File ID', 'File Name', 'Data Category', 'Data Type',
    # 'Project ID', 'Case ID', 'Sample ID', 'Sample Type'

    # Normalize column names (strip whitespace)
    ss.columns = ss.columns.str.strip()

    id_col      = 'File ID'
    project_col = 'Project ID'

    if id_col not in ss.columns or project_col not in ss.columns:
        raise ValueError(
            f"Expected columns '{id_col}' and '{project_col}' in sample sheet.\n"
            f"Found: {list(ss.columns)}"
        )

    mapping = {}
    for _, row in ss.iterrows():
        fid    = str(row[id_col]).strip()
        cancer = str(row[project_col]).strip().replace('TCGA-', '')
        mapping[fid] = cancer

    print(f"[Sample sheet] Loaded {len(mapping)} file → cancer type mappings")
    return mapping


def parse_single_maf(filepath: str) -> pd.DataFrame:
    """
    Parses one .maf.gz file.
    Returns DataFrame with columns: Tumor_Sample_Barcode, Hugo_Symbol
    Only non-silent mutations are kept.
    """
    try:
        df = pd.read_csv(
            filepath,
            sep='\t',
            comment='#',
            usecols=['Tumor_Sample_Barcode', 'Hugo_Symbol', 'Variant_Classification'],
            low_memory=False,
            compression='gzip'
        )
        df = df[df['Variant_Classification'].isin(KEEP_CLASSIFICATIONS)]
        return df[['Tumor_Sample_Barcode', 'Hugo_Symbol']].copy()
    except Exception as e:
        print(f"  [WARN] Failed to parse {os.path.basename(filepath)}: {e}")
        return pd.DataFrame()


def parse_all_mafs(maf_dir: str, id_to_cancer: dict) -> pd.DataFrame:
    """
    Walks maf_dir, parses every .maf.gz file found,
    attaches cancer type labels, returns combined long DataFrame.
    """
    # Find all .maf.gz files (nested inside UUID subfolders)
    pattern = os.path.join(maf_dir, '**', '*.maf.gz')
    all_files = glob.glob(pattern, recursive=True)

    if not all_files:
        raise FileNotFoundError(
            f"No .maf.gz files found under '{maf_dir}'.\n"
            f"Make sure you extracted the GDC tar.gz into that folder."
        )

    print(f"\n[Step 1] Found {len(all_files)} MAF files — parsing...")

    records = []
    skipped = 0

    for fpath in tqdm(all_files, desc='Parsing MAFs'):
        # UUID is the name of the immediate parent folder
        uuid = os.path.basename(os.path.dirname(fpath))
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
# 3. STEP 2 — BUILD BINARY FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_matrix(
    long_df: pd.DataFrame,
    min_freq: float = MIN_MUTATION_FREQ
) -> tuple:
    """
    Converts long-format mutation records into a binary patient × gene matrix.

    Returns:
        X  : pd.DataFrame  — shape (n_patients, n_genes + 1)  [+1 for TMB]
        y  : pd.Series     — cancer type label per patient
    """
    print(f"\n[Step 2] Building feature matrix (frequency filter >= {min_freq*100:.0f}%)...")

    # Deduplicate: we only want to know *whether* a gene was mutated,
    # not how many times (multiple mutations in same gene → still 1)
    deduped = long_df[['Tumor_Sample_Barcode', 'Hugo_Symbol', 'cancer_type']] \
                .drop_duplicates(subset=['Tumor_Sample_Barcode', 'Hugo_Symbol'])

    # TMB = total unique genes mutated per patient (before frequency filter)
    tmb = deduped.groupby('Tumor_Sample_Barcode')['Hugo_Symbol'].count()
    tmb.name = 'TMB'

    # Extract one label per patient (all rows for same patient have same type)
    labels = deduped.drop_duplicates('Tumor_Sample_Barcode') \
                    .set_index('Tumor_Sample_Barcode')['cancer_type']

    # Pivot to binary matrix
    matrix = deduped.pivot_table(
        index='Tumor_Sample_Barcode',
        columns='Hugo_Symbol',
        aggfunc='size',
        fill_value=0
    ).clip(upper=1).astype(np.int8)

    print(f"  Raw matrix shape       : {matrix.shape}")

    # Frequency filter — drop genes mutated in < min_freq of samples
    freq = matrix.mean(axis=0)
    keep = freq[freq >= min_freq].index
    matrix = matrix[keep]
    print(f"  After frequency filter : {matrix.shape}")

    # Attach TMB as an extra numeric feature
    matrix['TMB'] = tmb.reindex(matrix.index).fillna(0).astype(int)

    # Align labels
    y = labels.reindex(matrix.index)

    print(f"  Final feature count    : {matrix.shape[1]}")
    print(f"  Patients               : {matrix.shape[0]}")
    print(f"\n  Label distribution:")
    for ct, count in y.value_counts().items():
        bar = '█' * (count // 20)
        print(f"    {ct:<8} {count:>5}  {bar}")

    return matrix, y


# ─────────────────────────────────────────────────────────────────────────────
# 4. STEP 3 — TRAIN / EVALUATE MODELS
# ─────────────────────────────────────────────────────────────────────────────
def encode_labels(y: pd.Series):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le


def train_and_evaluate(X: pd.DataFrame, y: pd.Series):
    """
    Trains XGBoost + Random Forest, evaluates on held-out test set,
    prints classification report, saves confusion matrix PNG.
    Returns: best model, label encoder, feature names, X_test, y_test
    """
    print("\n[Step 3] Training models...")

    feature_names = list(X.columns)
    X_arr = X.values
    y_enc, le = encode_labels(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_enc,
        test_size=TEST_SIZE,
        stratify=y_enc,
        random_state=RANDOM_STATE
    )
    print(f"  Train samples : {len(X_train)}")
    print(f"  Test samples  : {len(X_test)}")

    results = {}

    # ── Model A: XGBoost ────────────────────────────────────────────────────
    print("\n  Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    xgb_preds = xgb.predict(X_test)
    results['XGBoost'] = {
        'model': xgb,
        'preds': xgb_preds,
        'acc':   accuracy_score(y_test, xgb_preds),
        'f1':    f1_score(y_test, xgb_preds, average='macro')
    }

    # ── Model B: Random Forest ───────────────────────────────────────────────
    print("  Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    results['RandomForest'] = {
        'model': rf,
        'preds': rf_preds,
        'acc':   accuracy_score(y_test, rf_preds),
        'f1':    f1_score(y_test, rf_preds, average='macro')
    }

    # ── Print comparison ─────────────────────────────────────────────────────
    print("\n" + "─" * 45)
    print(f"  {'Model':<20} {'Accuracy':>10} {'Macro F1':>10}")
    print("─" * 45)
    for name, res in results.items():
        print(f"  {name:<20} {res['acc']:>10.4f} {res['f1']:>10.4f}")
    print("─" * 45)

    # Best model = highest macro F1
    best_name = max(results, key=lambda k: results[k]['f1'])
    best = results[best_name]
    print(f"\n  Best model: {best_name}")

    # ── Full classification report ───────────────────────────────────────────
    print(f"\n  Classification report ({best_name}):")
    print(classification_report(
        y_test, best['preds'],
        target_names=le.classes_
    ))

    # ── Confusion matrix ─────────────────────────────────────────────────────
    print("  Saving confusion matrix...")
    cm = confusion_matrix(y_test, best['preds'])
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True, fmt='d',
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        cmap='Blues',
        ax=ax,
        linewidths=0.3
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(f'Confusion Matrix — {best_name}', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Saved → {cm_path}")

    return best['model'], le, feature_names, X_test, y_test, best_name


# ─────────────────────────────────────────────────────────────────────────────
# 5. STEP 4 — SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────────────────
def run_shap_analysis(model, X_test, feature_names, le, model_name, n_samples=500):
    """
    Computes SHAP values and saves:
      - Global feature importance summary plot
      - Per-class top-gene bar plots (one per cancer type)
    """
    print(f"\n[Step 4] Running SHAP analysis (n={n_samples} test samples)...")

    # Use a subset for speed
    idx = np.random.choice(len(X_test), size=min(n_samples, len(X_test)), replace=False)
    X_sub = X_test[idx]

    explainer   = shap.TreeExplainer(
        model,
        feature_perturbation='interventional'
    )
    shap_values = explainer.shap_values(X_sub, check_additivity=False)

    # ── Normalise shap_values to consistent shape (n_classes, n_samples, n_features)
    # Different shap versions + model types return different shapes:
    #   A) list of C arrays, each (N, F)          → older shap + RF
    #   B) ndarray (N, F, C)                      → newer shap multi-class
    #   C) ndarray (N, F) or (F, C)               → edge cases
    n_features = len(feature_names)
    n_classes  = len(le.classes_)

    if isinstance(shap_values, list):
        # Case A: list of (N, F) → stack to (C, N, F)
        sv_3d = np.stack(shap_values, axis=0)
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            if shap_values.shape[2] == n_classes:
                # Case B: (N, F, C) → transpose to (C, N, F)
                sv_3d = shap_values.transpose(2, 0, 1)
            elif shap_values.shape[0] == n_classes:
                # already (C, N, F)
                sv_3d = shap_values
            else:
                sv_3d = shap_values.transpose(2, 0, 1)
        elif shap_values.ndim == 2:
            if shap_values.shape == (n_features, n_classes):
                # Case C: (F, C) mean importances — expand dims
                sv_3d = shap_values.T[:, np.newaxis, :]   # (C, 1, F)
            elif shap_values.shape[1] == n_features:
                # (N, F) binary — wrap
                sv_3d = shap_values[np.newaxis, ...]
            else:
                sv_3d = shap_values[np.newaxis, ...]
        else:
            sv_3d = shap_values[np.newaxis, np.newaxis, ...]
    else:
        sv_3d = np.array(shap_values)[np.newaxis, ...]

    print(f"  SHAP array shape after normalisation: {sv_3d.shape}")

    # ── Global summary (all classes, top 25 features) ────────────────────────
    print("  Generating global summary plot...")
    # mean |SHAP| across classes and samples → (n_features,)
    mean_abs_shap = np.abs(sv_3d).mean(axis=(0, 1))    # (F,)
    mean_abs_shap = mean_abs_shap.flatten()[:n_features]

    feature_importance = pd.Series(
        mean_abs_shap,
        index=feature_names
    ).sort_values(ascending=False)

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

    # ── Per-class top genes ───────────────────────────────────────────────────
    print("  Generating per-class SHAP plots...")
    cols = min(3, n_classes)
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 4))
    axes = np.array(axes).flatten()

    for i, cancer_type in enumerate(le.classes_):
        sv = sv_3d[i]                                   # (N, F)
        class_importance = pd.Series(
            np.abs(sv).mean(axis=0),
            index=feature_names
        ).sort_values(ascending=False).head(10)

        axes[i].barh(
            range(len(class_importance)),
            class_importance.values[::-1],
            color='#D85A30'
        )
        axes[i].set_yticks(range(len(class_importance)))
        axes[i].set_yticklabels(class_importance.index[::-1], fontsize=9)
        axes[i].set_title(cancer_type, fontsize=11, fontweight='bold')
        axes[i].set_xlabel('Mean |SHAP|', fontsize=8)

    # Hide any unused subplots
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
def save_outputs(model, le, X_test, y_test, feature_names, feature_importance):
    """
    Saves:
      - predictions.csv   : true label, predicted label, probabilities
      - feature_importance.csv : ranked gene importances
    """
    print("\n[Step 5] Saving outputs...")

    # Predictions with probabilities
    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)

    pred_df = pd.DataFrame(proba, columns=[f'prob_{c}' for c in le.classes_])
    pred_df['true_label']      = le.inverse_transform(y_test)
    pred_df['predicted_label'] = le.inverse_transform(preds)
    pred_df['correct']         = pred_df['true_label'] == pred_df['predicted_label']

    pred_path = os.path.join(OUTPUT_DIR, 'predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved → {pred_path}")

    # Feature importance
    fi_path = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    feature_importance.reset_index() \
        .rename(columns={'index': 'gene', 0: 'shap_importance'}) \
        .to_csv(fi_path, index=False)
    print(f"  Saved → {fi_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. MAIN — RUN EVERYTHING
# ─────────────────────────────────────────────────────────────────────────────
def main():

    # ── Check if we already have a cleaned matrix (skip re-parsing) ──────────
    if os.path.exists(CLEANED_MATRIX) and os.path.exists(CLEANED_LABELS):
        print("\n[Cache] Found pre-built feature matrix — skipping MAF parsing.")
        print(f"  Loading {CLEANED_MATRIX}...")
        X = pd.read_csv(CLEANED_MATRIX, index_col=0)
        y = pd.read_csv(CLEANED_LABELS, index_col=0).squeeze()
        print(f"  Loaded: {X.shape[0]} patients × {X.shape[1]} features")
        print(f"  Cancer types: {sorted(y.unique())}")

    else:
        # ── Parse from raw MAF files ──────────────────────────────────────────
        if not os.path.exists(SAMPLE_SHEET):
            raise FileNotFoundError(
                f"Sample sheet not found at '{SAMPLE_SHEET}'.\n"
                "Download it from GDC portal → Cart → Download → Sample Sheet"
            )

        id_to_cancer = load_sample_sheet(SAMPLE_SHEET)
        long_df      = parse_all_mafs(MAF_DIR, id_to_cancer)
        X, y         = build_feature_matrix(long_df, MIN_MUTATION_FREQ)

        # Cache to disk so next run is instant
        X.to_csv(CLEANED_MATRIX)
        y.to_csv(CLEANED_LABELS, header=True)
        print(f"\n  [Cache] Saved cleaned matrix to {CLEANED_MATRIX}")

    # ── Train & evaluate ──────────────────────────────────────────────────────
    model, le, feature_names, X_test, y_test, model_name = \
        train_and_evaluate(X, y)

    # ── SHAP explainability ───────────────────────────────────────────────────
    feature_importance = run_shap_analysis(
        model, X_test, feature_names, le, model_name
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    save_outputs(model, le, X_test, y_test, feature_names, feature_importance)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(" PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  outputs/confusion_matrix.png     — per-class accuracy heatmap")
    print(f"  outputs/shap_global_importance.png — top genes across all types")
    print(f"  outputs/shap_per_class.png         — top genes per cancer type")
    print(f"  outputs/predictions.csv            — test set predictions")
    print(f"  outputs/feature_importance.csv     — ranked gene importances")
    print("=" * 65)

    print("\n  Top 15 most important genes (global SHAP):")
    print(feature_importance.head(15).to_string())


if __name__ == '__main__':
    main()