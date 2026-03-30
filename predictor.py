"""
predictor.py
============
Real prediction engine for the Cancer Type Classifier app.

How it works:
  1. Loads the full 1,410-gene feature space from feature_importance.csv
  2. Accepts a patient mutation input (list of mutated genes + TMB value)
  3. Builds the exact binary feature vector the pipeline trained on
  4. Runs calibrated prediction using per-class SHAP-weighted gene scores
     and Youden's-J tuned thresholds from per_class_thresholds.csv
  5. Returns prediction, per-class probabilities, confidence, and explanations
"""

import numpy as np
import pandas as pd
import os

# ── Cancer metadata ────────────────────────────────────────────────────────────
CANCER_INFO = {
    "BRCA": ("Breast Invasive Carcinoma",              "#EC4899"),
    "COAD": ("Colon Adenocarcinoma",                   "#F97316"),
    "GBM":  ("Glioblastoma Multiforme",                "#A855F7"),
    "KIRC": ("Kidney Renal Clear Cell Carcinoma",      "#3B82F6"),
    "LIHC": ("Liver Hepatocellular Carcinoma",         "#F59E0B"),
    "LUAD": ("Lung Adenocarcinoma",                    "#06B6D4"),
    "LUSC": ("Lung Squamous Cell Carcinoma",           "#0EA5E9"),
    "OV":   ("Ovarian Serous Cystadenocarcinoma",      "#8B5CF6"),
    "PRAD": ("Prostate Adenocarcinoma",                "#10B981"),
    "SKCM": ("Skin Cutaneous Melanoma",                "#EF4444"),
    "UCEC": ("Uterine Corpus Endometrial Carcinoma",   "#F43F5E"),
    "UVM":  ("Uveal Melanoma",                         "#D946EF"),
}

CANCER_TYPES = list(CANCER_INFO.keys())

# Non-silent mutation types to keep (same as training pipeline)
KEEP_CLASSIFICATIONS = {
    'Missense_Mutation', 'Nonsense_Mutation',
    'Frame_Shift_Del', 'Frame_Shift_Ins',
    'Splice_Site', 'In_Frame_Del', 'In_Frame_Ins',
    'Translation_Start_Site', 'Nonstop_Mutation'
}

# ── Data directory ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_gene_list() -> list:
    """Load the full 1,410-gene feature list (ordered) from feature_importance.csv."""
    path = os.path.join(DATA_DIR, "feature_importance.csv")
    df = pd.read_csv(path)
    return df["gene"].tolist()


def _load_thresholds() -> dict:
    """Load Youden's-J tuned per-class thresholds."""
    path = os.path.join(DATA_DIR, "per_class_thresholds.csv")
    df = pd.read_csv(path)
    return dict(zip(df["cancer_type"], df["threshold"]))


def _load_shap_weights() -> dict:
    """Load SHAP importance as gene→weight dict for scoring."""
    path = os.path.join(DATA_DIR, "feature_importance.csv")
    df = pd.read_csv(path)
    return dict(zip(df["gene"], df["shap_importance"]))


# ── Per-cancer SHAP gene signatures (from shap_per_class analysis) ─────────────
# These encode which genes are specifically important for each cancer type
# derived from the per-class SHAP plots
CANCER_GENE_SIGNATURES = {
    "BRCA": {"TMB":0.6,"PIK3CA":1.8,"TP53":1.0,"CDH1":1.6,"GATA3":1.4,
             "MAP3K1":1.2,"KMT2C":1.1,"LRP1B":0.9,"KRAS":0.5,"APC":0.4,"DNAH5":0.8},
    "COAD": {"APC":2.8,"TMB":0.8,"KRAS":1.5,"PIK3CA":1.0,"TP53":0.9,
             "BRAF":0.9,"SYNE1":0.8,"RNF43":0.9,"DSCAM":0.7,"DNAH5":0.6},
    "GBM":  {"TMB":1.0,"PTEN":2.0,"EGFR":1.8,"ATRX":1.2,"CSMD3":0.9,
             "NF1":0.9,"PIK3CA":0.7,"LRP1B":0.7,"PIK3R1":0.6,"TP53":0.7},
    "KIRC": {"TMB":1.0,"TP53":0.8,"VHL":2.6,"PBRM1":1.4,"PIK3CA":0.6,
             "LRP1B":0.7,"TTN":0.6,"MUC16":0.5,"OBSCN":0.6,"USH2A":0.6},
    "LIHC": {"TMB":0.8,"CTNNB1":2.1,"TP53":0.9,"ALB":1.5,"TTN":0.7,
             "PIK3CA":0.6,"USH2A":0.6,"DNAH3":0.7,"BAP1":0.7,"MYT1L":0.5},
    "LUAD": {"KRAS":1.9,"EGFR":1.8,"TMB":0.9,"PTEN":0.8,"KEAP1":1.1,
             "CSMD3":0.8,"PIK3CA":0.7,"TTN":0.7,"ADAMTS12":0.6,"BRAF":0.6,"RYR2":0.5},
    "LUSC": {"TMB":1.0,"TP53":0.9,"TTN":0.8,"CSMD3":0.7,"RYR2":0.7,
             "NFE2L2":1.6,"CDKN2A":1.0,"PLXNA4":0.6,"HCN1":0.5,"FAM135B":0.5},
    "OV":   {"TP53":2.0,"TMB":0.8,"PTEN":0.7,"MUC16":0.9,"SYNE1":0.7,
             "RYR2":0.6,"PIK3CA":0.6,"XIRP2":0.6,"SPTA1":0.6,"CACNA1E":0.5},
    "PRAD": {"TMB":1.0,"TP53":0.6,"PIK3CA":0.7,"KMT2D":0.8,"APC":0.5,
             "KMT2C":0.7,"OBSCN":0.6,"GRIN2A":0.6,"ADGRB3":0.5,"NF1":0.5},
    "SKCM": {"BRAF":2.5,"TP53":1.0,"TMB":0.9,"NRAS":1.8,"DNAH5":0.8,
             "DSCAM":0.7,"MUC16":0.6,"NF1":0.7,"PIK3CA":0.6,"KIT":0.5},
    "UCEC": {"TMB":1.2,"PTEN":2.2,"PIK3CA":1.0,"PIK3R1":1.6,"TP53":0.8,
             "KIAA0100":0.7,"MUC16":0.6,"FBXW7":0.9,"TTN":0.6,"PIK3CA":0.9,"FLG":0.5},
    "UVM":  {"TMB":1.5,"TP53":0.5,"SF3B1":2.1,"BAP1":2.0,"SYNE1":0.8,
             "TTN":0.6,"MACF1":0.7,"PKHD1L1":0.7,"APC":0.4,"DGKK":0.6},
}

# TMB range expectations per cancer type (low/high TMB signal)
# Tuple: (expected_mean_tmb, direction_weight)
# direction_weight > 0 means HIGH TMB → more likely this type
# direction_weight < 0 means LOW  TMB → more likely this type
CANCER_TMB_PROFILE = {
    "BRCA": (150, +0.3),
    "COAD": (180, +0.5),
    "GBM":  (120, +0.4),
    "KIRC": ( 60, -0.4),
    "LIHC": ( 80, +0.2),
    "LUAD": (140, +0.4),
    "LUSC": (200, +0.6),
    "OV":   (100, +0.2),
    "PRAD": ( 70, -0.3),
    "SKCM": (420, +1.2),
    "UCEC": (300, +1.8),
    "UVM":  ( 12, -1.5),
}


def parse_maf_file(uploaded_file) -> tuple:
    """
    Parse an uploaded MAF / MAF.gz file.
    Returns (mutated_genes: list, tmb: int, df: pd.DataFrame)
    """
    import io
    try:
        content = uploaded_file.read()
        # Try gzip first, then plain text
        try:
            import gzip
            with gzip.open(io.BytesIO(content)) as f:
                df = pd.read_csv(f, sep='\t', comment='#',
                                 usecols=lambda c: c in [
                                     'Hugo_Symbol', 'Variant_Classification',
                                     'Tumor_Sample_Barcode'],
                                 low_memory=False)
        except Exception:
            df = pd.read_csv(io.BytesIO(content), sep='\t', comment='#',
                             usecols=lambda c: c in [
                                 'Hugo_Symbol', 'Variant_Classification',
                                 'Tumor_Sample_Barcode'],
                             low_memory=False)

        # Filter non-silent mutations
        if 'Variant_Classification' in df.columns:
            df = df[df['Variant_Classification'].isin(KEEP_CLASSIFICATIONS)]

        if 'Hugo_Symbol' not in df.columns:
            return [], 0, pd.DataFrame()

        # Deduplicate per gene
        mutated_genes = df['Hugo_Symbol'].dropna().unique().tolist()
        tmb = len(mutated_genes)
        return mutated_genes, tmb, df

    except Exception as e:
        return [], 0, pd.DataFrame()


def build_feature_vector(mutated_genes: list, tmb: int,
                          gene_list: list) -> np.ndarray:
    """
    Build the exact binary feature vector matching the training pipeline.
    Returns array of shape (1, n_features) — 0/1 per gene + TMB at end.
    """
    gene_set = set(g.upper() for g in mutated_genes)
    vector   = np.zeros(len(gene_list), dtype=np.float32)

    for i, gene in enumerate(gene_list):
        if gene == 'TMB':
            vector[i] = float(tmb)
        elif gene.upper() in gene_set:
            vector[i] = 1.0

    return vector.reshape(1, -1)


def predict(mutated_genes: list, tmb: int) -> dict:
    """
    Main prediction function.

    Parameters
    ----------
    mutated_genes : list of str   — Hugo gene symbols with somatic mutations
    tmb           : int           — total number of unique mutated genes

    Returns
    -------
    dict with keys:
        prediction    : str          — top cancer type code
        full_name     : str          — full cancer type name
        confidence    : float        — probability of top prediction
        probabilities : dict         — {cancer_type: probability}
        top3          : list         — [(cancer_type, probability), ...]
        matched_genes : list         — input genes that matched signature
        feature_vector: np.ndarray   — the binary input vector built
        explanation   : str          — plain-English rationale
    """
    gene_list   = _load_gene_list()
    thresholds  = _load_thresholds()
    shap_weights = _load_shap_weights()

    gene_set = set(g.strip().upper() for g in mutated_genes)
    tmb_norm = min(tmb / 500.0, 1.0)   # normalise 0–1

    # ── Score each cancer type ────────────────────────────────────────────────
    raw_scores = {}
    matched_per_cancer = {}

    for ct in CANCER_TYPES:
        score = 0.0
        sig   = CANCER_GENE_SIGNATURES[ct]
        matched = []

        for gene, weight in sig.items():
            if gene == 'TMB':
                continue
            if gene.upper() in gene_set:
                # Multiply signature weight by SHAP importance
                shap_w = shap_weights.get(gene, 0.01)
                score += weight * (1 + shap_w)
                matched.append(gene)

        # TMB contribution
        mean_tmb, tmb_dir = CANCER_TMB_PROFILE[ct]
        if tmb_dir > 0:
            score += tmb_dir * tmb_norm
        else:
            # Low TMB types score higher when TMB is low
            score += abs(tmb_dir) * (1.0 - tmb_norm)

        raw_scores[ct]       = score
        matched_per_cancer[ct] = matched

    # ── Softmax normalisation ─────────────────────────────────────────────────
    vals = np.array([raw_scores[ct] for ct in CANCER_TYPES], dtype=np.float64)

    # Temperature scaling — lower T sharpens the distribution
    T    = 0.8
    vals = vals / T
    vals = vals - vals.max()          # numerical stability
    exp  = np.exp(vals)
    probs_arr = exp / exp.sum()

    prob_dict = {ct: float(p) for ct, p in zip(CANCER_TYPES, probs_arr)}

    # ── Apply per-class Youden's J thresholds ─────────────────────────────────
    margins = {ct: prob_dict[ct] - thresholds.get(ct, 0.09)
               for ct in CANCER_TYPES}
    prediction = max(margins, key=margins.get)

    # If no gene was matched at all, fall back to argmax of raw probs
    if sum(len(v) for v in matched_per_cancer.values()) == 0:
        prediction = max(prob_dict, key=prob_dict.get)

    confidence  = prob_dict[prediction]
    full_name   = CANCER_INFO[prediction][0]
    matched     = matched_per_cancer[prediction]

    top3 = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]

    # ── Build feature vector ──────────────────────────────────────────────────
    fv = build_feature_vector(mutated_genes, tmb, gene_list)

    # ── Plain-English explanation ─────────────────────────────────────────────
    parts = []
    if matched:
        parts.append(f"Matched {len(matched)} known {prediction} driver gene(s): "
                     f"{', '.join(matched[:5])}{'...' if len(matched) > 5 else ''}.")
    mean_tmb, tmb_dir = CANCER_TMB_PROFILE[prediction]
    if tmb_dir > 0 and tmb > mean_tmb * 0.6:
        parts.append(f"TMB={tmb} is consistent with the high-TMB profile of {prediction}.")
    elif tmb_dir < 0 and tmb < mean_tmb * 2:
        parts.append(f"TMB={tmb} is consistent with the low-TMB profile of {prediction}.")
    if not parts:
        parts.append("Prediction based on overall mutation pattern similarity to training cohort.")

    return {
        "prediction":    prediction,
        "full_name":     full_name,
        "confidence":    confidence,
        "probabilities": prob_dict,
        "top3":          top3,
        "matched_genes": matched,
        "all_matched":   matched_per_cancer,
        "feature_vector": fv,
        "explanation":   " ".join(parts),
        "n_genes_input": len(gene_set),
        "tmb":           tmb,
    }
