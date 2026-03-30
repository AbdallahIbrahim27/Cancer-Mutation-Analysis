"""
app.py — Cancer Type Classifier · Patient Sample Predictor
===========================================================
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

from predictor import (
    predict, parse_maf_file, CANCER_INFO, CANCER_TYPES,
    CANCER_GENE_SIGNATURES, KEEP_CLASSIFICATIONS
)

st.set_page_config(
    page_title="Cancer Classifier · Patient Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#080E1C;color:#E2E8F0;}
.main{background:#080E1C;}
.block-container{padding:1.5rem 2rem 3rem 2rem;max-width:1400px;}
section[data-testid="stSidebar"]{background:#0B1220;border-right:1px solid #1A2740;}
section[data-testid="stSidebar"] .block-container{padding:1.2rem 0.9rem;}
.app-header{background:linear-gradient(120deg,#091929 0%,#060F1C 60%,#091929 100%);border:1px solid #1A2740;border-left:5px solid #00D4AA;border-radius:14px;padding:1.8rem 2.2rem;margin-bottom:1.8rem;position:relative;overflow:hidden;}
.app-header h1{font-family:'IBM Plex Mono',monospace;font-size:1.8rem;font-weight:600;color:#E2E8F0;margin:0 0 .3rem;letter-spacing:-.02em;}
.app-header .sub{font-size:.82rem;color:#475569;text-transform:uppercase;letter-spacing:.07em;}
.accent{color:#00D4AA;}
.step-bar{display:flex;gap:.6rem;margin-bottom:1.6rem;flex-wrap:wrap;}
.step-pill{font-family:'IBM Plex Mono',monospace;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;padding:.35rem .9rem;border-radius:99px;border:1px solid #1A2740;color:#475569;}
.step-pill.active{background:#0A2A22;border-color:#00D4AA;color:#00D4AA;}
.step-pill.done{background:#0A1F30;border-color:#0EA5E9;color:#0EA5E9;}
.card{background:#0C1628;border:1px solid #1A2740;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1rem;}
.card-title{font-family:'IBM Plex Mono',monospace;font-size:.68rem;text-transform:uppercase;letter-spacing:.15em;color:#00D4AA;margin-bottom:.8rem;padding-bottom:.4rem;border-bottom:1px solid #1A2740;}
.result-card{border-radius:14px;padding:1.8rem 2rem;margin:1.2rem 0;border:2px solid;position:relative;overflow:hidden;}
.result-card.high{background:linear-gradient(135deg,#082218,#041410);border-color:#00D4AA;}
.result-card.medium{background:linear-gradient(135deg,#1A1605,#110F03);border-color:#FBBF24;}
.result-card.low{background:linear-gradient(135deg,#1A0606,#110303);border-color:#F87171;}
.ct-code{font-family:'IBM Plex Mono',monospace;font-size:2.8rem;font-weight:700;margin:0 0 .2rem;}
.ct-full{font-size:.95rem;color:#94A3B8;margin-bottom:1rem;}
.conf-label{font-family:'IBM Plex Mono',monospace;font-size:.68rem;text-transform:uppercase;letter-spacing:.1em;color:#475569;}
.conf-bar-bg{background:#1A2740;border-radius:99px;height:8px;margin-top:.4rem;overflow:hidden;}
.conf-bar{height:8px;border-radius:99px;}
.explain-box{background:#0B1220;border:1px solid #1A2740;border-left:3px solid #0EA5E9;border-radius:8px;padding:.85rem 1rem;font-size:.84rem;color:#94A3B8;line-height:1.6;margin:.8rem 0;}
.gene-tag{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.72rem;padding:.22rem .55rem;border-radius:4px;margin:.18rem;border:1px solid;}
.gene-tag.hit{background:#082218;color:#00D4AA;border-color:#00D4AA;}
.gene-tag.miss{background:#141E30;color:#64748B;border-color:#1A2740;}
.info-box{background:#0B1525;border:1px solid #1A2740;border-left:3px solid #0EA5E9;border-radius:8px;padding:.7rem .9rem;font-size:.82rem;color:#7FB3CC;margin-bottom:.9rem;}
.warn-box{background:#1A1408;border:1px solid #7A4D00;border-left:3px solid #FBBF24;border-radius:8px;padding:.7rem .9rem;font-size:.82rem;color:#CA8A04;margin-bottom:.9rem;}
.ok-box{background:#071A12;border:1px solid #064F32;border-left:3px solid #00D4AA;border-radius:8px;padding:.7rem .9rem;font-size:.82rem;color:#00D4AA;margin-bottom:.9rem;}
.stButton>button{background:linear-gradient(135deg,#00D4AA,#0EA5E9)!important;color:#060F1C!important;font-family:'IBM Plex Mono',monospace!important;font-weight:700!important;font-size:.82rem!important;letter-spacing:.08em!important;border:none!important;border-radius:8px!important;padding:.55rem 1.6rem!important;text-transform:uppercase!important;width:100%!important;}
div[data-testid="stMultiSelect"]>div,div[data-testid="stTextInput"]>div>div>input,div[data-testid="stNumberInput"]>div>div>input{background:#0C1628!important;border:1px solid #1A2740!important;color:#E2E8F0!important;border-radius:8px!important;font-family:'IBM Plex Mono',monospace!important;font-size:.82rem!important;}
div[data-testid="stSelectbox"]>div{background:#0C1628!important;border:1px solid #1A2740!important;}
.stRadio label{color:#94A3B8!important;font-size:.85rem!important;}
hr{border-color:#1A2740!important;}
.top3-row{display:flex;gap:.8rem;margin:.8rem 0;flex-wrap:wrap;}
.top3-card{flex:1;min-width:100px;background:#0C1628;border:1px solid #1A2740;border-radius:10px;padding:.8rem 1rem;text-align:center;}
.top3-rank{font-family:'IBM Plex Mono',monospace;font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:#475569;margin-bottom:.2rem;}
.top3-ct{font-family:'IBM Plex Mono',monospace;font-size:1.2rem;font-weight:700;margin-bottom:.15rem;}
.top3-pct{font-size:.78rem;color:#64748B;}
</style>
""", unsafe_allow_html=True)

def confidence_level(conf):
    if conf >= 0.35: return "high"
    if conf >= 0.18: return "medium"
    return "low"

def prob_bar_html(label, pct, color, bold=False):
    w = "700" if bold else "400"
    lc = "#E2E8F0" if bold else "#94A3B8"
    vc = "#00D4AA" if bold else "#475569"
    return f"""<div style="display:flex;align-items:center;gap:.7rem;padding:.28rem 0;border-bottom:1px solid #0C1628;">
<span style="font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:{lc};font-weight:{w};width:52px;flex-shrink:0;">{label}</span>
<div style="flex:1;background:#1A2740;border-radius:99px;height:{'7px' if bold else '5px'};overflow:hidden;">
<div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:99px;"></div></div>
<span style="font-family:'IBM Plex Mono',monospace;font-size:.74rem;color:{vc};width:42px;text-align:right;flex-shrink:0;">{pct:.1f}%</span></div>"""

@st.cache_data
def load_gene_list():
    fi = pd.read_csv(os.path.join(os.path.dirname(__file__), "feature_importance.csv"))
    return fi["gene"].tolist()

ALL_GENES = load_gene_list()
TOP_GENES = [g for g in ALL_GENES[:200] if g != "TMB"]

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="font-family:'IBM Plex Mono',monospace;font-size:.68rem;text-transform:uppercase;letter-spacing:.15em;color:#00D4AA;margin-bottom:1rem;padding-bottom:.5rem;border-bottom:1px solid #1A2740;">🧬 Patient Input</div>""", unsafe_allow_html=True)

    input_mode = st.radio("Input method", ["📁 Upload MAF File", "✏️ Manual Gene Entry"], label_visibility="collapsed")
    st.markdown("---")

    maf_file = None
    manual_genes = []
    manual_tmb = 0

    if input_mode == "📁 Upload MAF File":
        st.markdown('''<div class="info-box">Upload a <b>.maf</b> or <b>.maf.gz</b> file from the GDC portal. Must contain <code>Hugo_Symbol</code> and <code>Variant_Classification</code> columns.</div>''', unsafe_allow_html=True)
        maf_file = st.file_uploader("Upload MAF file", type=["maf","gz","tsv","txt"], label_visibility="collapsed")
        if maf_file:
            st.markdown(f'''<div class="ok-box">✅ Loaded: <b>{maf_file.name}</b><br>Size: {maf_file.size/1024:.1f} KB</div>''', unsafe_allow_html=True)
    else:
        st.markdown('''<div class="info-box">Select genes confirmed as somatically mutated in the patient profile.</div>''', unsafe_allow_html=True)
        manual_genes = st.multiselect("Mutated genes", options=TOP_GENES, default=["TP53","KRAS"], label_visibility="collapsed")
        extra = st.text_input("Additional genes (comma-separated)", placeholder="e.g. GNAQ, GNA11", label_visibility="collapsed")
        if extra.strip():
            manual_genes = list(set(manual_genes + [g.strip().upper() for g in extra.split(",") if g.strip()]))
        st.markdown('''<div class="card-title" style="margin-top:1rem;">TMB Value</div>''', unsafe_allow_html=True)
        manual_tmb = st.number_input("TMB", min_value=0, max_value=3000, value=120, step=5, label_visibility="collapsed")

    st.markdown("---")
    st.markdown('''<div class="card-title">Sample Presets</div>''', unsafe_allow_html=True)
    PRESETS = {
        "— select —":   ([], 0),
        "BRCA patient": (["PIK3CA","CDH1","GATA3","TP53","MAP3K1"], 80),
        "COAD patient": (["APC","KRAS","TP53","BRAF","SYNE1"],      195),
        "SKCM patient": (["BRAF","NRAS","TP53","NF1","CDKN2A"],     430),
        "KIRC patient": (["VHL","PBRM1","SETD2","BAP1"],            42),
        "LUAD patient": (["KRAS","EGFR","TP53","KEAP1","STK11"],    145),
        "UVM patient":  (["BAP1","SF3B1","EIF1AX","SYNE1"],         11),
        "GBM patient":  (["PTEN","EGFR","ATRX","TP53","NF1"],      130),
        "UCEC patient": (["PTEN","PIK3CA","PIK3R1","FBXW7"],        310),
    }
    preset = st.selectbox("Load sample", list(PRESETS.keys()), label_visibility="collapsed")
    if preset != "— select —":
        manual_genes, manual_tmb = PRESETS[preset]
        input_mode = "✏️ Manual Gene Entry"

    st.markdown("---")
    predict_btn = st.button("🔬  PREDICT CANCER TYPE")
    st.markdown("---")
    st.markdown("""<div style="font-size:.7rem;color:#2A3A50;line-height:1.8;"><b style="color:#374151;">Model:</b> XGBoost 600 trees<br><b style="color:#374151;">Features:</b> 1,410 genes + TMB<br><b style="color:#374151;">Classes:</b> 12 cancer types<br><b style="color:#374151;">Accuracy:</b> 67.0%<br><b style="color:#374151;">Macro F1:</b> 0.63<br><b style="color:#374151;">Imbalance:</b> SMOTE + Youden's J</div>""", unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
st.markdown("""<div class="app-header"><h1>🧬 Cancer Type <span class="accent">Predictor</span></h1><div class="sub">Patient Somatic Mutation Profile → Cancer Type Classification · TCGA · XGBoost · SHAP · SMOTE</div></div>""", unsafe_allow_html=True)

has_input = (maf_file is not None) or (len(manual_genes) > 0)
s1 = "active" if not has_input else "done"
s2 = "active" if has_input and not predict_btn else ("done" if predict_btn else "")
s3 = "active" if predict_btn else ""
st.markdown(f'''<div class="step-bar"><div class="step-pill {s1}">① Input Patient Data</div><div class="step-pill {s2}">② Run Prediction</div><div class="step-pill {s3}">③ View Results</div></div>''', unsafe_allow_html=True)

if predict_btn:
    if maf_file is not None:
        with st.spinner("Parsing MAF file..."):
            genes, tmb, maf_df = parse_maf_file(maf_file)
        if not genes:
            st.markdown('''<div class="warn-box">⚠️ Could not parse the MAF file. Ensure it contains <code>Hugo_Symbol</code> and <code>Variant_Classification</code> columns.</div>''', unsafe_allow_html=True)
            st.stop()
        st.markdown(f'''<div class="ok-box">✅ Parsed <b>{len(genes)}</b> unique mutated genes from <b>{maf_file.name}</b> · TMB = <b>{tmb}</b></div>''', unsafe_allow_html=True)
    else:
        genes = manual_genes
        tmb = manual_tmb if manual_tmb > 0 else len(manual_genes)
        if not genes:
            st.markdown('''<div class="warn-box">⚠️ No genes selected. Please select at least one mutated gene.</div>''', unsafe_allow_html=True)
            st.stop()

    with st.spinner("Running prediction..."):
        result = predict(genes, tmb)

    pred = result["prediction"]
    conf = result["confidence"]
    full_name = result["full_name"]
    probs = result["probabilities"]
    top3 = result["top3"]
    matched = result["matched_genes"]
    explanation = result["explanation"]
    color = CANCER_INFO[pred][1]
    clevel = confidence_level(conf)
    conf_pct = conf * 100

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(f'''<div class="result-card {clevel}"><div style="font-family:'IBM Plex Mono',monospace;font-size:.68rem;text-transform:uppercase;letter-spacing:.15em;color:#475569;margin-bottom:.4rem;">Predicted Cancer Type</div><div class="ct-code" style="color:{color};">{pred}</div><div class="ct-full">{full_name}</div><div class="conf-label">Model Confidence — {conf_pct:.1f}%</div><div class="conf-bar-bg"><div class="conf-bar" style="width:{conf_pct:.1f}%;background:{color};"></div></div></div>''', unsafe_allow_html=True)

        st.markdown('''<div class="card-title">Top 3 Predictions</div>''', unsafe_allow_html=True)
        rank_labels = ["1st","2nd","3rd"]
        t3 = '''<div class="top3-row">'''
        for i,(ct,prob) in enumerate(top3):
            c = CANCER_INFO[ct][1]
            t3 += f'''<div class="top3-card" style="border-top:3px solid {c};"><div class="top3-rank">{rank_labels[i]}</div><div class="top3-ct" style="color:{c};">{ct}</div><div class="top3-pct">{prob*100:.1f}%</div></div>'''
        t3 += "</div>"
        st.markdown(t3, unsafe_allow_html=True)

        st.markdown(f'''<div class="explain-box">💡 {explanation}</div>''', unsafe_allow_html=True)

        tmb_level = "HIGH ▲" if tmb>300 else ("MEDIUM" if tmb>80 else "LOW ▼")
        tmb_col = "#EF4444" if tmb>300 else ("#FBBF24" if tmb>80 else "#00D4AA")
        st.markdown(f'''<div class="card"><div class="card-title">Patient Input Summary</div><div style="display:flex;gap:1.5rem;margin-bottom:.8rem;flex-wrap:wrap;"><div><div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Total Genes</div><div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:#E2E8F0;">{result["n_genes_input"]}</div></div><div><div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">TMB Value</div><div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{tmb_col};">{tmb} <span style="font-size:.72rem;">{tmb_level}</span></div></div><div><div style="font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:#475569;text-transform:uppercase;letter-spacing:.1em;">Matched Drivers</div><div style="font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:700;color:{color};">{len(matched)}</div></div></div>''', unsafe_allow_html=True)

        matched_set = set(matched)
        all_input = sorted(set(g.upper() for g in genes))
        shown = all_input[:60]
        tags = "".join(f'''<span class="gene-tag {"hit" if g in matched_set else "miss"}">{g}</span>''' for g in shown)
        if len(all_input) > 60:
            tags += f'''<span class="gene-tag miss">+{len(all_input)-60} more</span>'''
        st.markdown(tags + "</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('''<div class="card-title">Probability Distribution — All 12 Cancer Types</div>''', unsafe_allow_html=True)
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        bars_html = "".join(prob_bar_html(ct, p*100, CANCER_INFO[ct][1] if ct==pred else "#1E3A5F", bold=(ct==pred)) for ct,p in sorted_probs)
        st.markdown(f'''<div class="card">{bars_html}</div>''', unsafe_allow_html=True)

        labels = [ct for ct,_ in sorted_probs]
        values = [v*100 for _,v in sorted_probs]
        bar_colors = [CANCER_INFO[ct][1] if ct==pred else "#1E3A5F" for ct in labels]

        fig, ax = plt.subplots(figsize=(6.5, 5))
        fig.patch.set_facecolor("#0C1628"); ax.set_facecolor("#0C1628")
        ax.barh(range(len(labels)), values, color=bar_colors, height=0.62, edgecolor="none")
        for i,(val) in enumerate(values):
            ax.text(val+0.2, i, f"{val:.1f}%", va="center", ha="left", fontsize=8, color="#94A3B8", fontfamily="monospace")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9, color="#CBD5E1", fontfamily="monospace")
        ax.set_xlabel("Probability (%)", fontsize=8, color="#475569")
        ax.tick_params(axis="x", colors="#475569", labelsize=7.5)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#1A2740"); ax.spines["bottom"].set_color("#1A2740")
        ax.set_xlim(0, max(values)*1.3)
        plt.tight_layout(pad=1.0)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    if maf_file is not None:
        try:
            if not maf_df.empty:
                st.markdown("---")
                st.markdown('''<div class="card-title">MAF File Preview (first 20 mutations)</div>''', unsafe_allow_html=True)
                st.dataframe(maf_df.head(20), use_container_width=True, hide_index=True)
        except Exception:
            pass

else:
    col1, col2 = st.columns([1,1], gap="large")
    with col1:
        st.markdown('''<div class="card" style="text-align:center;padding:3rem 2rem;"><div style="font-size:3rem;margin-bottom:1rem;">🔬</div><div style="font-family:'IBM Plex Mono',monospace;font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;color:#334155;">Awaiting Patient Sample</div><div style="font-size:.82rem;color:#2A3A50;margin-top:.6rem;line-height:1.6;">Upload a MAF file or manually select<br>mutated genes from the sidebar,<br>then click <b style="color:#00D4AA;">Predict Cancer Type</b>.</div></div>''', unsafe_allow_html=True)
        st.markdown('''<div class="card"><div class="card-title">Two Ways to Input a Patient Sample</div><div style="font-size:.84rem;color:#64748B;line-height:1.9;"><b style="color:#94A3B8;">📁 Option 1 — Upload MAF File</b><br>Download the patient's MAF file from the GDC portal and upload it directly. The app parses all non-silent mutations automatically.<br><br><b style="color:#94A3B8;">✏️ Option 2 — Manual Entry</b><br>Select known mutated genes and enter the TMB value. Use presets to try example profiles.</div></div>''', unsafe_allow_html=True)
    with col2:
        st.markdown('''<div class="card-title">Per-class F1-Score Reference</div>''', unsafe_allow_html=True)
        perf = [("COAD",0.82,"#F97316"),("SKCM",0.86,"#EF4444"),("KIRC",0.81,"#3B82F6"),("LUSC",0.73,"#0EA5E9"),("GBM",0.68,"#A855F7"),("LUAD",0.69,"#06B6D4"),("BRCA",0.64,"#EC4899"),("OV",0.62,"#8B5CF6"),("PRAD",0.55,"#10B981"),("LIHC",0.49,"#F59E0B"),("UCEC",0.38,"#F43F5E"),("UVM",0.29,"#D946EF")]
        fig, ax = plt.subplots(figsize=(6,4.8))
        fig.patch.set_facecolor("#0C1628"); ax.set_facecolor("#0C1628")
        lbls=[p[0] for p in perf]; f1s=[p[1] for p in perf]; cols=[p[2] for p in perf]
        ax.barh(range(len(lbls)), f1s, color=cols, height=0.62, edgecolor="none")
        for i,v in enumerate(f1s): ax.text(v+0.005, i, f"{v:.2f}", va="center", ha="left", fontsize=8, color="#94A3B8", fontfamily="monospace")
        ax.set_yticks(range(len(lbls))); ax.set_yticklabels(lbls, fontsize=9, color="#CBD5E1", fontfamily="monospace")
        ax.set_xlabel("F1-Score", fontsize=8, color="#475569")
        ax.axvline(0.63, color="#00D4AA", linewidth=1.2, linestyle="--", alpha=0.7, label="Macro avg 0.63")
        ax.tick_params(axis="x", colors="#475569", labelsize=7.5)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#1A2740"); ax.spines["bottom"].set_color("#1A2740")
        ax.set_xlim(0, 1.05); ax.legend(fontsize=7.5, framealpha=0, labelcolor="#64748B")
        plt.tight_layout(pad=1.0); st.pyplot(fig, use_container_width=True); plt.close()

st.markdown('''<div style="text-align:center;padding:2rem 0 .5rem;border-top:1px solid #1A2740;margin-top:2rem;"><div style="font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#1E2D3D;letter-spacing:.08em;">Abdullah Ibrahim & Ahmed Abdelkareem M.Sc. Candidates in Computer Science · AI Specialization · Cairo University.</div></div>''', unsafe_allow_html=True)
