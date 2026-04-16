"""
app.py — Interface Streamlit du système NIDS (NSL-KDD)
Lancer avec : streamlit run app.py
"""

import io
import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from utils import (
    preprocess_data,
    get_model_metrics,
    log_alert,
    read_alerts,
    clear_alerts,
    generate_sample_nsl_kdd,
    NSL_KDD_COLUMNS,
)

# ══════════════════════════════════════════════
# Configuration globale & Thème
# ══════════════════════════════════════════════

st.set_page_config(
    page_title="NIDS · Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialisation du thème dans la session
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Fonction pour changer le thème
def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# ── CSS Dynamique ──────────────────────────────────────────────────────────
def get_css(theme):
    if theme == "dark":
        bg_app = "#0f1117"
        bg_card = "#161b27"
        text_main = "#e2e8f0"
        border = "#2d3748"
        text_muted = "#64748b"
        tab_bg = "#161b27"
    else:
        bg_app = "#f8fafc"
        bg_card = "#ffffff"
        text_main = "#1e293b"
        border = "#e2e8f0"
        text_muted = "#475569"
        tab_bg = "#f1f5f9"

    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Syne:wght@400;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'Syne', sans-serif; }}

    .stApp {{ background: {bg_app}; color: {text_main}; }}
    .block-container {{ padding: 2rem 3rem; }}

    [data-testid="stSidebar"] {{
        background: {bg_card};
        border-right: 1px solid {border};
    }}

    .stTabs [data-baseweb="tab-list"] {{
        background: {tab_bg};
        border-radius: 12px;
        padding: 6px;
        gap: 4px;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {text_muted};
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 600;
    }}
    .stTabs [aria-selected="true"] {{
        background: #1e40af !important;
        color: #ffffff !important;
    }}

    [data-testid="metric-container"] {{
        background: {bg_card};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.2rem;
    }}
    [data-testid="metric-container"] label {{ color: {text_muted} !important; }}
    [data-testid="metric-container"] [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace;
        color: #38bdf8 !important;
    }}

    .stButton > button {{
        background: linear-gradient(135deg, #1e40af, #0ea5e9);
        color: white; border: none; border-radius: 8px;
        font-weight: 700; transition: all 0.2s;
    }}
    .stButton > button:hover {{ transform: translateY(-2px); box-shadow: 0 8px 20px rgba(14,165,233,0.35); }}

    .stTextArea textarea {{
        background: #0d1117 !important;
        color: #4ade80 !important;
        font-family: 'JetBrains Mono', monospace !important;
        border: 1px solid {border} !important;
    }}

    hr {{ border-color: {border}; }}
    h1 {{ color: {text_main}; font-weight: 800; }}
    h2 {{ color: {text_main}; font-weight: 700; }}
    h3 {{ color: {text_main}; font-weight: 600; }}

    .badge-danger {{ background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.4); border-radius: 6px; padding: 3px 10px; }}
    .badge-ok {{ background: rgba(74,222,128,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.35); border-radius: 6px; padding: 3px 10px; }}
    </style>
    """

st.markdown(get_css(st.session_state.theme), unsafe_allow_html=True)

# ── Palette matplotlib dynamique ────────────────────────────────────────────
def update_plot_params(theme):
    if theme == "dark":
        plt.rcParams.update({
            "figure.facecolor": "#161b27", "axes.facecolor": "#161b27",
            "axes.edgecolor": "#2d3748", "axes.labelcolor": "#94a3b8",
            "xtick.color": "#64748b", "ytick.color": "#64748b",
            "text.color": "#e2e8f0", "grid.color": "#2d3748"
        })
    else:
        plt.rcParams.update({
            "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#e2e8f0", "axes.labelcolor": "#475569",
            "xtick.color": "#64748b", "ytick.color": "#64748b",
            "text.color": "#1e293b", "grid.color": "#f1f5f9"
        })

update_plot_params(st.session_state.theme)

PALETTE_MAIN   = ["#1e40af", "#0ea5e9", "#38bdf8", "#7dd3fc", "#bae6fd"]
PALETTE_ATTACK = {"Normal": "#4ade80", "DoS": "#f87171",
                  "Probe": "#fb923c", "R2L": "#facc15", "U2R": "#c084fc"}

# ══════════════════════════════════════════════
# Chargement des données (mis en cache)
# ══════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_demo_data() -> pd.DataFrame:
    return generate_sample_nsl_kdd(n=500)

@st.cache_data(show_spinner=False)
def load_uploaded_csv(raw_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(raw_bytes))

# ══════════════════════════════════════════════
# En-tête principal
# ══════════════════════════════════════════════

col_logo, col_title = st.columns([1, 9])
with col_logo:
    st.markdown("<div style='font-size:3.5rem;margin-top:4px;'>🛡️</div>", unsafe_allow_html=True)
with col_title:
    st.markdown(f"""
        <h1 style='margin-bottom:0;font-size:2rem;'>NIDS — Network Intrusion Detection System</h1>
        <p style='color:#64748b;margin-top:4px;font-size:0.9rem;'>
            Machine Learning · NSL-KDD · Random Forest &amp; XGBoost
        </p>
        """, unsafe_allow_html=True)

st.markdown("---")

# ── Barre latérale ────────────────────────────────────────────────────────────
with st.sidebar:
    # Switch Thème
    theme_icon = "☀️ Light Mode" if st.session_state.theme == "dark" else "🌙 Dark Mode"
    if st.button(theme_icon, use_container_width=True):
        toggle_theme()
        st.rerun()
        
    st.markdown("### ⚙️ Paramètres")
    selected_model = st.selectbox("Modèle actif", ["Random Forest", "XGBoost"])
    threshold = st.slider("Seuil de confiance (%)", 50, 99, 75, step=1)
    st.markdown("---")
    st.markdown("""
        <p style='color:#475569;font-size:0.78rem;'>
        Dataset : <strong style='color:#94a3b8;'>NSL-KDD</strong><br>
        Features : <strong style='color:#94a3b8;'>41</strong><br>
        Classes : Normal / DoS / Probe / R2L / U2R
        </p>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# Onglets (Le reste du code reste identique à l'original)
# ══════════════════════════════════════════════

tab_eda, tab_pre, tab_models, tab_detect, tab_arch = st.tabs([
    "📊 Analyse Exploratoire",
    "⚙️ Preprocessing & Pipeline",
    "🧠 Comparaison de Modèles",
    "🔍 Détection Active",
    "📂 Architecture Système",
])

# [Note : Le contenu des onglets est identique à votre code initial, 
#  les graphiques s'adapteront automatiquement grâce à update_plot_params]

with tab_eda:
    st.subheader("📊 Analyse Exploratoire du Dataset NSL-KDD")
    st.caption("Données de démonstration générées synthétiquement (500 connexions)")
    df = load_demo_data()
    n_total = len(df)
    n_attacks = (df["category"] != "Normal").sum()
    n_normal = n_total - n_attacks
    ratio = n_attacks / n_total * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Connexions totales", f"{n_total:,}")
    k2.metric("Trafic normal", f"{n_normal:,}", f"{100-ratio:.1f}%")
    k3.metric("Attaques détectées", f"{n_attacks:,}", f"-{ratio:.1f}%", delta_color="inverse")
    k4.metric("Catégories d'attaque", "4 types")

    st.markdown("---")
 # ── Row 1 : pie + bar protocoles ─────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Répartition Normal vs Attaque")
        fig, ax = plt.subplots(figsize=(5, 4))
        cats   = df["category"].value_counts()
        colors = [PALETTE_ATTACK.get(c, "#94a3b8") for c in cats.index]
        wedges, texts, autotexts = ax.pie(
            cats.values, labels=cats.index,
            autopct="%1.1f%%", colors=colors,
            pctdistance=0.78, startangle=140,
            wedgeprops={"linewidth": 2, "edgecolor": "#0f1117"},
        )
        for at in autotexts:
            at.set_color("#0f1117")
            at.set_fontweight("bold")
            at.set_fontsize(9)
        ax.set_title("Distribution des catégories", pad=14, fontsize=11)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        st.markdown("#### Protocoles les plus fréquents")
        fig, ax = plt.subplots(figsize=(5, 4))
        proto_counts = df["protocol_type"].value_counts()
        bars = ax.bar(
            proto_counts.index, proto_counts.values,
            color=PALETTE_MAIN[:len(proto_counts)],
            edgecolor="#0f1117", linewidth=1.5,
            width=0.55, zorder=3,
        )
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.set_xlabel("Protocole")
        ax.set_ylabel("Nombre de connexions")
        ax.set_title("Fréquence par protocole", fontsize=11)
        ax.grid(axis="y", zorder=0)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 3, f"{int(h)}",
                    ha="center", va="bottom", fontsize=9, color="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Row 2 : attaques par service + heatmap ────────────────────────────────
    c3, c4 = st.columns([1, 1.4])

    with c3:
        st.markdown("#### Top 8 services — Attaques vs Normal")
        attack_df = df[df["category"] != "Normal"]
        top8 = attack_df["service"].value_counts().head(8).index
        pivot = (
            df[df["service"].isin(top8)]
            .groupby(["service", "category"])
            .size()
            .unstack(fill_value=0)
        )
        fig, ax = plt.subplots(figsize=(5, 4.5))
        pivot.plot(
            kind="barh", ax=ax, stacked=True,
            color=[PALETTE_ATTACK.get(c, "#94a3b8") for c in pivot.columns],
            edgecolor="#0f1117", linewidth=0.8,
        )
        ax.set_xlabel("Connexions")
        ax.set_ylabel("")
        ax.set_title("Attaques par service", fontsize=11)
        ax.legend(loc="lower right", fontsize=8,
                  facecolor="#161b27", edgecolor="#2d3748")
        ax.grid(axis="x", zorder=0)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c4:
        st.markdown("#### Heatmap de corrélation (features numériques)")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Réduire à 12 features pour lisibilité
        selected_cols = [
            "duration", "src_bytes", "dst_bytes", "hot",
            "num_failed_logins", "logged_in", "count", "srv_count",
            "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
        ]
        selected_cols = [c for c in selected_cols if c in df.columns]
        corr = df[selected_cols].corr()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        sns.heatmap(
            corr, ax=ax, annot=False, fmt=".2f",
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            vmin=-1, vmax=1, center=0,
            linewidths=0.4, linecolor="#0f1117",
            cbar_kws={"shrink": 0.75},
        )
        ax.set_title("Corrélations entre features", fontsize=11, pad=10)
        plt.xticks(rotation=40, ha="right", fontsize=7.5)
        plt.yticks(rotation=0, fontsize=7.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Aperçu tabulaire ─────────────────────────────────────────────────────
    with st.expander("🔎 Aperçu du DataFrame brut"):
        st.dataframe(df.head(30), use_container_width=True, height=300)


# ══════════════════════════════════════════════
# TAB 2 — Preprocessing & Pipeline
# ══════════════════════════════════════════════

with tab_pre:
    st.subheader("⚙️ Preprocessing & Pipeline ML")

    st.markdown(
        """
        Le pipeline de prétraitement (défini dans `utils.py → preprocess_data()`)
        transforme les données brutes NSL-KDD en un tenseur numérique normalisé
        prêt pour l'inférence.
        """
    )

    # ── Diagramme du pipeline ─────────────────────────────────────────────────
    steps = [
        ("1", "Chargement CSV", "Lecture du fichier brut NSL-KDD\n(41 colonnes + label)", "#1e40af"),
        ("2", "Encodage\nCatégoriel", "LabelEncoder sur\nprotocol_type · service · flag", "#0369a1"),
        ("3", "Sélection\nNumérique", "Isolation des colonnes float/int\n(hors label)", "#0c4a6e"),
        ("4", "Mise à l'échelle", "StandardScaler :\nμ=0, σ=1 sur chaque feature", "#164e63"),
        ("5", "Données prêtes", "DataFrame normalisé\nprêt pour Random Forest / XGBoost", "#065f46"),
    ]

    fig, ax = plt.subplots(figsize=(13, 2.8))
    ax.axis("off")
    n = len(steps)
    box_w, box_h = 2.0, 1.4
    gap = 0.5
    total_w = n * box_w + (n - 1) * gap
    x_start = (13 - total_w) / 2

    for i, (num, title, desc, color) in enumerate(steps):
        x = x_start + i * (box_w + gap)
        # Rectangle
        rect = plt.Rectangle((x, 0.5), box_w, box_h,
                              facecolor=color, edgecolor="#0f1117",
                              linewidth=2, zorder=2, alpha=0.9)
        ax.add_patch(rect)
        # Badge numéro
        circle = plt.Circle((x + 0.22, 1.75), 0.17,
                             color="#f8fafc", zorder=3)
        ax.add_patch(circle)
        ax.text(x + 0.22, 1.75, num, ha="center", va="center",
                fontsize=8, fontweight="bold", color="#0f1117", zorder=4)
        # Textes
        ax.text(x + box_w / 2, 1.58, title, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="#f1f5f9", zorder=3,
                linespacing=1.3)
        ax.text(x + box_w / 2, 0.94, desc, ha="center", va="center",
                fontsize=6.8, color="#cbd5e1", zorder=3, linespacing=1.35)
        # Flèche
        if i < n - 1:
            ax.annotate("", xy=(x + box_w + gap, 1.2),
                        xytext=(x + box_w, 1.2),
                        arrowprops=dict(arrowstyle="->", color="#38bdf8",
                                        lw=2), zorder=5)

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 2.5)
    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # ── Avant / Après ────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    df_demo = load_demo_data()

    with col_a:
        st.markdown("##### Données **brutes** (extrait)")
        raw_cols = ["protocol_type", "service", "flag",
                    "src_bytes", "dst_bytes", "serror_rate"]
        st.dataframe(
            df_demo[[c for c in raw_cols if c in df_demo.columns]].head(8),
            use_container_width=True,
        )

    with col_b:
        st.markdown("##### Données **prétraitées** (extrait)")
        df_proc = preprocess_data(
            df_demo.drop(columns=["label", "category"], errors="ignore")
        )
        proc_cols = ["protocol_type", "service", "flag",
                     "src_bytes", "dst_bytes", "serror_rate"]
        st.dataframe(
            df_proc[[c for c in proc_cols if c in df_proc.columns]].head(8),
            use_container_width=True,
        )

    st.markdown("---")

    # ── Extrait de code utils.py ─────────────────────────────────────────────
    st.markdown("##### 📄 Extrait de `utils.py` — fonction `preprocess_data`")
    st.code(
        '''
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    categorical_cols = ["protocol_type", "service", "flag"]
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    exclude = categorical_cols + (["label"] if "label" in df.columns else [])
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
        ''',
        language="python",
    )


# ══════════════════════════════════════════════
# TAB 3 — Comparaison de Modèles
# ══════════════════════════════════════════════

with tab_models:
    st.subheader("🧠 Comparaison des Modèles ML")
    st.caption("Métriques de référence sur le benchmark NSL-KDD (test set)")

    rf_metrics  = get_model_metrics("Random Forest")
    xgb_metrics = get_model_metrics("XGBoost")

    # ── Métriques côte à côte ─────────────────────────────────────────────────
    col_rf, col_xgb = st.columns(2)

    with col_rf:
        st.markdown(
            """
            <div style='background:#161b27;border:1px solid #2d3748;
            border-radius:12px;padding:1rem 1.2rem 0.5rem;margin-bottom:1rem;'>
            <span style='font-size:1.5rem;'>🌲</span>
            <strong style='color:#38bdf8;font-size:1.1rem;margin-left:8px;'>Random Forest</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",   f"{rf_metrics['Accuracy']*100:.2f} %")
        m2.metric("Recall",     f"{rf_metrics['Recall']*100:.2f} %")
        m3, m4 = st.columns(2)
        m3.metric("F1-Score",   f"{rf_metrics['F1-Score']*100:.2f} %")
        m4.metric("AUC-ROC",    f"{rf_metrics['AUC-ROC']*100:.2f} %")
        st.metric("Temps d'entraînement", f"{rf_metrics['Train Time (s)']} s")

    with col_xgb:
        st.markdown(
            """
            <div style='background:#161b27;border:1px solid #2d3748;
            border-radius:12px;padding:1rem 1.2rem 0.5rem;margin-bottom:1rem;'>
            <span style='font-size:1.5rem;'>⚡</span>
            <strong style='color:#fb923c;font-size:1.1rem;margin-left:8px;'>XGBoost</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("Accuracy",   f"{xgb_metrics['Accuracy']*100:.2f} %",
                  f"+{(xgb_metrics['Accuracy']-rf_metrics['Accuracy'])*100:.2f}%")
        m2.metric("Recall",     f"{xgb_metrics['Recall']*100:.2f} %",
                  f"+{(xgb_metrics['Recall']-rf_metrics['Recall'])*100:.2f}%")
        m3, m4 = st.columns(2)
        m3.metric("F1-Score",   f"{xgb_metrics['F1-Score']*100:.2f} %",
                  f"+{(xgb_metrics['F1-Score']-rf_metrics['F1-Score'])*100:.2f}%")
        m4.metric("AUC-ROC",    f"{xgb_metrics['AUC-ROC']*100:.2f} %",
                  f"+{(xgb_metrics['AUC-ROC']-rf_metrics['AUC-ROC'])*100:.2f}%")
        st.metric("Temps d'entraînement", f"{xgb_metrics['Train Time (s)']} s",
                  f"{xgb_metrics['Train Time (s)']-rf_metrics['Train Time (s)']:.1f} s")

    st.markdown("---")

    # ── Graphique radar / barres groupées ─────────────────────────────────────
    metrics_to_plot = ["Accuracy", "Recall", "F1-Score", "AUC-ROC"]
    rf_vals  = [rf_metrics[m] * 100 for m in metrics_to_plot]
    xgb_vals = [xgb_metrics[m] * 100 for m in metrics_to_plot]
    x = np.arange(len(metrics_to_plot))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    b1 = ax.bar(x - w/2, rf_vals,  w, label="Random Forest",
                color="#1e40af", edgecolor="#0f1117", linewidth=1.2)
    b2 = ax.bar(x + w/2, xgb_vals, w, label="XGBoost",
                color="#fb923c", edgecolor="#0f1117", linewidth=1.2)
    ax.set_ylim(98, 100.2)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11)
    ax.set_ylabel("Score (%)")
    ax.set_title("Comparaison des performances — Random Forest vs XGBoost", fontsize=12)
    ax.legend(facecolor="#161b27", edgecolor="#2d3748", fontsize=10)
    ax.grid(axis="y", zorder=0)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f"{h:.2f}%", ha="center", va="bottom", fontsize=8.5, color="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Tableau récapitulatif ─────────────────────────────────────────────────
    with st.expander("📋 Tableau récapitulatif complet"):
        summary = pd.DataFrame(
            {
                "Métrique":      metrics_to_plot + ["Precision", "Train Time (s)"],
                "Random Forest": [rf_metrics[m] for m in metrics_to_plot]
                                 + [rf_metrics["Precision"], rf_metrics["Train Time (s)"]],
                "XGBoost":       [xgb_metrics[m] for m in metrics_to_plot]
                                 + [xgb_metrics["Precision"], xgb_metrics["Train Time (s)"]],
            }
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# TAB 4 — Détection Active
# ══════════════════════════════════════════════

with tab_detect:
    st.subheader("🔍 Détection Active de Trafic Réseau")

    col_up, col_info = st.columns([2, 1])

    with col_up:
        uploaded = st.file_uploader(
            "Charger un fichier CSV de trafic réseau",
            type=["csv"],
            help="Format attendu : colonnes NSL-KDD (41 features + label optionnel)",
        )
    with col_info:
        st.info(
            "**Format attendu :** CSV avec colonnes NSL-KDD.\n\n"
            "Un dataset de démonstration (100 connexions) est utilisé "
            "si aucun fichier n'est chargé.",
            icon="ℹ️",
        )

    # ── Données à analyser ────────────────────────────────────────────────────
    if uploaded is not None:
        raw_bytes = uploaded.read()
        df_input  = load_uploaded_csv(raw_bytes)
        st.success(f"✅ Fichier chargé : **{uploaded.name}** — {len(df_input):,} lignes")
    else:
        df_input = generate_sample_nsl_kdd(n=100, seed=7)
        st.caption("📌 Données de démonstration utilisées (100 connexions synthétiques)")

    st.markdown("---")

    # ── Bouton d'analyse ──────────────────────────────────────────────────────
    if st.button("🚨 Analyser le trafic", use_container_width=False):
        with st.spinner("Analyse en cours …"):
            time.sleep(0.6)

            df_proc = preprocess_data(
                df_input.drop(columns=["label", "category", "difficulty_level"],
                               errors="ignore")
            )

            # Simulation d'inférence : si "label" présent on l'utilise,
            # sinon on génère des prédictions aléatoires pondérées.
            attack_cats = ["Normal", "DoS", "Probe", "R2L", "U2R"]
            if "category" in df_input.columns:
                predictions = df_input["category"].tolist()
            elif "label" in df_input.columns:
                from utils import map_attack_category
                predictions = [map_attack_category(l) for l in df_input["label"]]
            else:
                rng = np.random.default_rng(42)
                predictions = rng.choice(
                    attack_cats, len(df_input),
                    p=[0.55, 0.20, 0.12, 0.08, 0.05]
                ).tolist()

            confidences = np.random.uniform(0.70, 0.99, len(predictions)).round(3)

            # Journalisation
            clear_alerts()
            for pred, conf in zip(predictions, confidences):
                if pred != "Normal" or conf > (threshold / 100):
                    log_alert(pred, source_ip=f"10.0.{np.random.randint(0,255)}.{np.random.randint(1,254)}", confidence=conf)

            # ── Résumé ────────────────────────────────────────────────────────
            n_total_pred   = len(predictions)
            n_attack_pred  = sum(1 for p in predictions if p != "Normal")
            n_normal_pred  = n_total_pred - n_attack_pred

            r1, r2, r3 = st.columns(3)
            r1.metric("Connexions analysées", f"{n_total_pred:,}")
            r2.metric("🟢 Trafic sain",       f"{n_normal_pred:,}")
            r3.metric("🔴 Attaques détectées", f"{n_attack_pred:,}",
                      delta=f"{n_attack_pred/n_total_pred*100:.1f}%",
                      delta_color="inverse")

            # ── Tableau des résultats avec code couleur ────────────────────────
            st.markdown("##### Résultats détaillés (50 premières connexions)")
            df_result = df_input.head(50).copy()
            df_result["Prédiction"]  = predictions[:50]
            df_result["Confiance"]   = (confidences[:50] * 100).round(1)
            df_result["Statut"]      = df_result["Prédiction"].apply(
                lambda x: "🔴 DANGER" if x != "Normal" else "🟢 Sain"
            )

            display_cols = ["protocol_type", "service", "flag",
                            "src_bytes", "dst_bytes",
                            "Prédiction", "Confiance", "Statut"]
            display_cols = [c for c in display_cols if c in df_result.columns]

            # Coloration ligne par ligne
            def color_row(row):
                color = "rgba(239,68,68,0.08)" if row["Prédiction"] != "Normal" \
                        else "rgba(74,222,128,0.06)"
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                df_result[display_cols].style.apply(color_row, axis=1),
                use_container_width=True,
                height=380,
            )

    # ── Journal des alertes ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 📋 Journal des alertes en temps réel (`alerts.log`)")

    col_log, col_btn = st.columns([5, 1])
    with col_btn:
        if st.button("🗑️ Vider le log"):
            clear_alerts()
            st.rerun()

    log_content = read_alerts(max_lines=80)
    st.text_area(
        "",
        value=log_content,
        height=260,
        key="log_area",
        label_visibility="collapsed",
    )


# ══════════════════════════════════════════════
# TAB 5 — Architecture Système
# ══════════════════════════════════════════════

with tab_arch:
    st.subheader("📂 Architecture du Projet NIDS")

    col_tree, col_desc = st.columns([1, 1])

    with col_tree:
        st.markdown("##### Arborescence")
        st.code(
            """
NIDS_Project/
│
├── app.py                  # Interface Streamlit (onglets)
├── utils.py                # Fonctions ML & utilitaires
├── alerts.log              # Journal des détections (auto-généré)
│
├── data/
│   ├── KDDTrain+.txt       # Dataset d'entraînement NSL-KDD
│   ├── KDDTest+.txt        # Dataset de test NSL-KDD
│   └── sample_traffic.csv  # Exemple de trafic CSV (optionnel)
│
├── models/
│   ├── random_forest.pkl   # Modèle RF sérialisé (joblib)
│   └── xgboost.pkl         # Modèle XGBoost sérialisé
│
├── notebooks/
│   ├── 01_EDA.ipynb        # Analyse exploratoire
│   ├── 02_Training.ipynb   # Entraînement & évaluation
│   └── 03_Evaluation.ipynb # Métriques & courbes ROC
│
├── assets/
│   └── logo.png            # Logo du projet (optionnel)
│
├── requirements.txt        # Dépendances Python
└── README.md               # Documentation du projet
            """,
            language="bash",
        )

    with col_desc:
        st.markdown("##### Description des modules")
        st.markdown(
            """
| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit multi-onglets |
| `utils.py` | Preprocessing, métriques, logging |
| `alerts.log` | Historique temps réel des alertes |
| `data/` | Datasets NSL-KDD bruts |
| `models/` | Modèles ML entraînés (joblib) |
| `notebooks/` | Expérimentations Jupyter |
| `requirements.txt` | Dépendances du projet |
            """
        )

        st.markdown("##### `requirements.txt`")
        st.code(
            """
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
xgboost>=2.0.0
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.3.0
            """,
            language="text",
        )

    st.markdown("---")

    # ── Flux de données ───────────────────────────────────────────────────────
    st.markdown("##### Flux de données")
    st.code(
        """
[Fichier CSV]
     │
     ▼
[utils.preprocess_data()]
  ├─ LabelEncoder  (protocol_type, service, flag)
  └─ StandardScaler (toutes colonnes numériques)
     │
     ▼
[Modèle ML] ─── Random Forest  ──→ predict_proba()
             └── XGBoost        ──→ predict_proba()
     │
     ▼
[Résultats] ──→ DataFrame colorisé (st.dataframe)
             └─ [utils.log_alert()] ──→ alerts.log ──→ st.text_area
        """,
        language="text",
    )
