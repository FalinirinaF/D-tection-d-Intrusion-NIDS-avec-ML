"""
utils.py — Fonctions utilitaires pour le système NIDS (NSL-KDD)
"""

import os
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ─────────────────────────────────────────────
# 1. Prétraitement des données
# ─────────────────────────────────────────────

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles et normalise les colonnes numériques.

    Colonnes catégorielles encodées :
        - protocol_type  (ex : tcp, udp, icmp)
        - service        (ex : http, ftp, smtp …)
        - flag           (ex : SF, S0, REJ …)

    Toutes les autres colonnes numériques (hors 'label' / 'class') sont
    standardisées avec un StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut issu du dataset NSL-KDD.

    Returns
    -------
    pd.DataFrame
        DataFrame transformé prêt pour l'inférence.
    """
    df = df.copy()

    categorical_cols = ['protocol_type', 'service', 'flag']
    label_col = 'label'   # colonne cible éventuelle

    # ── Encodage Label des colonnes catégorielles ──────────────────────────
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # ── Identification des colonnes numériques ─────────────────────────────
    exclude = categorical_cols + ([label_col] if label_col in df.columns else [])
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    # ── Mise à l'échelle (StandardScaler) ─────────────────────────────────
    if numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


# ─────────────────────────────────────────────
# 2. Métriques simulées des modèles
# ─────────────────────────────────────────────

# Scores de référence tirés d'expériences publiées sur NSL-KDD
_MODEL_METRICS = {
    "Random Forest": {
        "Accuracy": 0.9921,
        "Recall":   0.9887,
        "F1-Score": 0.9904,
        "Precision": 0.9921,
        "AUC-ROC":  0.9963,
        "Train Time (s)": 12.4,
    },
    "XGBoost": {
        "Accuracy": 0.9956,
        "Recall":   0.9931,
        "F1-Score": 0.9943,
        "Precision": 0.9956,
        "AUC-ROC":  0.9981,
        "Train Time (s)": 8.7,
    },
}


def get_model_metrics(model_name: str) -> dict:
    """
    Retourne un dictionnaire de métriques simulées pour un modèle donné.

    Parameters
    ----------
    model_name : str
        Nom du modèle — 'Random Forest' ou 'XGBoost'.

    Returns
    -------
    dict
        Dictionnaire {metric_name: value}.

    Raises
    ------
    ValueError
        Si le nom du modèle est inconnu.
    """
    if model_name not in _MODEL_METRICS:
        raise ValueError(
            f"Modèle '{model_name}' inconnu. "
            f"Choisissez parmi : {list(_MODEL_METRICS.keys())}"
        )
    return _MODEL_METRICS[model_name]


# ─────────────────────────────────────────────
# 3. Journalisation des alertes
# ─────────────────────────────────────────────

LOG_FILE = "alerts.log"


def log_alert(attack_type: str, source_ip: str = "N/A", confidence: float = 0.0) -> None:
    """
    Enregistre une alerte d'intrusion dans le fichier `alerts.log`.

    Format de la ligne :
        [YYYY-MM-DD HH:MM:SS] | ALERT | Type: <attack_type> | IP: <source_ip> | Confiance: <confidence>%

    Parameters
    ----------
    attack_type : str
        Catégorie de l'attaque détectée (ex : 'DoS', 'Probe', 'R2L', 'U2R', 'Normal').
    source_ip : str, optional
        Adresse IP source (si disponible).
    confidence : float, optional
        Score de confiance du modèle (0–1).
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level = "INFO" if attack_type.lower() == "normal" else "ALERT"
    line = (
        f"[{timestamp}] | {level} | "
        f"Type: {attack_type:<12} | "
        f"IP: {source_ip:<15} | "
        f"Confiance: {confidence * 100:.1f}%\n"
    )
    with open(LOG_FILE, "a", encoding="utf-8") as fh:
        fh.write(line)


def read_alerts(max_lines: int = 100) -> str:
    """
    Lit les dernières lignes du fichier `alerts.log`.

    Parameters
    ----------
    max_lines : int
        Nombre maximum de lignes à retourner (les plus récentes).

    Returns
    -------
    str
        Contenu du log sous forme de chaîne.
    """
    if not os.path.exists(LOG_FILE):
        return "Aucune alerte enregistrée pour le moment."
    with open(LOG_FILE, "r", encoding="utf-8") as fh:
        lines = fh.readlines()
    return "".join(lines[-max_lines:])


def clear_alerts() -> None:
    """Vide le fichier alerts.log."""
    with open(LOG_FILE, "w", encoding="utf-8") as fh:
        fh.write("")


# ─────────────────────────────────────────────
# 4. Helpers dataset NSL-KDD
# ─────────────────────────────────────────────

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty_level",
]

ATTACK_CATEGORIES = {
    "normal":    "Normal",
    "neptune":   "DoS", "back":      "DoS", "land":      "DoS",
    "pod":       "DoS", "smurf":     "DoS", "teardrop":  "DoS",
    "mailbomb":  "DoS", "apache2":   "DoS", "processtable": "DoS",
    "udpstorm":  "DoS",
    "ipsweep":   "Probe","nmap":      "Probe","portsweep": "Probe",
    "satan":     "Probe","mscan":     "Probe","saint":     "Probe",
    "ftp_write": "R2L", "guess_passwd": "R2L","imap":     "R2L",
    "multihop":  "R2L", "phf":       "R2L", "spy":       "R2L",
    "warezclient": "R2L","warezmaster": "R2L","sendmail":  "R2L",
    "named":     "R2L", "snmpgetattack": "R2L","snmpguess": "R2L",
    "xlock":     "R2L", "xsnoop":    "R2L", "worm":      "R2L",
    "buffer_overflow": "U2R","loadmodule": "U2R","perl": "U2R",
    "rootkit":   "U2R", "httptunnel": "U2R","ps":        "U2R",
    "sqlattack": "U2R", "xterm":     "U2R",
}


def map_attack_category(label: str) -> str:
    """Mappe un label NSL-KDD vers sa catégorie d'attaque."""
    return ATTACK_CATEGORIES.get(label.lower(), "Unknown")


def generate_sample_nsl_kdd(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Génère un DataFrame NSL-KDD synthétique pour les démonstrations.

    Parameters
    ----------
    n : int
        Nombre de lignes à générer.
    seed : int
        Graine aléatoire pour la reproductibilité.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)

    protocols  = rng.choice(["tcp", "udp", "icmp"], n)
    services   = rng.choice(["http", "ftp", "smtp", "ssh", "dns", "telnet"], n)
    flags      = rng.choice(["SF", "S0", "REJ", "RSTO", "SH"], n)
    labels     = rng.choice(
        ["normal", "neptune", "ipsweep", "smurf", "satan", "portsweep"],
        n, p=[0.45, 0.25, 0.10, 0.10, 0.05, 0.05]
    )

    df = pd.DataFrame({
        "duration":         rng.integers(0, 60000, n),
        "protocol_type":    protocols,
        "service":          services,
        "flag":             flags,
        "src_bytes":        rng.integers(0, 1_000_000, n),
        "dst_bytes":        rng.integers(0, 500_000, n),
        "land":             rng.integers(0, 2, n),
        "wrong_fragment":   rng.integers(0, 3, n),
        "urgent":           rng.integers(0, 2, n),
        "hot":              rng.integers(0, 30, n),
        "num_failed_logins":rng.integers(0, 5, n),
        "logged_in":        rng.integers(0, 2, n),
        "count":            rng.integers(1, 512, n),
        "srv_count":        rng.integers(1, 512, n),
        "serror_rate":      rng.uniform(0, 1, n).round(2),
        "rerror_rate":      rng.uniform(0, 1, n).round(2),
        "same_srv_rate":    rng.uniform(0, 1, n).round(2),
        "diff_srv_rate":    rng.uniform(0, 1, n).round(2),
        "dst_host_count":   rng.integers(1, 255, n),
        "dst_host_srv_count": rng.integers(1, 255, n),
        "label":            labels,
    })
    df["category"] = df["label"].map(map_attack_category)
    return df
