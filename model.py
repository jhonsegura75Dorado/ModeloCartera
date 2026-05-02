"""
Modelo CatBoostClassifier para riesgo de pérdida / no compensación CB
====================================================================
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier, Pool
except ImportError as exc:
    raise ImportError(
        "No tienes instalado catboost. Instálalo con: pip install catboost"
    ) from exc

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# CONFIGURACIÓN
# ============================================================

BASE_DIR = Path(os.getenv("BASE_DIR", "_modelo_riesgo_cb"))
DATASET_PATH = Path(
    os.getenv("DATASET_MODELO", BASE_DIR / "dataset_modelo_cb_v1_limpio.parquet")
)
VARIABLES_PATH = Path(
    os.getenv("VARIABLES_MODELO", BASE_DIR / "variables_predictoras_seguras.csv")
)
PANEL_HISTORICO_PATH = Path(
    os.getenv("PANEL_HISTORICO", BASE_DIR / "panel_historico_mensual_cb.parquet")
)
ALERTAS_ACTUALES_PATH = Path(
    os.getenv("ALERTAS_ACTUALES", BASE_DIR / "alertas_actuales_riesgo_cb.parquet")
)

OUTPUT_DIR = Path(os.getenv("OUTPUT_MODELO_DIR", BASE_DIR / "modelo_catboost_v1"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Salidas limpias para producción/tablero.
# Se separan dos flujos:
#   1) PREVENTIVO: CB que todavía se pueden gestionar antes de la pérdida.
#   2) RECUPERACIÓN: CB que ya están cerrados/cerrados con saldo y requieren cartera/recuperación.
PRODUCTION_OUTPUT_DIR = Path(
    os.getenv("PRODUCTION_OUTPUT_DIR", BASE_DIR / "salidas_produccion")
)
PRODUCTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCORING_SNAPSHOT_DIR = PRODUCTION_OUTPUT_DIR / "_snapshots_scoring"
SCORING_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

PREVENTIVE_STATES = {"ACTIVO", "PROCESO DE CIERRE"}
RECOVERY_STATES = {"CERRADO", "CERRADO CON SALDO"}
TOP_K_PREVENTIVO = int(os.getenv("TOP_K_PREVENTIVO", "500"))
TOP_K_RECUPERACION = int(os.getenv("TOP_K_RECUPERACION", "1000"))

TARGET = "evento_riesgo_90d"
ID_COLS = ["codigo_punto", "fecha_corte", "periodo"]
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Umbrales de negocio a evaluar. El mejor umbral por F2 se calcula con validación.
THRESHOLDS_TO_EVALUATE = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50]
TOP_K_VALUES = [50, 100, 250, 500, 1000, 2000]
TOP_PCT_VALUES = [0.005, 0.01, 0.02, 0.05, 0.10]

# Variables que NUNCA deben entrar como predictoras.
# Aunque accidentalmente aparezcan en variables_predictoras_seguras.csv, se excluyen.
FORBIDDEN_COLUMNS = {
    "codigo_punto",
    "codigo_corresponsal",
    "codigo_dane",
    "nombre_punto",
    "fecha_corte",
    "periodo",
    "estado_actual",
    "estado_al_corte",
    "fecha_cierre",
    "fecha_castigo",
    "fecha_evento_riesgo",
    "tipo_evento_riesgo",
    "dias_hasta_evento_riesgo",
    "ya_evento_riesgo_al_corte",
    "evento_riesgo_30d",
    "evento_riesgo_60d",
    "evento_riesgo_90d",
    "valor_castigado_original",
    "saldo_castigado_pendiente",
    "monto_recuperado_castigo",
    "pct_recuperado_castigo",
    "motivo_cierre_castigado",
    "rango_mora_castigados",
    "flag_en_archivo_castigados",
    "fecha_primera_trx_mes",
    "fecha_ultima_trx_mes",
    "fecha_ultima_trx_acum",
    "fecha_ultima_compensacion_hist",
    "fecha_snapshot_saldo_usada",
    "fecha_corte_original_base",
    "fuente_base_historica",
}

# Variables que se excluyen por recomendación de modelado V1.
# No son fuga, pero no aportan o son redundantes con otras.
EXCLUDE_V1 = {
    "categoria",  # suele ser constante UNO A UNO
    "dias_sin_transacciones_limpio",  # duplica dias_sin_transacciones
    "score_riesgo_historico",  # score de reglas; usarlo como benchmark, no como predictor V1
}


# ============================================================
# UTILIDADES DE LECTURA Y VALIDACIÓN
# ============================================================


def safe_to_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def read_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No se encontró dataset: {DATASET_PATH}")
    df = pd.read_parquet(DATASET_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"El dataset no tiene el target requerido: {TARGET}")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype(int)
    if "fecha_corte" in df.columns:
        df["fecha_corte"] = pd.to_datetime(df["fecha_corte"], errors="coerce")
    return df


def read_feature_list(df: pd.DataFrame) -> List[str]:
    if not VARIABLES_PATH.exists():
        raise FileNotFoundError(f"No se encontró lista de variables: {VARIABLES_PATH}")

    vars_df = pd.read_csv(VARIABLES_PATH)
    if "variable" not in vars_df.columns:
        raise ValueError(
            "variables_predictoras_seguras.csv debe tener columna 'variable'."
        )

    features = vars_df["variable"].dropna().astype(str).tolist()
    features = [c for c in features if c in df.columns]
    features = [c for c in features if c not in FORBIDDEN_COLUMNS]
    features = [c for c in features if c not in EXCLUDE_V1]
    features = [c for c in features if not c.startswith("fecha_")]
    features = [c for c in features if not c.startswith("evento_")]
    features = [c for c in features if not c.startswith("dias_hasta_evento")]

    if not features:
        raise ValueError(
            "No quedaron variables predictoras válidas después de filtros."
        )

    return sorted(set(features))


def infer_categorical_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    cat_features = []
    for col in features:
        if col not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(
            df[col]
        ):
            cat_features.append(col)
        elif str(df[col].dtype).startswith("string"):
            cat_features.append(col)
    return cat_features


def clean_features_for_catboost(
    df: pd.DataFrame, features: List[str], cat_features: List[str]
) -> pd.DataFrame:
    X = df[features].copy()

    for col in features:
        if col in cat_features:
            X[col] = X[col].astype("string").fillna("__MISSING__").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            # CatBoost maneja NaN en numéricas, pero se reemplazan inf/-inf.
            X[col] = X[col].replace([np.inf, -np.inf], np.nan)

    return X


def validate_dataset(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    rows = []

    rows.append({"validacion": "filas", "valor": len(df)})
    rows.append({"validacion": "columnas", "valor": df.shape[1]})
    rows.append(
        {
            "validacion": "cb_unicos",
            "valor": (
                df["codigo_punto"].nunique() if "codigo_punto" in df.columns else np.nan
            ),
        }
    )
    rows.append({"validacion": "target_positivos", "valor": int(df[TARGET].sum())})
    rows.append({"validacion": "target_tasa", "valor": float(df[TARGET].mean())})
    rows.append({"validacion": "variables_modelo", "valor": len(features)})

    if {"codigo_punto", "fecha_corte"}.issubset(df.columns):
        dup = int(df.duplicated(["codigo_punto", "fecha_corte"]).sum())
        rows.append({"validacion": "duplicados_codigo_fecha", "valor": dup})

    leakage_present = sorted([c for c in FORBIDDEN_COLUMNS if c in features])
    rows.append(
        {"validacion": "variables_fuga_en_features", "valor": len(leakage_present)}
    )
    rows.append(
        {
            "validacion": "lista_variables_fuga_en_features",
            "valor": ", ".join(leakage_present),
        }
    )

    if "fecha_corte" in df.columns:
        rows.append({"validacion": "fecha_min", "valor": str(df["fecha_corte"].min())})
        rows.append({"validacion": "fecha_max", "valor": str(df["fecha_corte"].max())})
        rows.append(
            {
                "validacion": "periodos",
                "valor": int(df["fecha_corte"].dt.to_period("M").nunique()),
            }
        )

    estado = "OK_PARA_ENTRENAR"
    if len(df) < 100_000:
        estado = "REVISAR_POCAS_FILAS"
    if df[TARGET].sum() < 500:
        estado = "REVISAR_POCOS_EVENTOS"
    if df[TARGET].nunique() < 2:
        estado = "NO_ENTRENAR_TARGET_UNA_CLASE"
    if leakage_present:
        estado = "NO_ENTRENAR_FUGA_EN_FEATURES"

    rows.append({"validacion": "estado_dataset", "valor": estado})
    return pd.DataFrame(rows)


# ============================================================
# PARTICIÓN TEMPORAL
# ============================================================


def make_temporal_split(
    df: pd.DataFrame,
    date_col: str = "fecha_corte",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    if date_col not in df.columns:
        raise ValueError(f"No existe columna temporal {date_col}.")

    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    periods = sorted(data[date_col].dt.to_period("M").unique())

    if len(periods) < 6:
        raise ValueError(
            "Se requieren al menos 6 periodos mensuales para validación temporal robusta."
        )

    n = len(periods)
    train_end_idx = max(1, int(np.floor(n * train_ratio)))
    valid_end_idx = max(
        train_end_idx + 1, int(np.floor(n * (train_ratio + valid_ratio)))
    )
    valid_end_idx = min(valid_end_idx, n - 1)

    train_periods = periods[:train_end_idx]
    valid_periods = periods[train_end_idx:valid_end_idx]
    test_periods = periods[valid_end_idx:]

    train = data[data[date_col].dt.to_period("M").isin(train_periods)].copy()
    valid = data[data[date_col].dt.to_period("M").isin(valid_periods)].copy()
    test = data[data[date_col].dt.to_period("M").isin(test_periods)].copy()

    # Si validación o test quedan sin positivos, usar últimos periodos con eventos.
    # Esto evita métricas inválidas en muestras extremadamente desbalanceadas.
    if valid[TARGET].sum() == 0 or test[TARGET].sum() == 0:
        # Fallback determinista: 70/15/15 por filas ordenadas.
        n_rows = len(data)
        i1 = int(n_rows * train_ratio)
        i2 = int(n_rows * (train_ratio + valid_ratio))
        train = data.iloc[:i1].copy()
        valid = data.iloc[i1:i2].copy()
        test = data.iloc[i2:].copy()

    info = {
        "train_periodos": f"{train[date_col].min()} -> {train[date_col].max()}",
        "valid_periodos": f"{valid[date_col].min()} -> {valid[date_col].max()}",
        "test_periodos": f"{test[date_col].min()} -> {test[date_col].max()}",
        "train_filas": str(len(train)),
        "valid_filas": str(len(valid)),
        "test_filas": str(len(test)),
        "train_eventos": str(int(train[TARGET].sum())),
        "valid_eventos": str(int(valid[TARGET].sum())),
        "test_eventos": str(int(test[TARGET].sum())),
    }

    return train, valid, test, info


# ============================================================
# MÉTRICAS ROBUSTAS
# ============================================================


def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return (
        float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else np.nan
    )


def safe_prauc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return (
        float(average_precision_score(y_true, y_prob))
        if len(np.unique(y_true)) == 2
        else np.nan
    )


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_true[mask])
        bin_conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def global_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, dataset_name: str
) -> pd.DataFrame:
    y_prob_clip = np.clip(y_prob, 1e-8, 1 - 1e-8)
    roc = safe_auc(y_true, y_prob)
    pr = safe_prauc(y_true, y_prob)
    rows = [
        {"dataset": dataset_name, "metrica": "n", "valor": len(y_true)},
        {"dataset": dataset_name, "metrica": "eventos", "valor": int(np.sum(y_true))},
        {
            "dataset": dataset_name,
            "metrica": "tasa_evento",
            "valor": float(np.mean(y_true)),
        },
        {"dataset": dataset_name, "metrica": "roc_auc", "valor": roc},
        {"dataset": dataset_name, "metrica": "pr_auc", "valor": pr},
        {
            "dataset": dataset_name,
            "metrica": "gini",
            "valor": float(2 * roc - 1) if not np.isnan(roc) else np.nan,
        },
        {
            "dataset": dataset_name,
            "metrica": "ks",
            "valor": ks_statistic(y_true, y_prob),
        },
        {
            "dataset": dataset_name,
            "metrica": "brier_score",
            "valor": float(brier_score_loss(y_true, y_prob_clip)),
        },
        {
            "dataset": dataset_name,
            "metrica": "log_loss",
            "valor": float(log_loss(y_true, y_prob_clip, labels=[0, 1])),
        },
        {
            "dataset": dataset_name,
            "metrica": "ece_10_bins",
            "valor": expected_calibration_error(y_true, y_prob_clip, 10),
        },
    ]
    return pd.DataFrame(rows)


def threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Iterable[float],
    dataset_name: str,
) -> pd.DataFrame:
    rows = []
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "dataset": dataset_name,
                "threshold": float(thr),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "f1": f1_score(y_true, pred, zero_division=0),
                "f2": fbeta_score(y_true, pred, beta=2, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, pred),
                "tasa_alerta": float(np.mean(pred)),
                "alertas": int(np.sum(pred)),
            }
        )
    return pd.DataFrame(rows)


def topk_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, dataset_name: str
) -> pd.DataFrame:
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    total_events = np.sum(y_true)
    base_rate = np.mean(y_true)
    rows = []

    for k in TOP_K_VALUES:
        if k > len(y_true):
            continue
        top = y_sorted[:k]
        precision = np.mean(top) if k > 0 else np.nan
        recall = np.sum(top) / total_events if total_events > 0 else np.nan
        lift = precision / base_rate if base_rate > 0 else np.nan
        rows.append(
            {
                "dataset": dataset_name,
                "tipo": "top_k",
                "k_o_pct": k,
                "filas_seleccionadas": k,
                "eventos_capturados": int(np.sum(top)),
                "precision": float(precision),
                "recall": float(recall),
                "lift": float(lift),
            }
        )

    for pct in TOP_PCT_VALUES:
        k = max(1, int(len(y_true) * pct))
        top = y_sorted[:k]
        precision = np.mean(top)
        recall = np.sum(top) / total_events if total_events > 0 else np.nan
        lift = precision / base_rate if base_rate > 0 else np.nan
        rows.append(
            {
                "dataset": dataset_name,
                "tipo": "top_pct",
                "k_o_pct": pct,
                "filas_seleccionadas": k,
                "eventos_capturados": int(np.sum(top)),
                "precision": float(precision),
                "recall": float(recall),
                "lift": float(lift),
            }
        )

    return pd.DataFrame(rows)


def calibration_table(
    y_true: np.ndarray, y_prob: np.ndarray, dataset_name: str, n_bins: int = 10
) -> pd.DataFrame:
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    # qcut puede fallar si hay muchos empates; duplicates='drop' lo hace robusto.
    df["decil"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = (
        df.groupby("decil", observed=False)
        .agg(
            n=("y", "size"),
            prob_min=("p", "min"),
            prob_max=("p", "max"),
            prob_prom=("p", "mean"),
            eventos=("y", "sum"),
            tasa_real=("y", "mean"),
        )
        .reset_index()
    )
    out["dataset"] = dataset_name
    return out


def find_best_threshold_by_f2(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # Evaluar muchos umbrales entre percentiles de probabilidad para evitar saltos.
    qs = np.linspace(0.50, 0.995, 150)
    thresholds = sorted(set(np.quantile(y_prob, qs).tolist() + THRESHOLDS_TO_EVALUATE))
    best_thr = 0.10
    best_f2 = -1
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = thr
    return float(best_thr)


# ============================================================
# CALIBRACIÓN
# ============================================================


def fit_platt_calibrator(
    y_valid: np.ndarray, p_valid: np.ndarray
) -> LogisticRegression:
    p = np.clip(p_valid, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    cal = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=RANDOM_SEED)
    cal.fit(logit, y_valid)
    return cal


def apply_platt(calibrator: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


# ============================================================
# ENTRENAMIENTO
# ============================================================


def train_catboost_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: List[str],
    cat_features: List[str],
) -> CatBoostClassifier:
    X_train = clean_features_for_catboost(train_df, features, cat_features)
    y_train = train_df[TARGET].astype(int)
    X_valid = clean_features_for_catboost(valid_df, features, cat_features)
    y_valid = valid_df[TARGET].astype(int)

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        custom_metric=["AUC", "PRAUC", "F1", "Precision", "Recall", "Logloss"],
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=8,
        random_strength=1.2,
        bootstrap_type="Bayesian",
        bagging_temperature=0.7,
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        od_type="Iter",
        od_wait=200,
        verbose=200,
        allow_writing_files=False,
    )

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    return model


def predict_proba_model(
    model: CatBoostClassifier,
    df: pd.DataFrame,
    features: List[str],
    cat_features: List[str],
) -> np.ndarray:
    X = clean_features_for_catboost(df, features, cat_features)
    pool = Pool(X, cat_features=cat_features)
    return model.predict_proba(pool)[:, 1]


# ============================================================
# GRÁFICAS
# ============================================================


def save_plots(
    y_test: np.ndarray, p_test: np.ndarray, feature_importance: pd.DataFrame
) -> None:
    if plt is None:
        print("matplotlib no está disponible; se omiten gráficas.")
        return

    # ROC
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, p_test)
        plt.figure(figsize=(7, 5))
        plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc_score(y_test, p_test):.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR / Recall")
        plt.title("Curva ROC - Test")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "roc_curve_test.png", dpi=160)
        plt.close()

    # PR curve manual
    try:
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(y_test, p_test)
        plt.figure(figsize=(7, 5))
        plt.plot(
            recall,
            precision,
            label=f"PR-AUC={average_precision_score(y_test, p_test):.4f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Curva Precision-Recall - Test")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pr_curve_test.png", dpi=160)
        plt.close()
    except Exception:
        pass

    # Calibración
    try:
        prob_true, prob_pred = calibration_curve(
            y_test, p_test, n_bins=10, strategy="quantile"
        )
        plt.figure(figsize=(7, 5))
        plt.plot(prob_pred, prob_true, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("Probabilidad promedio predicha")
        plt.ylabel("Tasa real")
        plt.title("Calibración - Test")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "calibration_curve_test.png", dpi=160)
        plt.close()
    except Exception:
        pass

    # Importancia top 25
    try:
        top = feature_importance.head(25).iloc[::-1]
        plt.figure(figsize=(9, 7))
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Importancia")
        plt.title("Top 25 variables - CatBoost")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance_top25.png", dpi=160)
        plt.close()
    except Exception:
        pass


# ============================================================
# SCORING ACTUAL OPCIONAL / SALIDAS DE PRODUCCIÓN
# ============================================================


def normalizar_estado_operativo(serie: pd.Series) -> pd.Series:
    """Normaliza estados para separar flujos preventivo y recuperación."""
    return (
        serie.astype("string")
        .fillna("SIN_ESTADO")
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def asignar_segmento_operativo(estado: pd.Series) -> pd.Series:
    estado_norm = normalizar_estado_operativo(estado)
    return np.select(
        [estado_norm.isin(PREVENTIVE_STATES), estado_norm.isin(RECOVERY_STATES)],
        ["PREVENTIVO", "RECUPERACION"],
        default="OTRO",
    )


def clasificar_nivel_final(row: pd.Series, threshold_f2: float) -> str:
    """
    Nivel final combinando modelo + reglas actuales.
    La probabilidad predice riesgo 90d; las reglas actuales identifican urgencia operativa.
    """
    nivel_reglas = str(row.get("nivel_riesgo_reglas", "")).upper().strip()
    prob = float(row.get("probabilidad_riesgo_90d", 0) or 0)
    ranking_segmento = row.get("ranking_segmento", np.nan)
    segmento = str(row.get("segmento_operativo", ""))

    try:
        ranking_segmento = int(ranking_segmento)
    except Exception:
        ranking_segmento = 999999999

    if segmento == "PREVENTIVO":
        if (
            ranking_segmento <= TOP_K_PREVENTIVO
            or prob >= threshold_f2
            or nivel_reglas == "CRÍTICO"
        ):
            return "Crítico"
        if (
            ranking_segmento <= TOP_K_PREVENTIVO * 2
            or prob >= threshold_f2 / 2
            or nivel_reglas == "ALTO"
        ):
            return "Alto"
        if prob >= 0.02 or nivel_reglas == "MEDIO":
            return "Medio"
        return "Bajo"

    if segmento == "RECUPERACION":
        if ranking_segmento <= TOP_K_RECUPERACION or nivel_reglas == "CRÍTICO":
            return "Recuperación prioritaria"
        return "Recuperación normal"

    if prob >= threshold_f2:
        return "Revisar"
    return "Bajo"


def asignar_accion_final(row: pd.Series) -> str:
    segmento = str(row.get("segmento_operativo", ""))
    nivel = str(row.get("nivel_alerta_final", ""))

    if segmento == "PREVENTIVO":
        if nivel == "Crítico":
            return "Gestión preventiva inmediata: validar saldo, contactar CB y evaluar bloqueo preventivo."
        if nivel == "Alto":
            return "Contacto el mismo día y seguimiento de compensación."
        if nivel == "Medio":
            return "Monitoreo preventivo y recordatorio de compensación."
        return "Seguimiento normal."

    if segmento == "RECUPERACION":
        if nivel == "Recuperación prioritaria":
            return "Enviar a recuperación/cartera prioritaria; no mezclar con alerta preventiva."
        return "Seguimiento por recuperación/cartera."

    return "Revisar estado operativo del CB."


def cargar_top500_preventivo_anterior() -> set:
    """Carga el último snapshot previo para detectar nuevos CB que entran al Top preventivo."""
    archivos = sorted(SCORING_SNAPSHOT_DIR.glob("scoring_actual_cb_*.parquet"))
    if not archivos:
        return set()

    ultimo = archivos[-1]
    try:
        prev = pd.read_parquet(ultimo)
        if (
            "segmento_operativo" not in prev.columns
            or "ranking_segmento" not in prev.columns
        ):
            return set()
        top_prev = prev[
            prev["segmento_operativo"].eq("PREVENTIVO")
            & pd.to_numeric(prev["ranking_segmento"], errors="coerce").le(
                TOP_K_PREVENTIVO
            )
        ]
        return set(top_prev["codigo_punto"].dropna().astype(int).tolist())
    except Exception:
        return set()


def guardar_snapshot_scoring(scoring: pd.DataFrame) -> None:
    fecha = pd.Timestamp.today().strftime("%Y%m%d_%H%M%S")
    ruta = SCORING_SNAPSHOT_DIR / f"scoring_actual_cb_{fecha}.parquet"
    safe_to_parquet(scoring, ruta)


def generar_resumen_operativo(scoring: pd.DataFrame) -> pd.DataFrame:
    resumen = (
        scoring.groupby(["segmento_operativo", "nivel_alerta_final"], dropna=False)
        .agg(
            puntos=("codigo_punto", "nunique"),
            saldo_expuesto=("saldo_expuesto", "sum"),
            perdida_esperada_modelo=("perdida_esperada_modelo", "sum"),
            probabilidad_promedio=("probabilidad_riesgo_90d", "mean"),
        )
        .reset_index()
    )
    return resumen.sort_values(["segmento_operativo", "nivel_alerta_final"])


def generar_control_calidad_scoring(scoring: pd.DataFrame) -> pd.DataFrame:
    filas = []
    filas.append({"control": "filas_scoring", "valor": len(scoring)})
    filas.append(
        {
            "control": "cb_unicos",
            "valor": (
                scoring["codigo_punto"].nunique()
                if "codigo_punto" in scoring.columns
                else np.nan
            ),
        }
    )
    filas.append(
        {
            "control": "probabilidad_nula",
            "valor": int(scoring["probabilidad_riesgo_90d"].isna().sum()),
        }
    )
    if "dias_sin_transacciones" in scoring.columns:
        filas.append(
            {
                "control": "dias_sin_transacciones_nulos",
                "valor": int(scoring["dias_sin_transacciones"].isna().sum()),
            }
        )
    if "consumo_cupo_punto" in scoring.columns:
        filas.append(
            {
                "control": "consumo_cupo_punto_negativos",
                "valor": int(
                    (
                        pd.to_numeric(scoring["consumo_cupo_punto"], errors="coerce")
                        < 0
                    ).sum()
                ),
            }
        )
    if "consumo_cupo_visual" in scoring.columns:
        filas.append(
            {
                "control": "consumo_cupo_visual_negativos",
                "valor": int(
                    (
                        pd.to_numeric(scoring["consumo_cupo_visual"], errors="coerce")
                        < 0
                    ).sum()
                ),
            }
        )
    filas.append(
        {
            "control": "saldo_expuesto_total",
            "valor": float(
                scoring.get("saldo_expuesto", pd.Series([0])).fillna(0).sum()
            ),
        }
    )

    if "segmento_operativo" in scoring.columns:
        for seg, n in scoring["segmento_operativo"].value_counts(dropna=False).items():
            filas.append({"control": f"segmento_{seg}", "valor": int(n)})

    if "flag_nuevo_top500_preventivo" in scoring.columns:
        filas.append(
            {
                "control": "nuevos_top500_preventivo",
                "valor": int(scoring["flag_nuevo_top500_preventivo"].sum()),
            }
        )

    return pd.DataFrame(filas)


def build_current_scoring(
    model: CatBoostClassifier,
    calibrator: LogisticRegression,
    features: List[str],
    cat_features: List[str],
    threshold_f2: float,
) -> None:
    """
    Genera scoring actual y salidas limpias para producción.

    Mantiene las salidas originales:
    - OUTPUT_DIR/scoring_actual_cb.parquet
    - OUTPUT_DIR/scoring_actual_cb_top5000.csv

    Agrega salidas separadas para tablero/operación:
    - salidas_produccion/scoring_actual_cb_limpio.parquet
    - salidas_produccion/scoring_preventivo_activos.parquet
    - salidas_produccion/scoring_recuperacion_cerrados.parquet
    - salidas_produccion/scoring_top500_preventivo.csv
    - salidas_produccion/scoring_top1000_recuperacion.csv
    - salidas_produccion/resumen_operativo_dashboard.csv
    - salidas_produccion/control_calidad_scoring.csv
    """
    if not PANEL_HISTORICO_PATH.exists():
        print("No se encontró panel histórico para scoring actual.")
        return

    panel = pd.read_parquet(PANEL_HISTORICO_PATH)
    if "fecha_corte" not in panel.columns or "codigo_punto" not in panel.columns:
        print("Panel histórico no tiene columnas requeridas para scoring actual.")
        return

    panel["fecha_corte"] = pd.to_datetime(panel["fecha_corte"], errors="coerce")
    latest = (
        panel.sort_values(["codigo_punto", "fecha_corte"])
        .groupby("codigo_punto", as_index=False)
        .tail(1)
        .copy()
    )

    # Asegurar columnas faltantes del modelo.
    for col in features:
        if col not in latest.columns:
            latest[col] = np.nan

    p_raw = predict_proba_model(model, latest, features, cat_features)
    p_cal = apply_platt(calibrator, p_raw)
    latest["probabilidad_riesgo_90d_raw"] = p_raw
    latest["probabilidad_riesgo_90d"] = p_cal

    # Merge con alertas actuales si existe.
    if ALERTAS_ACTUALES_PATH.exists():
        alertas = pd.read_parquet(ALERTAS_ACTUALES_PATH)
        keep_alertas = [
            "codigo_punto",
            "estado_actual",
            "nombre_punto",
            "departamento",
            "municipio",
            "red_cb",
            "saldo_expuesto",
            "dias_sin_compensar",
            "dias_sin_transacciones",
            "consumo_cupo_punto",
            "score_riesgo_reglas",
            "nivel_riesgo_reglas",
            "motivo_alerta",
            "accion_recomendada",
        ]
        keep_alertas = [c for c in keep_alertas if c in alertas.columns]
        latest = latest.merge(
            alertas[keep_alertas],
            on="codigo_punto",
            how="left",
            suffixes=("", "_alerta"),
        )

        # Si el panel histórico ya tenía columnas con el mismo nombre, pandas crea
        # columnas *_alerta. Para producción priorizamos el dato actual de alertas,
        # especialmente dias_sin_transacciones, saldo, consumo y estado operativo.
        for col in [c for c in keep_alertas if c != "codigo_punto"]:
            col_alerta = f"{col}_alerta"
            if col_alerta in latest.columns:
                if col in latest.columns:
                    latest[col] = latest[col_alerta].combine_first(latest[col])
                else:
                    latest[col] = latest[col_alerta]
                latest = latest.drop(columns=[col_alerta])

    # Asegurar columnas operativas mínimas y corregir nulos por cruces/sufijos.
    for col in [
        "saldo_expuesto",
        "dias_sin_compensar",
        "dias_sin_transacciones",
        "consumo_cupo_punto",
    ]:
        if col not in latest.columns:
            latest[col] = np.nan
        latest[col] = pd.to_numeric(latest[col], errors="coerce")

    # Campo para tablero: conserva consumo_cupo_punto original para auditoría,
    # pero evita mostrar porcentajes negativos cuando el saldo viene a favor del punto.
    latest["consumo_cupo_visual"] = latest["consumo_cupo_punto"].clip(lower=0)

    if "estado_actual" not in latest.columns:
        latest["estado_actual"] = "SIN_ESTADO"

    latest["fecha_scoring"] = pd.Timestamp.today().normalize()
    latest["estado_actual_norm"] = normalizar_estado_operativo(latest["estado_actual"])
    latest["segmento_operativo"] = asignar_segmento_operativo(latest["estado_actual"])

    latest["perdida_esperada_modelo"] = latest["probabilidad_riesgo_90d"] * latest[
        "saldo_expuesto"
    ].fillna(0)

    latest["nivel_modelo"] = pd.cut(
        latest["probabilidad_riesgo_90d"],
        bins=[-0.001, threshold_f2 / 2, threshold_f2, min(1, threshold_f2 * 1.5), 1.0],
        labels=["Bajo", "Medio", "Alto", "Crítico"],
        duplicates="drop",
    ).astype(str)

    # Ranking global y por segmento. Prioriza impacto financiero y luego probabilidad.
    latest = latest.sort_values(
        ["perdida_esperada_modelo", "probabilidad_riesgo_90d"], ascending=[False, False]
    ).reset_index(drop=True)
    latest["ranking_global"] = np.arange(1, len(latest) + 1)
    latest["ranking_segmento"] = (
        latest.groupby("segmento_operativo")["perdida_esperada_modelo"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    latest["flag_top500_preventivo"] = (
        latest["segmento_operativo"].eq("PREVENTIVO")
        & latest["ranking_segmento"].le(TOP_K_PREVENTIVO)
    ).astype(int)

    top500_anterior = cargar_top500_preventivo_anterior()
    # En la primera ejecución no existe snapshot anterior; en ese caso NO marcamos
    # 500 puntos como nuevos. Desde la segunda corrida sí se compara contra el Top anterior.
    if len(top500_anterior) == 0:
        latest["flag_nuevo_top500_preventivo"] = 0
    else:
        latest["flag_nuevo_top500_preventivo"] = (
            latest["flag_top500_preventivo"].eq(1)
            & ~latest["codigo_punto"].astype(int).isin(top500_anterior)
        ).astype(int)

    latest["nivel_alerta_final"] = latest.apply(
        lambda row: clasificar_nivel_final(row, threshold_f2), axis=1
    )
    latest["accion_final_produccion"] = latest.apply(asignar_accion_final, axis=1)

    output_cols = [
        "fecha_scoring",
        "codigo_punto",
        "fecha_corte",
        "nombre_punto",
        "estado_actual",
        "estado_actual_norm",
        "segmento_operativo",
        "departamento",
        "municipio",
        "red_cb",
        "probabilidad_riesgo_90d",
        "probabilidad_riesgo_90d_raw",
        "nivel_modelo",
        "nivel_riesgo_reglas",
        "nivel_alerta_final",
        "ranking_global",
        "ranking_segmento",
        "flag_top500_preventivo",
        "flag_nuevo_top500_preventivo",
        "perdida_esperada_modelo",
        "saldo_expuesto",
        "dias_sin_compensar",
        "dias_sin_transacciones",
        "consumo_cupo_punto",
        "consumo_cupo_visual",
        "score_riesgo_reglas",
        "motivo_alerta",
        "accion_recomendada",
        "accion_final_produccion",
    ]
    output_cols = [c for c in output_cols if c in latest.columns]

    scoring_limpio = latest[output_cols].copy()

    # Salidas originales para no romper tu proceso actual.
    safe_to_parquet(scoring_limpio, OUTPUT_DIR / "scoring_actual_cb.parquet")
    scoring_limpio.head(5000).to_csv(
        OUTPUT_DIR / "scoring_actual_cb_top5000.csv", index=False, encoding="utf-8-sig"
    )

    # Nuevas salidas limpias separadas por flujo.
    preventivo = scoring_limpio[
        scoring_limpio["segmento_operativo"].eq("PREVENTIVO")
    ].copy()
    recuperacion = scoring_limpio[
        scoring_limpio["segmento_operativo"].eq("RECUPERACION")
    ].copy()
    otros = scoring_limpio[scoring_limpio["segmento_operativo"].eq("OTRO")].copy()

    safe_to_parquet(
        scoring_limpio, PRODUCTION_OUTPUT_DIR / "scoring_actual_cb_limpio.parquet"
    )
    safe_to_parquet(
        preventivo, PRODUCTION_OUTPUT_DIR / "scoring_preventivo_activos.parquet"
    )
    safe_to_parquet(
        recuperacion, PRODUCTION_OUTPUT_DIR / "scoring_recuperacion_cerrados.parquet"
    )
    safe_to_parquet(otros, PRODUCTION_OUTPUT_DIR / "scoring_otros_estados.parquet")

    preventivo.head(TOP_K_PREVENTIVO).to_csv(
        PRODUCTION_OUTPUT_DIR / f"scoring_top{TOP_K_PREVENTIVO}_preventivo.csv",
        index=False,
        encoding="utf-8-sig",
    )
    recuperacion.head(TOP_K_RECUPERACION).to_csv(
        PRODUCTION_OUTPUT_DIR / f"scoring_top{TOP_K_RECUPERACION}_recuperacion.csv",
        index=False,
        encoding="utf-8-sig",
    )

    resumen = generar_resumen_operativo(scoring_limpio)
    resumen.to_csv(
        PRODUCTION_OUTPUT_DIR / "resumen_operativo_dashboard.csv",
        index=False,
        encoding="utf-8-sig",
    )

    control = generar_control_calidad_scoring(scoring_limpio)
    control.to_csv(
        PRODUCTION_OUTPUT_DIR / "control_calidad_scoring.csv",
        index=False,
        encoding="utf-8-sig",
    )

    guardar_snapshot_scoring(scoring_limpio)

    print("\nScoring producción generado:")
    print(f"- {PRODUCTION_OUTPUT_DIR / 'scoring_actual_cb_limpio.parquet'}")
    print(f"- {PRODUCTION_OUTPUT_DIR / 'scoring_preventivo_activos.parquet'}")
    print(f"- {PRODUCTION_OUTPUT_DIR / 'scoring_recuperacion_cerrados.parquet'}")
    print(f"- {PRODUCTION_OUTPUT_DIR / 'resumen_operativo_dashboard.csv'}")


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print("====================================================")
    print("ENTRENAMIENTO CATBOOST RIESGO CB V1 - FLUJOS PRODUCCION AJUSTADOS")
    print("====================================================")

    df = read_dataset()
    features = read_feature_list(df)
    cat_features = infer_categorical_features(df, features)

    validation = validate_dataset(df, features)
    validation.to_csv(
        OUTPUT_DIR / "00_validacion_dataset.csv", index=False, encoding="utf-8-sig"
    )
    print(validation.to_string(index=False))

    estado = validation.loc[
        validation["validacion"].eq("estado_dataset"), "valor"
    ].iloc[0]
    if estado != "OK_PARA_ENTRENAR":
        raise ValueError(
            f"El dataset no pasó validación: {estado}. Revisa 00_validacion_dataset.csv"
        )

    # Guardar variables efectivas usadas.
    pd.DataFrame(
        {
            "variable": features,
            "tipo": [
                "categorica" if c in cat_features else "numerica" for c in features
            ],
        }
    ).to_csv(
        OUTPUT_DIR / "01_variables_usadas_modelo.csv", index=False, encoding="utf-8-sig"
    )

    train_df, valid_df, test_df, split_info = make_temporal_split(df)
    pd.DataFrame([split_info]).to_csv(
        OUTPUT_DIR / "02_split_temporal.csv", index=False, encoding="utf-8-sig"
    )
    print("\nSplit temporal:")
    print(json.dumps(split_info, indent=2, ensure_ascii=False))

    print("\nEntrenando CatBoost...")
    model = train_catboost_model(train_df, valid_df, features, cat_features)

    # Predicciones raw
    p_train_raw = predict_proba_model(model, train_df, features, cat_features)
    p_valid_raw = predict_proba_model(model, valid_df, features, cat_features)
    p_test_raw = predict_proba_model(model, test_df, features, cat_features)

    # Calibración con validación
    calibrator = fit_platt_calibrator(valid_df[TARGET].values, p_valid_raw)
    p_train = apply_platt(calibrator, p_train_raw)
    p_valid = apply_platt(calibrator, p_valid_raw)
    p_test = apply_platt(calibrator, p_test_raw)

    # Mejor umbral por F2 en validación calibrada
    best_thr_f2 = find_best_threshold_by_f2(valid_df[TARGET].values, p_valid)
    thresholds = sorted(set(THRESHOLDS_TO_EVALUATE + [best_thr_f2]))

    # Métricas globales
    metrics_global = pd.concat(
        [
            global_metrics(train_df[TARGET].values, p_train, "train_calibrado"),
            global_metrics(valid_df[TARGET].values, p_valid, "valid_calibrado"),
            global_metrics(test_df[TARGET].values, p_test, "test_calibrado"),
            global_metrics(test_df[TARGET].values, p_test_raw, "test_raw_sin_calibrar"),
        ],
        ignore_index=True,
    )
    metrics_global.to_csv(
        OUTPUT_DIR / "03_metricas_globales.csv", index=False, encoding="utf-8-sig"
    )

    # Métricas por threshold
    thr_metrics = pd.concat(
        [
            threshold_metrics(
                valid_df[TARGET].values, p_valid, thresholds, "valid_calibrado"
            ),
            threshold_metrics(
                test_df[TARGET].values, p_test, thresholds, "test_calibrado"
            ),
        ],
        ignore_index=True,
    )
    thr_metrics["umbral_seleccionado_f2_valid"] = best_thr_f2
    thr_metrics.to_csv(
        OUTPUT_DIR / "04_metricas_por_umbral.csv", index=False, encoding="utf-8-sig"
    )

    # Top-K
    topk = pd.concat(
        [
            topk_metrics(valid_df[TARGET].values, p_valid, "valid_calibrado"),
            topk_metrics(test_df[TARGET].values, p_test, "test_calibrado"),
        ],
        ignore_index=True,
    )
    topk.to_csv(OUTPUT_DIR / "05_metricas_topk.csv", index=False, encoding="utf-8-sig")

    # Calibración por deciles
    calib = pd.concat(
        [
            calibration_table(valid_df[TARGET].values, p_valid, "valid_calibrado"),
            calibration_table(test_df[TARGET].values, p_test, "test_calibrado"),
        ],
        ignore_index=True,
    )
    calib.to_csv(
        OUTPUT_DIR / "06_calibracion_deciles.csv", index=False, encoding="utf-8-sig"
    )

    # Predicciones test con trazabilidad
    pred_test = test_df[[c for c in ID_COLS if c in test_df.columns] + [TARGET]].copy()
    pred_test["probabilidad_riesgo_90d_raw"] = p_test_raw
    pred_test["probabilidad_riesgo_90d"] = p_test
    pred_test["pred_umbral_f2"] = (p_test >= best_thr_f2).astype(int)
    pred_test.to_csv(
        OUTPUT_DIR / "07_predicciones_test.csv", index=False, encoding="utf-8-sig"
    )

    # Importancia de variables
    train_pool = Pool(
        clean_features_for_catboost(train_df, features, cat_features),
        train_df[TARGET].astype(int),
        cat_features=cat_features,
    )
    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.get_feature_importance(
                train_pool, type="FeatureImportance"
            ),
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(
        OUTPUT_DIR / "08_importancia_variables.csv", index=False, encoding="utf-8-sig"
    )

    # Guardar modelo y calibrador
    model.save_model(str(OUTPUT_DIR / "catboost_riesgo_cb_v1.cbm"))
    joblib.dump(calibrator, OUTPUT_DIR / "calibrador_platt_v1.joblib")
    joblib.dump(
        {
            "features": features,
            "cat_features": cat_features,
            "target": TARGET,
            "best_threshold_f2_valid": best_thr_f2,
            "random_seed": RANDOM_SEED,
            "data_path": str(DATASET_PATH),
        },
        OUTPUT_DIR / "metadata_modelo_v1.joblib",
    )

    # Resumen ejecutivo
    test_metrics = (
        metrics_global[metrics_global["dataset"].eq("test_calibrado")]
        .set_index("metrica")["valor"]
        .to_dict()
    )
    top500 = topk[
        (topk["dataset"].eq("test_calibrado"))
        & (topk["tipo"].eq("top_k"))
        & (topk["k_o_pct"].eq(500))
    ]
    precision_top500 = (
        float(top500["precision"].iloc[0]) if not top500.empty else np.nan
    )
    recall_top500 = float(top500["recall"].iloc[0]) if not top500.empty else np.nan

    resumen = pd.DataFrame(
        [
            {"indicador": "roc_auc_test", "valor": test_metrics.get("roc_auc", np.nan)},
            {"indicador": "pr_auc_test", "valor": test_metrics.get("pr_auc", np.nan)},
            {"indicador": "ks_test", "valor": test_metrics.get("ks", np.nan)},
            {
                "indicador": "brier_score_test",
                "valor": test_metrics.get("brier_score", np.nan),
            },
            {"indicador": "precision_top500_test", "valor": precision_top500},
            {"indicador": "recall_top500_test", "valor": recall_top500},
            {"indicador": "umbral_f2_valid", "valor": best_thr_f2},
            {"indicador": "variables_usadas", "valor": len(features)},
            {"indicador": "variables_categoricas", "valor": len(cat_features)},
        ]
    )
    resumen.to_csv(
        OUTPUT_DIR / "09_resumen_ejecutivo_metricas.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Gráficas
    save_plots(test_df[TARGET].values, p_test, importance)

    # Scoring actual opcional
    build_current_scoring(model, calibrator, features, cat_features, best_thr_f2)

    print("\n====================================================")
    print("ENTRENAMIENTO FINALIZADO")
    print(f"Artefactos guardados en: {OUTPUT_DIR}")
    print("Archivos clave:")
    print("- 03_metricas_globales.csv")
    print("- 04_metricas_por_umbral.csv")
    print("- 05_metricas_topk.csv")
    print("- 06_calibracion_deciles.csv")
    print("- 08_importancia_variables.csv")
    print("- 09_resumen_ejecutivo_metricas.csv")
    print("- catboost_riesgo_cb_v1.cbm")
    print("====================================================")


if __name__ == "__main__":
    main()
