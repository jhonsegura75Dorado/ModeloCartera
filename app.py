# -*- coding: utf-8 -*-
"""
Tablero ejecutivo y operativo - Riesgo CB / No compensación
==========================================================

Objetivo
--------
Leer las salidas productivas del modelo CatBoost y entregar una vista clara para:
1. Identificar CB preventivos con mayor probabilidad de pérdida/no compensación.
2. Priorizar por pérdida esperada, saldo expuesto, días sin compensar e inactividad.
3. Separar prevención de recuperación.
4. Descargar bases de gestión para enviar a gestores.

Archivos esperados
------------------
_modelo_riesgo_cb/salidas_produccion/scoring_preventivo_activos.parquet
_modelo_riesgo_cb/salidas_produccion/scoring_recuperacion_cerrados.parquet
_modelo_riesgo_cb/salidas_produccion/resumen_operativo_dashboard.csv
_modelo_riesgo_cb/salidas_produccion/control_calidad_scoring.csv

Ejecución
---------
streamlit run app_streamlit_riesgo_cb_decisiones.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Riesgo CB | Decisión Preventiva",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR_DEFAULT = Path("_modelo_riesgo_cb") / "salidas_produccion"

PASTEL = {
    "bg": "#F7F8FC",
    "card": "#FFFFFF",
    "blue": "#DDEBFF",
    "green": "#DFF7E8",
    "yellow": "#FFF4CF",
    "orange": "#FFE4D6",
    "red": "#FFE1E1",
    "purple": "#EFE4FF",
    "gray": "#EEF1F6",
    "text": "#243042",
    "muted": "#667085",
    "border": "#E6E8EF",
}

CRITICAL_LEVELS = ["Crítico", "Alto", "Recuperación prioritaria"]

NUMERIC_COLUMNS_TO_FORMAT = [
    "probabilidad_riesgo_90d",
    "probabilidad_riesgo_90d_raw",
    "perdida_esperada_modelo",
    "saldo_expuesto",
    "dias_sin_compensar",
    "dias_sin_transacciones",
    "consumo_cupo_punto",
    "consumo_cupo_visual",
    "score_riesgo_reglas",
]

DISPLAY_COLUMNS_PREVENTIVO = [
    "ranking_segmento",
    "codigo_punto",
    "nombre_punto",
    "estado_actual",
    "departamento",
    "municipio",
    "red_cb",
    "nivel_alerta_final",
    "probabilidad_riesgo_90d",
    "saldo_expuesto",
    "perdida_esperada_modelo",
    "dias_sin_compensar",
    "dias_sin_transacciones",
    "consumo_cupo_visual",
    "nivel_riesgo_reglas",
    "flag_nuevo_top500_preventivo",
    "motivo_alerta",
    "accion_final_produccion",
]

DISPLAY_COLUMNS_RECUPERACION = [
    "ranking_segmento",
    "codigo_punto",
    "nombre_punto",
    "estado_actual",
    "departamento",
    "municipio",
    "red_cb",
    "nivel_alerta_final",
    "probabilidad_riesgo_90d",
    "saldo_expuesto",
    "perdida_esperada_modelo",
    "dias_sin_compensar",
    "dias_sin_transacciones",
    "consumo_cupo_visual",
    "motivo_alerta",
    "accion_final_produccion",
]


# ============================================================
# ESTILOS
# ============================================================

st.markdown(
    f"""
    <style>
        .stApp {{
            background: {PASTEL['bg']};
            color: {PASTEL['text']};
        }}
        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid {PASTEL['border']};
        }}
        .main-title {{
            font-size: 2.15rem;
            font-weight: 800;
            color: {PASTEL['text']};
            margin-bottom: 0.2rem;
        }}
        .subtitle {{
            color: {PASTEL['muted']};
            font-size: 1rem;
            margin-bottom: 1.1rem;
        }}
        .card {{
            background: {PASTEL['card']};
            border: 1px solid {PASTEL['border']};
            border-radius: 18px;
            padding: 18px 18px;
            box-shadow: 0 6px 18px rgba(36, 48, 66, 0.06);
            min-height: 118px;
        }}
        .kpi-label {{
            color: {PASTEL['muted']};
            font-size: 0.86rem;
            font-weight: 650;
            margin-bottom: 0.35rem;
        }}
        .kpi-value {{
            color: {PASTEL['text']};
            font-size: 1.85rem;
            font-weight: 850;
            line-height: 1.05;
        }}
        .kpi-help {{
            color: {PASTEL['muted']};
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }}
        .tag {{
            display: inline-block;
            padding: 6px 11px;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.78rem;
            margin: 2px 4px 2px 0;
            border: 1px solid rgba(0,0,0,0.05);
        }}
        .tag-red {{ background: {PASTEL['red']}; color: #9B1C1C; }}
        .tag-orange {{ background: {PASTEL['orange']}; color: #A64000; }}
        .tag-yellow {{ background: {PASTEL['yellow']}; color: #866000; }}
        .tag-green {{ background: {PASTEL['green']}; color: #0F6B3E; }}
        .tag-blue {{ background: {PASTEL['blue']}; color: #2457A6; }}
        .tag-purple {{ background: {PASTEL['purple']}; color: #553C9A; }}
        .decision-box {{
            background: linear-gradient(135deg, #FFFFFF, #F8FBFF);
            border-left: 8px solid #A7C7E7;
            border-radius: 18px;
            padding: 16px 18px;
            border-top: 1px solid {PASTEL['border']};
            border-right: 1px solid {PASTEL['border']};
            border-bottom: 1px solid {PASTEL['border']};
            box-shadow: 0 6px 18px rgba(36, 48, 66, 0.05);
            margin-bottom: 14px;
        }}
        .decision-title {{
            font-weight: 850;
            font-size: 1.05rem;
            color: {PASTEL['text']};
            margin-bottom: 4px;
        }}
        .decision-text {{
            color: {PASTEL['muted']};
            font-size: 0.92rem;
        }}
        .section-title {{
            font-size: 1.35rem;
            font-weight: 850;
            color: {PASTEL['text']};
            margin: 0.4rem 0 0.9rem 0;
        }}
        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid {PASTEL['border']};
            padding: 13px;
            border-radius: 15px;
            box-shadow: 0 6px 18px rgba(36, 48, 66, 0.05);
        }}
        .stDataFrame {{
            border-radius: 16px;
            overflow: hidden;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CARGA Y UTILIDADES
# ============================================================


@st.cache_data(show_spinner=False)
def load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def money_fmt(value: float) -> str:
    if pd.isna(value):
        return "$0"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return (
            f"${value / 1_000_000_000:,.1f} MM".replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
    if abs(value) >= 1_000_000:
        return (
            f"${value / 1_000_000:,.1f} M".replace(",", "X")
            .replace(".", ",")
            .replace("X", ".")
        )
    return f"${value:,.0f}".replace(",", ".")


def pct_fmt(value: float) -> str:
    if pd.isna(value):
        return "0,0%"
    return f"{value * 100:,.1f}%".replace(",", "X").replace(".", ",").replace("X", ".")


def int_fmt(value: float) -> str:
    if pd.isna(value):
        return "0"
    return f"{int(round(float(value))):,}".replace(",", ".")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in NUMERIC_COLUMNS_TO_FORMAT:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ensure_visual_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "consumo_cupo_visual" not in out.columns and "consumo_cupo_punto" in out.columns:
        out["consumo_cupo_visual"] = pd.to_numeric(
            out["consumo_cupo_punto"], errors="coerce"
        ).clip(lower=0)
    if "perdida_esperada_modelo" not in out.columns:
        if {"probabilidad_riesgo_90d", "saldo_expuesto"}.issubset(out.columns):
            out["perdida_esperada_modelo"] = pd.to_numeric(
                out["probabilidad_riesgo_90d"], errors="coerce"
            ).fillna(0) * pd.to_numeric(out["saldo_expuesto"], errors="coerce").fillna(
                0
            )
    if "nivel_alerta_final" not in out.columns:
        out["nivel_alerta_final"] = "Sin nivel"
    if "ranking_segmento" not in out.columns:
        out["ranking_segmento"] = np.arange(1, len(out) + 1)
    return normalize_cols(out)


def filter_df(
    df: pd.DataFrame,
    departamentos: list[str],
    municipios: list[str],
    redes: list[str],
    niveles: list[str],
    prob_min: float,
    saldo_min: float,
    dias_comp_min: int,
    dias_trx_min: int,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    if departamentos and "departamento" in out.columns:
        out = out[out["departamento"].fillna("Sin dato").isin(departamentos)]
    if municipios and "municipio" in out.columns:
        out = out[out["municipio"].fillna("Sin dato").isin(municipios)]
    if redes and "red_cb" in out.columns:
        out = out[out["red_cb"].fillna("Sin dato").isin(redes)]
    if niveles and "nivel_alerta_final" in out.columns:
        out = out[out["nivel_alerta_final"].fillna("Sin nivel").isin(niveles)]
    if "probabilidad_riesgo_90d" in out.columns:
        out = out[out["probabilidad_riesgo_90d"].fillna(0) >= prob_min]
    if "saldo_expuesto" in out.columns:
        out = out[out["saldo_expuesto"].fillna(0) >= saldo_min]
    if "dias_sin_compensar" in out.columns:
        out = out[out["dias_sin_compensar"].fillna(0) >= dias_comp_min]
    if "dias_sin_transacciones" in out.columns:
        out = out[out["dias_sin_transacciones"].fillna(0) >= dias_trx_min]
    return out


def style_priority(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    fmt = {}
    if "probabilidad_riesgo_90d" in df.columns:
        fmt["probabilidad_riesgo_90d"] = lambda x: pct_fmt(x)
    if "saldo_expuesto" in df.columns:
        fmt["saldo_expuesto"] = lambda x: money_fmt(x)
    if "perdida_esperada_modelo" in df.columns:
        fmt["perdida_esperada_modelo"] = lambda x: money_fmt(x)
    if "consumo_cupo_visual" in df.columns:
        fmt["consumo_cupo_visual"] = lambda x: pct_fmt(x)
    if "dias_sin_compensar" in df.columns:
        fmt["dias_sin_compensar"] = lambda x: int_fmt(x)
    if "dias_sin_transacciones" in df.columns:
        fmt["dias_sin_transacciones"] = lambda x: int_fmt(x)

    def color_level(val):
        v = str(val)
        if "Crítico" in v or "prioritaria" in v:
            return (
                f"background-color: {PASTEL['red']}; color: #8A1F1F; font-weight: 700;"
            )
        if "Alto" in v:
            return f"background-color: {PASTEL['orange']}; color: #9A3D00; font-weight: 700;"
        if "Medio" in v:
            return f"background-color: {PASTEL['yellow']}; color: #755900; font-weight: 700;"
        if "Bajo" in v:
            return f"background-color: {PASTEL['green']}; color: #146C43; font-weight: 700;"
        return ""

    styler = df.style.format(fmt)

    # Compatibilidad Pandas 2.x / 3.x:
    # En Pandas 3 se eliminó Styler.applymap; el reemplazo es Styler.map.
    def aplicar_estilo_celda(styler_obj, funcion, subset):
        if hasattr(styler_obj, "map"):
            return styler_obj.map(funcion, subset=subset)
        return styler_obj.applymap(funcion, subset=subset)

    if "nivel_alerta_final" in df.columns:
        styler = aplicar_estilo_celda(
            styler,
            color_level,
            subset=["nivel_alerta_final"],
        )

    if "flag_nuevo_top500_preventivo" in df.columns:

        def color_nuevo_top500(x):
            try:
                flag = int(x or 0)
            except Exception:
                flag = 0
            return (
                f"background-color: {PASTEL['purple']}; color: #553C9A; font-weight: 700;"
                if flag == 1
                else ""
            )

        styler = aplicar_estilo_celda(
            styler,
            color_nuevo_top500,
            subset=["flag_nuevo_top500_preventivo"],
        )

    return styler


def download_df_button(df: pd.DataFrame, label: str, filename: str) -> None:
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )


def kpi_card(label: str, value: str, help_text: str = "", color: str = "blue") -> None:
    bg = PASTEL.get(color, PASTEL["blue"])
    st.markdown(
        f"""
        <div class="card" style="background: linear-gradient(135deg, {bg}, #FFFFFF);">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def decision_box(title: str, text: str, tag: str = "") -> None:
    tag_html = f"<span class='tag tag-blue'>{tag}</span>" if tag else ""
    st.markdown(
        f"""
        <div class="decision-box">
            <div class="decision-title">{title} {tag_html}</div>
            <div class="decision-text">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_bar(
    df: pd.DataFrame, x: str, y: str, title: str, orientation: str = "v"
) -> None:
    if df.empty or not PLOTLY_OK:
        st.dataframe(df, use_container_width=True)
        return
    fig = px.bar(df, x=x, y=y, title=title, orientation=orientation, text_auto=True)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PASTEL["text"]),
        margin=dict(l=10, r=10, t=55, b=10),
        height=370,
    )
    fig.update_traces(marker_line_width=0, opacity=0.88)
    st.plotly_chart(fig, use_container_width=True)


def chart_pie(df: pd.DataFrame, names: str, values: str, title: str) -> None:
    if df.empty or not PLOTLY_OK:
        st.dataframe(df, use_container_width=True)
        return
    fig = px.pie(df, names=names, values=values, title=title, hole=0.48)
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=PASTEL["text"]),
        margin=dict(l=10, r=10, t=55, b=10),
        height=370,
    )
    st.plotly_chart(fig, use_container_width=True)


def get_available_values(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns or df.empty:
        return []
    return sorted(df[col].fillna("Sin dato").astype(str).unique().tolist())


def get_display_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("nivel_alerta_final", dropna=False)
        .agg(
            puntos=("codigo_punto", "nunique"),
            saldo_expuesto=("saldo_expuesto", "sum"),
            perdida_esperada=("perdida_esperada_modelo", "sum"),
            prob_prom=("probabilidad_riesgo_90d", "mean"),
        )
        .reset_index()
        .sort_values("perdida_esperada", ascending=False)
    )
    return out


def top_geo(df: pd.DataFrame, geo_col: str, n: int = 12) -> pd.DataFrame:
    if df.empty or geo_col not in df.columns:
        return pd.DataFrame()
    return (
        df.groupby(geo_col, dropna=False)
        .agg(
            puntos=("codigo_punto", "nunique"),
            perdida_esperada=("perdida_esperada_modelo", "sum"),
            saldo_expuesto=("saldo_expuesto", "sum"),
            prob_prom=("probabilidad_riesgo_90d", "mean"),
        )
        .reset_index()
        .sort_values("perdida_esperada", ascending=False)
        .head(n)
    )


# ============================================================
# SIDEBAR Y CARGA
# ============================================================

st.sidebar.markdown("### ⚙️ Configuración")
base_dir = Path(
    st.sidebar.text_input("Carpeta salidas producción", str(BASE_DIR_DEFAULT))
)

preventivo_path = base_dir / "scoring_preventivo_activos.parquet"
recuperacion_path = base_dir / "scoring_recuperacion_cerrados.parquet"
resumen_path = base_dir / "resumen_operativo_dashboard.csv"
control_path = base_dir / "control_calidad_scoring.csv"
actual_path = base_dir / "scoring_actual_cb_limpio.parquet"

with st.spinner("Cargando salidas del modelo..."):
    preventivo = ensure_visual_columns(load_parquet(str(preventivo_path)))
    recuperacion = ensure_visual_columns(load_parquet(str(recuperacion_path)))
    scoring_actual = ensure_visual_columns(load_parquet(str(actual_path)))
    resumen = load_csv(str(resumen_path))
    control = load_csv(str(control_path))

if preventivo.empty and recuperacion.empty:
    st.error(
        "No se encontraron salidas del modelo. Verifica la ruta de la carpeta salidas_produccion "
        "y que existan los archivos scoring_preventivo_activos.parquet y scoring_recuperacion_cerrados.parquet."
    )
    st.stop()

st.sidebar.markdown("### 🎯 Filtros")
base_filtros = pd.concat([preventivo, recuperacion], ignore_index=True, sort=False)

deps = st.sidebar.multiselect(
    "Departamento", get_available_values(base_filtros, "departamento")
)
mpios = st.sidebar.multiselect(
    "Municipio", get_available_values(base_filtros, "municipio")
)
redes = st.sidebar.multiselect("Red CB", get_available_values(base_filtros, "red_cb"))
niveles = st.sidebar.multiselect(
    "Nivel alerta", get_available_values(base_filtros, "nivel_alerta_final")
)
prob_min = st.sidebar.slider("Probabilidad mínima", 0.0, 1.0, 0.0, 0.01)
saldo_min = st.sidebar.number_input(
    "Saldo expuesto mínimo", min_value=0.0, value=0.0, step=500000.0
)
dias_comp_min = st.sidebar.number_input(
    "Días sin compensar mínimo", min_value=0, value=0, step=1
)
dias_trx_min = st.sidebar.number_input(
    "Días sin transacciones mínimo", min_value=0, value=0, step=1
)

top_n = st.sidebar.slider(
    "Cantidad de casos prioritarios", min_value=50, max_value=5000, value=500, step=50
)

preventivo_f = filter_df(
    preventivo,
    deps,
    mpios,
    redes,
    niveles,
    prob_min,
    saldo_min,
    dias_comp_min,
    dias_trx_min,
)
recuperacion_f = filter_df(
    recuperacion,
    deps,
    mpios,
    redes,
    niveles,
    prob_min,
    saldo_min,
    dias_comp_min,
    dias_trx_min,
)

# Orden principal por pérdida esperada y probabilidad.
for _df_name in ["preventivo_f", "recuperacion_f"]:
    _df = locals()[_df_name]
    if not _df.empty:
        locals()[_df_name] = _df.sort_values(
            ["perdida_esperada_modelo", "probabilidad_riesgo_90d"],
            ascending=[False, False],
        )

preventivo_top = preventivo_f.head(top_n).copy()
recuperacion_top = recuperacion_f.head(top_n).copy()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    "<div class='main-title'>🛡️ Tablero de Riesgo CB | Decisión Preventiva</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Modelo predictivo + reglas actuales + saldo expuesto para priorizar gestión y prevenir pérdida por no compensación.</div>",
    unsafe_allow_html=True,
)

col_status1, col_status2, col_status3, col_status4 = st.columns(4)
with col_status1:
    kpi_card(
        "CB preventivos",
        int_fmt(
            preventivo_f["codigo_punto"].nunique()
            if "codigo_punto" in preventivo_f
            else 0
        ),
        "ACTIVO / PROCESO DE CIERRE",
        "blue",
    )
with col_status2:
    kpi_card(
        "Top gestión seleccionado",
        int_fmt(len(preventivo_top)),
        "Casos priorizados por pérdida esperada",
        "purple",
    )
with col_status3:
    kpi_card(
        "Saldo expuesto preventivo",
        money_fmt(preventivo_f.get("saldo_expuesto", pd.Series(dtype=float)).sum()),
        "Impacto económico actual",
        "orange",
    )
with col_status4:
    kpi_card(
        "Pérdida esperada",
        money_fmt(
            preventivo_f.get("perdida_esperada_modelo", pd.Series(dtype=float)).sum()
        ),
        "Probabilidad x saldo expuesto",
        "red",
    )

st.markdown("---")


# ============================================================
# DECISIONES RÁPIDAS
# ============================================================

st.markdown(
    "<div class='section-title'>⚡ Decisiones rápidas para gestión</div>",
    unsafe_allow_html=True,
)

criticos = (
    preventivo_f[
        preventivo_f.get("nivel_alerta_final", "").astype(str).eq("Crítico")
    ].copy()
    if not preventivo_f.empty
    else pd.DataFrame()
)
altos = (
    preventivo_f[
        preventivo_f.get("nivel_alerta_final", "").astype(str).eq("Alto")
    ].copy()
    if not preventivo_f.empty
    else pd.DataFrame()
)
nuevos = (
    preventivo_f[
        preventivo_f.get("flag_nuevo_top500_preventivo", 0).fillna(0).astype(int).eq(1)
    ].copy()
    if "flag_nuevo_top500_preventivo" in preventivo_f
    else pd.DataFrame()
)
sin_comp_30 = (
    preventivo_f[preventivo_f.get("dias_sin_compensar", 0).fillna(0).ge(30)].copy()
    if "dias_sin_compensar" in preventivo_f
    else pd.DataFrame()
)
inactivos_30 = (
    preventivo_f[preventivo_f.get("dias_sin_transacciones", 0).fillna(0).ge(30)].copy()
    if "dias_sin_transacciones" in preventivo_f
    else pd.DataFrame()
)

c1, c2, c3 = st.columns(3)
with c1:
    decision_box(
        "Gestionar hoy",
        f"{int_fmt(len(criticos))} CB preventivos en nivel crítico. Priorizar llamadas, validación de saldo y posible bloqueo preventivo.",
        "Crítico",
    )
with c2:
    decision_box(
        "Enviar al gestor",
        f"Descarga el Top {top_n} preventivo para gestión. La prioridad está ordenada por pérdida esperada y probabilidad.",
        "Gestión",
    )
with c3:
    decision_box(
        "Nuevos casos en Top 500",
        f"{int_fmt(len(nuevos))} CB entraron nuevos al Top 500 preventivo frente a la corrida anterior.",
        "Nuevo",
    )

c4, c5, c6 = st.columns(3)
with c4:
    kpi_card(
        "Sin compensar ≥ 30 días",
        int_fmt(len(sin_comp_30)),
        "Señal operativa fuerte",
        "red",
    )
with c5:
    kpi_card(
        "Inactivos ≥ 30 días",
        int_fmt(len(inactivos_30)),
        "Riesgo por caída operativa",
        "yellow",
    )
with c6:
    kpi_card(
        "Nivel alto", int_fmt(len(altos)), "Siguiente grupo de seguimiento", "orange"
    )

st.markdown("---")


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🚨 Gestión preventiva",
        "📊 Resumen ejecutivo",
        "📍 Zonas y redes",
        "💰 Recuperación",
        "✅ Calidad del modelo",
        "⬇️ Descargas",
    ]
)


# ============================================================
# TAB 1: GESTIÓN PREVENTIVA
# ============================================================

with tab1:
    st.markdown(
        "<div class='section-title'>🚨 Bandeja priorizada de gestión preventiva</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Universo: ACTIVO y PROCESO DE CIERRE. Ordenado por pérdida esperada y probabilidad del modelo."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Casos filtrados", int_fmt(len(preventivo_f)))
    with c2:
        st.metric("Top seleccionado", int_fmt(len(preventivo_top)))
    with c3:
        st.metric(
            "Prob. promedio Top",
            pct_fmt(
                preventivo_top.get(
                    "probabilidad_riesgo_90d", pd.Series(dtype=float)
                ).mean()
            ),
        )
    with c4:
        st.metric(
            "Pérdida esperada Top",
            money_fmt(
                preventivo_top.get(
                    "perdida_esperada_modelo", pd.Series(dtype=float)
                ).sum()
            ),
        )

    cols = get_display_columns(preventivo_top, DISPLAY_COLUMNS_PREVENTIVO)
    st.dataframe(
        style_priority(preventivo_top[cols]), use_container_width=True, height=520
    )

    col_d1, col_d2 = st.columns([1, 1])
    with col_d1:
        download_df_button(
            preventivo_top[cols],
            f"⬇️ Descargar Top {top_n} para gestor",
            f"top_{top_n}_gestion_preventiva.csv",
        )
    with col_d2:
        critical_export = preventivo_f[
            preventivo_f["nivel_alerta_final"].astype(str).isin(["Crítico", "Alto"])
        ].copy()
        cols2 = get_display_columns(critical_export, DISPLAY_COLUMNS_PREVENTIVO)
        download_df_button(
            critical_export[cols2],
            "⬇️ Descargar Críticos y Altos",
            "criticos_altos_preventivo.csv",
        )

    st.markdown("#### Lectura sugerida")
    st.info(
        "Primero gestione los casos con mayor pérdida esperada. Si el CB está crítico, con muchos días sin compensar e inactivo, "
        "debe enviarse a gestión inmediata para validar saldo, contactar al punto y evaluar bloqueo preventivo."
    )


# ============================================================
# TAB 2: RESUMEN EJECUTIVO
# ============================================================

with tab2:
    st.markdown(
        "<div class='section-title'>📊 Resumen ejecutivo preventivo</div>",
        unsafe_allow_html=True,
    )

    summary = risk_summary(preventivo_f)
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if not summary.empty:
            chart_bar(
                summary, "nivel_alerta_final", "puntos", "Puntos por nivel de alerta"
            )
    with col_g2:
        if not summary.empty:
            chart_pie(
                summary,
                "nivel_alerta_final",
                "perdida_esperada",
                "Distribución de pérdida esperada",
            )

    st.markdown("#### Resumen por nivel")
    if not summary.empty:
        display_summary = summary.copy()
        display_summary["saldo_expuesto"] = display_summary["saldo_expuesto"].map(
            money_fmt
        )
        display_summary["perdida_esperada"] = display_summary["perdida_esperada"].map(
            money_fmt
        )
        display_summary["prob_prom"] = display_summary["prob_prom"].map(pct_fmt)
        st.dataframe(display_summary, use_container_width=True)
    else:
        st.warning("No hay datos para el resumen con los filtros seleccionados.")

    st.markdown("#### Distribución de probabilidad")
    if PLOTLY_OK and not preventivo_f.empty:
        fig = px.histogram(
            preventivo_f,
            x="probabilidad_riesgo_90d",
            nbins=40,
            title="Distribución de probabilidad de riesgo 90 días",
        )
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
    elif not preventivo_f.empty:
        st.dataframe(
            preventivo_f[["probabilidad_riesgo_90d"]].describe(),
            use_container_width=True,
        )


# ============================================================
# TAB 3: ZONAS Y REDES
# ============================================================

with tab3:
    st.markdown(
        "<div class='section-title'>📍 Zonas, municipios y redes con mayor exposición</div>",
        unsafe_allow_html=True,
    )

    geo1, geo2 = st.columns(2)
    with geo1:
        dept = top_geo(preventivo_f, "departamento", 12)
        if not dept.empty:
            chart_bar(
                dept.sort_values("perdida_esperada"),
                "perdida_esperada",
                "departamento",
                "Top departamentos por pérdida esperada",
                orientation="h",
            )
    with geo2:
        mpio = top_geo(preventivo_f, "municipio", 12)
        if not mpio.empty:
            chart_bar(
                mpio.sort_values("perdida_esperada"),
                "perdida_esperada",
                "municipio",
                "Top municipios por pérdida esperada",
                orientation="h",
            )

    red_col1, red_col2 = st.columns(2)
    with red_col1:
        red = top_geo(preventivo_f, "red_cb", 12)
        if not red.empty:
            chart_bar(
                red.sort_values("perdida_esperada"),
                "perdida_esperada",
                "red_cb",
                "Top redes por pérdida esperada",
                orientation="h",
            )
    with red_col2:
        if (
            not preventivo_f.empty
            and "departamento" in preventivo_f.columns
            and "nivel_alerta_final" in preventivo_f.columns
        ):
            heat = (
                preventivo_f.groupby(
                    ["departamento", "nivel_alerta_final"], dropna=False
                )
                .agg(puntos=("codigo_punto", "nunique"))
                .reset_index()
            )
            if PLOTLY_OK and not heat.empty:
                fig = px.treemap(
                    heat,
                    path=["nivel_alerta_final", "departamento"],
                    values="puntos",
                    title="Concentración por nivel y departamento",
                )
                fig.update_layout(height=370, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.dataframe(heat, use_container_width=True)

    st.markdown("#### Detalle por municipio")
    if not mpio.empty:
        mpio_show = mpio.copy()
        mpio_show["perdida_esperada"] = mpio_show["perdida_esperada"].map(money_fmt)
        mpio_show["saldo_expuesto"] = mpio_show["saldo_expuesto"].map(money_fmt)
        mpio_show["prob_prom"] = mpio_show["prob_prom"].map(pct_fmt)
        st.dataframe(mpio_show, use_container_width=True)


# ============================================================
# TAB 4: RECUPERACIÓN
# ============================================================

with tab4:
    st.markdown(
        "<div class='section-title'>💰 Flujo recuperación / cartera</div>",
        unsafe_allow_html=True,
    )
    st.caption("Universo: CERRADO y CERRADO CON SALDO. No mezclar con prevención.")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Casos recuperación", int_fmt(len(recuperacion_f)))
    with r2:
        st.metric(
            "Saldo expuesto",
            money_fmt(
                recuperacion_f.get("saldo_expuesto", pd.Series(dtype=float)).sum()
            ),
        )
    with r3:
        st.metric(
            "Pérdida esperada",
            money_fmt(
                recuperacion_f.get(
                    "perdida_esperada_modelo", pd.Series(dtype=float)
                ).sum()
            ),
        )
    with r4:
        st.metric(
            "Prob. promedio",
            pct_fmt(
                recuperacion_f.get(
                    "probabilidad_riesgo_90d", pd.Series(dtype=float)
                ).mean()
            ),
        )

    cols_rec = get_display_columns(recuperacion_top, DISPLAY_COLUMNS_RECUPERACION)
    st.dataframe(
        style_priority(recuperacion_top[cols_rec]), use_container_width=True, height=520
    )
    download_df_button(
        recuperacion_top[cols_rec],
        f"⬇️ Descargar Top {top_n} recuperación",
        f"top_{top_n}_recuperacion.csv",
    )


# ============================================================
# TAB 5: CALIDAD
# ============================================================

with tab5:
    st.markdown(
        "<div class='section-title'>✅ Calidad de datos y controles del scoring</div>",
        unsafe_allow_html=True,
    )

    if not control.empty:
        st.dataframe(control, use_container_width=True)
    else:
        st.warning("No se encontró control_calidad_scoring.csv")

    quality_checks = []
    if not scoring_actual.empty:
        quality_checks.append(
            (
                "Probabilidades nulas",
                int(
                    scoring_actual.get(
                        "probabilidad_riesgo_90d", pd.Series(dtype=float)
                    )
                    .isna()
                    .sum()
                ),
            )
        )
        quality_checks.append(
            (
                "Duplicados codigo_punto",
                (
                    int(scoring_actual.duplicated("codigo_punto").sum())
                    if "codigo_punto" in scoring_actual
                    else 0
                ),
            )
        )
        quality_checks.append(
            (
                "Días sin transacciones nulos",
                int(
                    scoring_actual.get("dias_sin_transacciones", pd.Series(dtype=float))
                    .isna()
                    .sum()
                ),
            )
        )
        quality_checks.append(
            (
                "Consumo visual negativo",
                int(
                    (
                        scoring_actual.get(
                            "consumo_cupo_visual", pd.Series(dtype=float)
                        ).fillna(0)
                        < 0
                    ).sum()
                ),
            )
        )
    quality_df = pd.DataFrame(quality_checks, columns=["control", "valor"])

    st.markdown("#### Validaciones rápidas")
    if not quality_df.empty:
        st.dataframe(quality_df, use_container_width=True)

    st.markdown("#### Interpretación")
    st.success(
        "El tablero usa la salida productiva del modelo. No entrena ni modifica el modelo. "
        "Para actualizar datos, ejecuta primero el pipeline SQL y luego el scoring/entrenamiento según tu calendario operativo."
    )


# ============================================================
# TAB 6: DESCARGAS
# ============================================================

with tab6:
    st.markdown(
        "<div class='section-title'>⬇️ Descargas para gestión</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        cols = get_display_columns(preventivo_f, DISPLAY_COLUMNS_PREVENTIVO)
        download_df_button(
            preventivo_f[cols], "⬇️ Preventivo filtrado", "preventivo_filtrado.csv"
        )
    with c2:
        cols = get_display_columns(preventivo_top, DISPLAY_COLUMNS_PREVENTIVO)
        download_df_button(
            preventivo_top[cols],
            f"⬇️ Top {top_n} preventivo",
            f"top_{top_n}_preventivo.csv",
        )
    with c3:
        cols = get_display_columns(recuperacion_f, DISPLAY_COLUMNS_RECUPERACION)
        download_df_button(
            recuperacion_f[cols],
            "⬇️ Recuperación filtrada",
            "recuperacion_filtrada.csv",
        )

    st.markdown("#### Archivo sugerido para gestor")
    gestor_cols = [
        "ranking_segmento",
        "codigo_punto",
        "nombre_punto",
        "estado_actual",
        "departamento",
        "municipio",
        "red_cb",
        "nivel_alerta_final",
        "probabilidad_riesgo_90d",
        "saldo_expuesto",
        "perdida_esperada_modelo",
        "dias_sin_compensar",
        "dias_sin_transacciones",
        "motivo_alerta",
        "accion_final_produccion",
    ]
    gestor_cols = get_display_columns(preventivo_top, gestor_cols)
    st.dataframe(
        style_priority(preventivo_top[gestor_cols]),
        use_container_width=True,
        height=420,
    )
    download_df_button(
        preventivo_top[gestor_cols],
        "📤 Descargar archivo para envío a gestores",
        "archivo_para_gestores_preventivo.csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Uso recomendado: priorizar Top preventivo por pérdida esperada. El modelo predice riesgo futuro; las reglas actuales indican urgencia operativa; el saldo expuesto mide impacto financiero."
)
