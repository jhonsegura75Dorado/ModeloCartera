"""
Pipeline robusto para generar datos del modelo de riesgo de apropiación / no compensación CB.

Objetivo
--------
Construir una base segura para entrenamiento y una base operativa actual de alertas.
El pipeline evita fuga de información, interpreta correctamente las variables de castigados
 y permite enriquecer con histórico real de compensaciones mediante snapshots.

Reglas clave
------------
1. SALCOD  -> codigo_punto.
2. SALFEC  -> fecha_ultima_compensacion_sql.
3. SALDIS  -> saldo_punto_sql / balance actual.
4. SALCUPO -> cupo_punto_saldo.
5. En el archivo de castigados:
   - Valor Castigado = valor original que quedó debiendo.
   - Saldo Actual / saldo_punto = saldo pendiente actual después de pagos.
6. Las columnas de castigados y de evento NO se usan como variables predictoras.
7. La compensación histórica solo es válida si proviene de snapshots guardados en el tiempo
   o de una tabla histórica real. No se reconstruye hacia atrás usando la foto actual.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyodbc

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# CONFIGURACIÓN
# ============================================================

DSN_IMPALA = os.getenv("DSN_IMPALA", "impala-prod")
DSN_NACIONAL = os.getenv("DSN_NACIONAL", "NACIONALET01")

# No quemar claves en el código. Definir en PowerShell:
# setx NACIONAL_UID "TU_USUARIO"
# setx NACIONAL_PWD "TU_CLAVE"
NACIONAL_UID = os.getenv("NACIONAL_UID", "NJASEGURA")
NACIONAL_PWD = os.getenv("NACIONAL_PWD", "TOR2925E")

FECHA_INICIO = os.getenv("FECHA_INICIO", "2025-01-01")
FECHA_FIN = os.getenv(
    "FECHA_FIN",
    (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
)

# Ajusta estas rutas a tu equipo.
RUTA_CBS_CASTIGADOS = os.getenv(
    "RUTA_CBS_CASTIGADOS",
    r"C:\Users\jasegura\Downloads\Información CBs Castigados 2025-2026 (1).xlsx",
)
RUTA_ULTIMA_TRX_COMP = os.getenv(
    "RUTA_ULTIMA_TRX_COMP",
    r"C:\Users\jasegura\Downloads\ULTIMA TRX Y COMPENSACION LZ - 20260430.xlsx",
)

# Archivo opcional simple con una sola fila por punto:
# codigo_punto | fecha  (fecha en formato yyyymmdd, ejemplo 20251229)
# IMPORTANTE: esta fecha final solo se usa para ALERTA ACTUAL, no para reconstruir historia.
RUTA_ULT_COMPENSACION_SIMPLE = os.getenv(
    "RUTA_ULT_COMPENSACION_SIMPLE",
    r"C:\Users\jasegura\Documents\ult_compensacion.xlsx",
)

SALIDA_DIR = Path(os.getenv("SALIDA_DIR", "_modelo_riesgo_cb"))
SNAPSHOT_SALDOS_DIR = SALIDA_DIR / "_snapshots_saldos_compensacion"
SALIDA_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_SALDOS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_TRX = int(os.getenv("BATCH_TRX", "6000"))
BATCH_SALDOS = int(os.getenv("BATCH_SALDOS", "1000"))
EXPORTAR_EXCEL_ALERTAS = True


# ============================================================
# CONEXIONES
# ============================================================


def conectar_impala():
    return pyodbc.connect(DSN=DSN_IMPALA, autocommit=True)


def conectar_nacional():
    if NACIONAL_UID and NACIONAL_PWD:
        return pyodbc.connect(
            f"DSN={DSN_NACIONAL};UID={NACIONAL_UID};PWD={NACIONAL_PWD}"
        )
    return pyodbc.connect(f"DSN={DSN_NACIONAL}")


# ============================================================
# UTILIDADES
# ============================================================


def dividir_en_batches(lista: list[int], tamano_batch: int) -> Iterable[list[int]]:
    for i in range(0, len(lista), tamano_batch):
        yield lista[i : i + tamano_batch]


def normalizar_texto_columna(col: str) -> str:
    col = str(col).strip().lower()
    reemplazos = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ñ": "n",
    }
    for k, v in reemplazos.items():
        col = col.replace(k, v)
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalizar_texto_columna(c) for c in df.columns]
    return df


def normalizar_codigo_punto(serie: pd.Series) -> pd.Series:
    s = serie.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.replace(["", "nan", "NaN", "None", "<NA>", "null", "NULL"], np.nan)
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def normalizar_monto(serie: pd.Series) -> pd.Series:
    """Convierte valores monetarios de Excel/SQL a numérico."""
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce").fillna(0)

    s = serie.astype(str).str.strip()
    s = s.replace(["", "nan", "NaN", "None", "<NA>", "null", "NULL"], np.nan)
    s = s.str.replace("$", "", regex=False)
    s = s.str.replace(" ", "", regex=False)

    # Caso frecuente Colombia: 29.044.110 o 29,044,110.
    # Si aparecen ambos separadores, asumimos coma decimal si está al final con 2 decimales.
    def limpiar_valor(x):
        if pd.isna(x):
            return np.nan
        x = str(x)
        if "," in x and "." in x:
            # 1.234.567,89 -> 1234567.89
            if re.search(r",\d{1,2}$", x):
                x = x.replace(".", "").replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "," in x:
            # 29,044,110 -> 29044110; 1234,56 -> 1234.56
            if re.search(r",\d{1,2}$", x) and x.count(",") == 1:
                x = x.replace(",", ".")
            else:
                x = x.replace(",", "")
        elif "." in x:
            # 29.044.110 -> 29044110; 1234.56 -> 1234.56
            if x.count(".") > 1:
                x = x.replace(".", "")
        return x

    return pd.to_numeric(s.map(limpiar_valor), errors="coerce").fillna(0)


def parse_fecha_flexible(serie: pd.Series) -> pd.Series:
    """Soporta datetime, yyyy-mm-dd, yyyymmdd y seriales de Excel."""
    if pd.api.types.is_datetime64_any_dtype(serie):
        return pd.to_datetime(serie, errors="coerce")

    s = serie.copy()
    s = s.replace([0, "0", "", "nan", "NaN", "None", "<NA>", None], np.nan)

    # Intento yyyymmdd cuando son 8 dígitos.
    s_str = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    mask_yyyymmdd = s_str.str.match(r"^\d{8}$", na=False)

    out = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")
    if mask_yyyymmdd.any():
        out.loc[mask_yyyymmdd] = pd.to_datetime(
            s_str.loc[mask_yyyymmdd], format="%Y%m%d", errors="coerce"
        )

    # Serial Excel razonable.
    s_num = pd.to_numeric(s_str, errors="coerce")
    mask_excel = out.isna() & s_num.between(20000, 60000)
    if mask_excel.any():
        out.loc[mask_excel] = pd.to_datetime(
            s_num.loc[mask_excel], unit="D", origin="1899-12-30", errors="coerce"
        )

    # Fechas normales.
    mask_rest = out.isna()
    if mask_rest.any():
        out.loc[mask_rest] = pd.to_datetime(s.loc[mask_rest], errors="coerce")

    return out


def crear_fecha(
    df: pd.DataFrame, col_anio: str, col_mes: str, col_dia: str
) -> pd.Series:
    fecha_partes = pd.DataFrame(
        {
            "year": pd.to_numeric(df.get(col_anio), errors="coerce"),
            "month": pd.to_numeric(df.get(col_mes), errors="coerce"),
            "day": pd.to_numeric(df.get(col_dia), errors="coerce"),
        }
    )
    return pd.to_datetime(fecha_partes, errors="coerce")


def guardar_parquet_seguro(df: pd.DataFrame, ruta: Path) -> None:
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(ruta, index=False)
        print(f"OK parquet: {ruta}")
    except Exception as e:
        ruta_csv = ruta.with_suffix(".csv")
        df.to_csv(ruta_csv, index=False, encoding="utf-8-sig")
        print(f"No fue posible guardar parquet ({e}). Se guardó CSV: {ruta_csv}")


def existe_archivo(ruta: str | Path) -> bool:
    return bool(ruta) and Path(ruta).exists()


# ============================================================
# BASE DE FUNCIONAMIENTO
# ============================================================


def obtener_base_funcionamiento_actual(conexion) -> pd.DataFrame:
    query = """
        WITH last_ingestion AS (
            SELECT year, ingestion_month, ingestion_day
            FROM resultados_vspc_canales.fco_base_funcionamiento
            ORDER BY year DESC, ingestion_month DESC, ingestion_day DESC
            LIMIT 1
        )
        SELECT
            t1.ingestion_year,
            t1.ingestion_month,
            t1.ingestion_day,
            t1.codigo_corresponsal,
            t1.codigo_punto,
            t1.codigo_dane,
            t1.region,
            t1.departamento,
            t1.municipio,
            t1.corregimiento,
            t1.nombre_punto,
            t1.formato_punto,
            t1.categoria,
            t1.tipo_persona,
            t1.regimen,
            t1.numero_datafonos,
            t1.red_cb,
            t1.cupo_actual,
            t1.anio_implementacion,
            t1.mes_implementacion,
            t1.dia_implementacion,
            t1.estado,
            t1.anio_cierre,
            t1.mes_cierre,
            t1.dia_cierre,
            t1.fuerza_comercial,
            t1.canal_comunicacion,
            t1.year
        FROM resultados_vspc_canales.fco_base_funcionamiento t1
        INNER JOIN last_ingestion t2
            ON  t1.year            = t2.year
            AND t1.ingestion_month = t2.ingestion_month
            AND t1.ingestion_day   = t2.ingestion_day
        WHERE t1.categoria = 'UNO A UNO'
    """
    df = pd.read_sql(query, conexion)
    df = normalizar_columnas(df)
    df["codigo_punto"] = normalizar_codigo_punto(df["codigo_punto"])
    df = df.dropna(subset=["codigo_punto"]).copy()
    df["codigo_punto"] = df["codigo_punto"].astype(int)
    df["fecha_implementacion"] = crear_fecha(
        df, "anio_implementacion", "mes_implementacion", "dia_implementacion"
    )
    df["fecha_cierre"] = crear_fecha(df, "anio_cierre", "mes_cierre", "dia_cierre")
    df = df.rename(columns={"estado": "estado_actual"})
    return df.drop_duplicates("codigo_punto")


def obtener_base_funcionamiento_historica_mensual(
    conexion, fecha_inicio: str, fecha_fin: str
) -> pd.DataFrame:
    """
    Toma el último snapshot disponible de cada mes y trae TODOS los CB de ese snapshot.

    Corrección clave:
    - No se usa ROW_NUMBER() particionado solo por mes, porque eso devuelve una sola fila por mes.
    - Primero se identifica el último día de ingestión de cada mes.
    - Luego se hace JOIN para traer todos los puntos UNO A UNO de ese corte mensual.
    """
    ini = pd.to_datetime(fecha_inicio)
    fin = pd.to_datetime(fecha_fin)

    query = f"""
        WITH cortes_mes AS (
            SELECT
                ingestion_year,
                ingestion_month,
                MAX(ingestion_day) AS ingestion_day
            FROM resultados_vspc_canales.fco_base_funcionamiento
            WHERE categoria = 'UNO A UNO'
              AND CAST(CONCAT(
                    CAST(ingestion_year AS STRING), '-',
                    LPAD(CAST(ingestion_month AS STRING), 2, '0'), '-',
                    LPAD(CAST(ingestion_day AS STRING), 2, '0')
                  ) AS DATE) BETWEEN CAST('{ini.strftime('%Y-%m-%d')}' AS DATE)
                                  AND CAST('{fin.strftime('%Y-%m-%d')}' AS DATE)
            GROUP BY ingestion_year, ingestion_month
        )
        SELECT
            t.ingestion_year,
            t.ingestion_month,
            t.ingestion_day,
            t.codigo_corresponsal,
            t.codigo_punto,
            t.codigo_dane,
            t.region,
            t.departamento,
            t.municipio,
            t.corregimiento,
            t.nombre_punto,
            t.formato_punto,
            t.categoria,
            t.tipo_persona,
            t.regimen,
            t.numero_datafonos,
            t.red_cb,
            t.cupo_actual,
            t.anio_implementacion,
            t.mes_implementacion,
            t.dia_implementacion,
            t.estado,
            t.anio_cierre,
            t.mes_cierre,
            t.dia_cierre,
            t.fuerza_comercial,
            t.canal_comunicacion,
            t.year
        FROM resultados_vspc_canales.fco_base_funcionamiento t
        INNER JOIN cortes_mes c
            ON  t.ingestion_year  = c.ingestion_year
            AND t.ingestion_month = c.ingestion_month
            AND t.ingestion_day   = c.ingestion_day
        WHERE t.categoria = 'UNO A UNO'
    """

    df = pd.read_sql(query, conexion)
    df = normalizar_columnas(df)

    if df.empty:
        return df

    df["codigo_punto"] = normalizar_codigo_punto(df["codigo_punto"])
    df = df.dropna(subset=["codigo_punto"]).copy()
    df["codigo_punto"] = df["codigo_punto"].astype(int)

    df["fecha_implementacion"] = crear_fecha(
        df, "anio_implementacion", "mes_implementacion", "dia_implementacion"
    )
    df["fecha_cierre"] = crear_fecha(df, "anio_cierre", "mes_cierre", "dia_cierre")
    df["fecha_corte"] = crear_fecha(
        df, "ingestion_year", "ingestion_month", "ingestion_day"
    )
    df["periodo"] = df["fecha_corte"].dt.to_period("M")
    df = df.rename(columns={"estado": "estado_al_corte"})

    df = df.sort_values(["periodo", "codigo_punto", "fecha_corte"])
    df = df.drop_duplicates(["codigo_punto", "periodo"], keep="last")

    return df


# ============================================================
# TRANSACCIONALIDAD HISTÓRICA MENSUAL
# ============================================================


def consultar_trx_mensual_por_batch(
    conexion, lista_puntos: list[int], fecha_inicio: str, fecha_fin: str
) -> pd.DataFrame:
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_dt = pd.to_datetime(fecha_fin)
    start_year = fecha_inicio_dt.year
    end_year = fecha_fin_dt.year
    resultados = []

    for idx, batch in enumerate(dividir_en_batches(lista_puntos, BATCH_TRX), start=1):
        tupla_sql = ",".join(str(int(x)) for x in batch)
        query = f"""
            WITH transactions AS (
                SELECT
                    CNCODPUS AS codigo_punto,
                    CNANORED AS ano,
                    CNMESRED AS mes,
                    CAST(CONCAT(
                        CAST(CNANORED AS STRING), '-',
                        LPAD(CAST(CNMESRED AS STRING), 2, '0'), '-',
                        LPAD(CAST(CNDIARED AS STRING), 2, '0')
                    ) AS DATE) AS fecha_trx,
                    CAST(cntiptrx AS INT) AS cntiptrx,
                    CAST(cnesttrx AS INT) AS cnesttrx,
                    CASE WHEN CNCODTRX = 782 THEN 0 ELSE CAST(CNVALTRX AS DOUBLE) END AS valor
                FROM s_canales.sai_scilibramd_sciffmovcn
                WHERE year BETWEEN {start_year} AND {end_year}
                  AND CNANORED BETWEEN {start_year} AND {end_year}
                  AND CNCODTRX NOT IN (702, 328, 760)
                  AND CNCODPUS IN ({tupla_sql})
                  AND CAST(CONCAT(
                        CAST(CNANORED AS STRING), '-',
                        LPAD(CAST(CNMESRED AS STRING), 2, '0'), '-',
                        LPAD(CAST(CNDIARED AS STRING), 2, '0')
                    ) AS DATE) BETWEEN CAST('{fecha_inicio_dt.strftime('%Y-%m-%d')}' AS DATE)
                                    AND CAST('{fecha_fin_dt.strftime('%Y-%m-%d')}' AS DATE)
            )
            SELECT
                codigo_punto,
                ano,
                mes,
                COUNT(1) AS total_trx,
                SUM(CASE WHEN cntiptrx = 1 AND cnesttrx = 1 THEN 1 ELSE 0 END) AS cnt_aprobada,
                SUM(CASE WHEN cntiptrx = 1 AND cnesttrx = 2 THEN 1 ELSE 0 END) AS cnt_rechazada,
                SUM(CASE WHEN cntiptrx = 2 AND cnesttrx = 1 THEN 1 ELSE 0 END) AS cnt_reversada,
                SUM(valor) AS valor_total,
                AVG(valor) AS ticket_promedio_mes,
                MIN(fecha_trx) AS fecha_primera_trx_mes,
                MAX(fecha_trx) AS fecha_ultima_trx_mes
            FROM transactions
            GROUP BY codigo_punto, ano, mes
        """
        print(f"Consultando TRX mensual batch {idx} con {len(batch)} puntos...")
        resultados.append(normalizar_columnas(pd.read_sql(query, conexion)))

    if not resultados:
        return pd.DataFrame()

    df = pd.concat(resultados, ignore_index=True)
    df["codigo_punto"] = normalizar_codigo_punto(df["codigo_punto"])
    df = df.dropna(subset=["codigo_punto"]).copy()
    df["codigo_punto"] = df["codigo_punto"].astype(int)
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
    df["periodo"] = pd.to_datetime(
        df["ano"].astype(str) + "-" + df["mes"].astype(str).str.zfill(2) + "-01",
        errors="coerce",
    ).dt.to_period("M")
    for col in ["fecha_primera_trx_mes", "fecha_ultima_trx_mes"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in [
        "total_trx",
        "cnt_aprobada",
        "cnt_rechazada",
        "cnt_reversada",
        "valor_total",
        "ticket_promedio_mes",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["tasa_aprobada"] = np.where(
        df["total_trx"] > 0, df["cnt_aprobada"] / df["total_trx"], 0
    )
    df["tasa_rechazada"] = np.where(
        df["total_trx"] > 0, df["cnt_rechazada"] / df["total_trx"], 0
    )
    df["tasa_reversada"] = np.where(
        df["total_trx"] > 0, df["cnt_reversada"] / df["total_trx"], 0
    )
    return df


# ============================================================
# SALDOS / COMPENSACIÓN SQL ACTUAL + SNAPSHOTS
# ============================================================


def consultar_saldos_actuales(conexion, lista_puntos: list[int]) -> pd.DataFrame:
    """
    Consulta la foto actual de compensación/saldos.

    Variables SQL:
    - SALCOD -> codigo_punto
    - SALFEC -> fecha_ultima_compensacion_sql
    - SALDIS -> saldo_punto_sql
    - SALCUPO -> cupo_punto_saldo
    """
    resultados = []
    for idx, batch in enumerate(
        dividir_en_batches(lista_puntos, BATCH_SALDOS), start=1
    ):
        placeholders = ",".join(["?"] * len(batch))
        query = f"""
            SELECT
                SALCOD   AS codigo_punto,
                SALCOCNB AS codigo_cnb,
                SALCUPO  AS cupo_punto_saldo,
                SALENT   AS entradas_punto,
                SALSAL   AS salidas_punto,
                SALDIS   AS saldo_punto_sql,
                SALFEC   AS fecha_ultima_compensacion_sql,
                SALBAN   AS compensado_favor_banco,
                SALCNB   AS compensado_favor_punto
            FROM SCILIBRAMD.SCIFFINSAL
            WHERE SALCOD IN ({placeholders})
        """
        print(
            f"Consultando saldos/compensación SQL batch {idx} con {len(batch)} puntos..."
        )
        resultados.append(
            normalizar_columnas(
                pd.read_sql(query, conexion, params=[int(x) for x in batch])
            )
        )

    if not resultados:
        return pd.DataFrame()

    df = pd.concat(resultados, ignore_index=True)
    df["codigo_punto"] = normalizar_codigo_punto(df["codigo_punto"])
    df = df.dropna(subset=["codigo_punto"]).copy()
    df["codigo_punto"] = df["codigo_punto"].astype(int)

    for col in [
        "cupo_punto_saldo",
        "entradas_punto",
        "salidas_punto",
        "saldo_punto_sql",
        "compensado_favor_banco",
        "compensado_favor_punto",
    ]:
        if col in df.columns:
            df[col] = normalizar_monto(df[col])

    df["fecha_ultima_compensacion_dt"] = parse_fecha_flexible(
        df["fecha_ultima_compensacion_sql"]
    )
    df["fecha_snapshot"] = pd.to_datetime(FECHA_FIN)
    df["saldo_expuesto_sql"] = df["saldo_punto_sql"].clip(lower=0)
    df["consumo_cupo_sql"] = np.where(
        df["cupo_punto_saldo"] > 0,
        df["saldo_punto_sql"] / df["cupo_punto_saldo"],
        0,
    )
    return df.drop_duplicates("codigo_punto")


def guardar_snapshot_saldos(df_saldos: pd.DataFrame) -> Path:
    fecha = pd.to_datetime(FECHA_FIN).strftime("%Y%m%d")
    ruta = SNAPSHOT_SALDOS_DIR / f"saldos_compensacion_snapshot_{fecha}.parquet"
    guardar_parquet_seguro(df_saldos, ruta)
    return ruta


def cargar_snapshots_saldos() -> pd.DataFrame:
    archivos = sorted(
        SNAPSHOT_SALDOS_DIR.glob("saldos_compensacion_snapshot_*.parquet")
    )
    frames = []
    for a in archivos:
        try:
            tmp = pd.read_parquet(a)
            tmp = normalizar_columnas(tmp)
            if "fecha_snapshot" not in tmp.columns:
                m = re.search(r"(\d{8})", a.name)
                tmp["fecha_snapshot"] = (
                    pd.to_datetime(m.group(1), format="%Y%m%d", errors="coerce")
                    if m
                    else pd.NaT
                )
            frames.append(tmp)
        except Exception as e:
            print(f"No se pudo leer snapshot {a}: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["codigo_punto"] = normalizar_codigo_punto(df["codigo_punto"])
    df = df.dropna(subset=["codigo_punto", "fecha_snapshot"]).copy()
    df["codigo_punto"] = df["codigo_punto"].astype(int)
    df["fecha_snapshot"] = pd.to_datetime(df["fecha_snapshot"], errors="coerce")
    if "fecha_ultima_compensacion_dt" in df.columns:
        df["fecha_ultima_compensacion_dt"] = parse_fecha_flexible(
            df["fecha_ultima_compensacion_dt"]
        )
    elif "fecha_ultima_compensacion_sql" in df.columns:
        df["fecha_ultima_compensacion_dt"] = parse_fecha_flexible(
            df["fecha_ultima_compensacion_sql"]
        )
    for col in [
        "saldo_punto_sql",
        "cupo_punto_saldo",
        "saldo_expuesto_sql",
        "consumo_cupo_sql",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df.sort_values(["codigo_punto", "fecha_snapshot"]).drop_duplicates(
        ["codigo_punto", "fecha_snapshot"], keep="last"
    )


def enriquecer_panel_con_compensacion_historica(
    panel: pd.DataFrame, snapshots: pd.DataFrame
) -> pd.DataFrame:
    """
    Une cada corte mensual con el último snapshot de saldo/compensación conocido ANTES o EN la fecha de corte.
    Esto evita usar información futura. Si solo hay snapshot actual, no se replica hacia meses anteriores.
    """
    panel = panel.copy()
    if snapshots is None or snapshots.empty:
        panel["fecha_snapshot_saldo_usada"] = pd.NaT
        panel["fecha_ultima_compensacion_hist"] = pd.NaT
        panel["dias_sin_compensar_hist"] = np.nan
        panel["saldo_punto_hist"] = np.nan
        panel["cupo_punto_saldo_hist"] = np.nan
        panel["saldo_expuesto_hist"] = np.nan
        panel["consumo_cupo_hist"] = np.nan
        panel["flag_sin_compensar_30d_hist"] = 0
        panel["flag_sin_compensar_60d_hist"] = 0
        panel["flag_saldo_expuesto_hist"] = 0
        panel["flag_consumo_alto_hist"] = 0
        panel["flag_comp_historica_disponible"] = 0
        return panel

    keep = [
        "codigo_punto",
        "fecha_snapshot",
        "fecha_ultima_compensacion_dt",
        "saldo_punto_sql",
        "cupo_punto_saldo",
    ]
    keep = [c for c in keep if c in snapshots.columns]
    snaps = snapshots[keep].copy()
    snaps = snaps.dropna(subset=["codigo_punto", "fecha_snapshot"]).sort_values(
        ["codigo_punto", "fecha_snapshot"]
    )

    left = panel.sort_values(["codigo_punto", "fecha_corte"]).copy()
    enriched_parts = []
    # merge_asof con by a veces falla si no está perfectamente ordenado; por seguridad se hace por grupo.
    for codigo, p_grp in left.groupby("codigo_punto", sort=False):
        s_grp = snaps[snaps["codigo_punto"] == codigo].sort_values("fecha_snapshot")
        if s_grp.empty:
            tmp = p_grp.copy()
            tmp["fecha_snapshot_saldo_usada"] = pd.NaT
            tmp["fecha_ultima_compensacion_hist"] = pd.NaT
            tmp["saldo_punto_hist"] = np.nan
            tmp["cupo_punto_saldo_hist"] = np.nan
        else:
            tmp = pd.merge_asof(
                p_grp.sort_values("fecha_corte"),
                s_grp.sort_values("fecha_snapshot").drop(columns=["codigo_punto"]),
                left_on="fecha_corte",
                right_on="fecha_snapshot",
                direction="backward",
            )
            tmp = tmp.rename(
                columns={
                    "fecha_snapshot": "fecha_snapshot_saldo_usada",
                    "fecha_ultima_compensacion_dt": "fecha_ultima_compensacion_hist",
                    "saldo_punto_sql": "saldo_punto_hist",
                    "cupo_punto_saldo": "cupo_punto_saldo_hist",
                }
            )
        enriched_parts.append(tmp)

    out = pd.concat(enriched_parts, ignore_index=True)
    out["dias_sin_compensar_hist"] = (
        out["fecha_corte"]
        - pd.to_datetime(out["fecha_ultima_compensacion_hist"], errors="coerce")
    ).dt.days
    out["dias_sin_compensar_hist"] = out["dias_sin_compensar_hist"].clip(lower=0)
    out["saldo_punto_hist"] = pd.to_numeric(out["saldo_punto_hist"], errors="coerce")
    out["cupo_punto_saldo_hist"] = pd.to_numeric(
        out["cupo_punto_saldo_hist"], errors="coerce"
    )
    out["saldo_expuesto_hist"] = out["saldo_punto_hist"].clip(lower=0)
    out["consumo_cupo_hist"] = np.where(
        out["cupo_punto_saldo_hist"].fillna(0) > 0,
        out["saldo_punto_hist"].fillna(0) / out["cupo_punto_saldo_hist"].fillna(0),
        np.nan,
    )
    out["flag_comp_historica_disponible"] = (
        out["fecha_snapshot_saldo_usada"].notna().astype(int)
    )
    out["flag_sin_compensar_30d_hist"] = (
        out["dias_sin_compensar_hist"].ge(30).fillna(False).astype(int)
    )
    out["flag_sin_compensar_60d_hist"] = (
        out["dias_sin_compensar_hist"].ge(60).fillna(False).astype(int)
    )
    out["flag_saldo_expuesto_hist"] = (
        out["saldo_expuesto_hist"].gt(0).fillna(False).astype(int)
    )
    out["flag_consumo_alto_hist"] = (
        out["consumo_cupo_hist"].ge(0.70).fillna(False).astype(int)
    )
    return out.sort_values(["codigo_punto", "fecha_corte"]).reset_index(drop=True)


# ============================================================
# ARCHIVO DE CASTIGADOS
# ============================================================


def primera_columna_existente(df: pd.DataFrame, candidatos: list[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidatos:
        c_norm = normalizar_texto_columna(c)
        if c_norm in cols:
            return c_norm
    return None


def cargar_cbs_castigados(ruta: str | Path) -> pd.DataFrame:
    """
    Interpreta correctamente:
    - CODIGO CB -> codigo_punto
    - Valor Castigado -> valor_castigado_original
    - Saldo Actual o saldo_punto -> saldo_castigado_pendiente
    """
    columnas_vacias = [
        "codigo_punto",
        "fecha_castigo",
        "valor_castigado_original",
        "saldo_castigado_pendiente",
        "monto_recuperado_castigo",
        "pct_recuperado_castigo",
        "motivo_cierre_castigado",
        "rango_mora_castigados",
        "flag_en_archivo_castigados",
    ]
    if not existe_archivo(ruta):
        print(f"No se encontró archivo de castigados: {ruta}")
        return pd.DataFrame(columns=columnas_vacias)

    df = pd.read_excel(ruta)
    df = normalizar_columnas(df)

    col_codigo = primera_columna_existente(
        df, ["CODIGO CB", "codigo_cb", "codigo_punto", "SALCOD"]
    )
    if col_codigo is None:
        raise ValueError("El archivo de castigados no tiene CODIGO CB/codigo_punto.")

    col_fecha = primera_columna_existente(
        df, ["FECHA CASTIGO", "fecha_castigo", "fecha"]
    )
    col_valor = primera_columna_existente(df, ["Valor Castigado", "valor_castigado"])
    col_saldo_actual = primera_columna_existente(
        df, ["Saldo Actual", "saldo_actual", "saldo_punto"]
    )
    col_motivo = primera_columna_existente(
        df, ["MOTIVO DE CIERRE", "motivo_de_cierre", "motivo_cierre"]
    )
    col_rango = primera_columna_existente(
        df, ["rango_mora", "rango_mor", "bloque_mora"]
    )

    out = pd.DataFrame()
    out["codigo_punto"] = normalizar_codigo_punto(df[col_codigo])
    out["fecha_castigo"] = parse_fecha_flexible(df[col_fecha]) if col_fecha else pd.NaT
    out["valor_castigado_original"] = (
        normalizar_monto(df[col_valor]) if col_valor else 0
    )
    out["saldo_castigado_pendiente"] = (
        normalizar_monto(df[col_saldo_actual]) if col_saldo_actual else 0
    )
    out["motivo_cierre_castigado"] = (
        df[col_motivo].fillna("Sin motivo informado").astype(str).str.strip()
        if col_motivo
        else "Sin motivo informado"
    )
    out["rango_mora_castigados"] = (
        df[col_rango].fillna("Sin rango").astype(str).str.strip()
        if col_rango
        else "Sin rango"
    )

    out = out.dropna(subset=["codigo_punto"]).copy()
    out["codigo_punto"] = out["codigo_punto"].astype(int)

    agg = {
        "fecha_castigo": "min",
        "valor_castigado_original": "sum",
        "saldo_castigado_pendiente": "sum",
        "motivo_cierre_castigado": lambda x: " | ".join(
            sorted(set(v for v in x.dropna().astype(str) if v.strip()))
        )[:500],
        "rango_mora_castigados": lambda x: " | ".join(
            sorted(set(v for v in x.dropna().astype(str) if v.strip()))
        )[:300],
    }
    out = out.groupby("codigo_punto", as_index=False).agg(agg)
    out["monto_recuperado_castigo"] = (
        out["valor_castigado_original"] - out["saldo_castigado_pendiente"]
    ).clip(lower=0)
    out["pct_recuperado_castigo"] = np.where(
        out["valor_castigado_original"] > 0,
        out["monto_recuperado_castigo"] / out["valor_castigado_original"],
        0,
    )
    out["flag_en_archivo_castigados"] = 1
    return out


# ============================================================
# PANEL HISTÓRICO Y TARGET
# ============================================================


def construir_panel_historico_mensual(
    df_base_actual: pd.DataFrame,
    df_base_hist: pd.DataFrame,
    df_trx_mensual: pd.DataFrame,
    df_castigados: pd.DataFrame,
    snapshots_saldos: pd.DataFrame,
    fecha_inicio: str,
    fecha_fin: str,
) -> pd.DataFrame:
    fecha_inicio_dt = pd.to_datetime(fecha_inicio)
    fecha_fin_dt = pd.to_datetime(fecha_fin)
    periodos = pd.period_range(fecha_inicio_dt, fecha_fin_dt, freq="M")
    df_periodos = pd.DataFrame({"periodo": periodos})
    df_periodos["fecha_corte"] = (
        df_periodos["periodo"].dt.to_timestamp("M").clip(upper=fecha_fin_dt)
    )

    min_cb_hist = min(
        1000, max(1, int(df_base_actual["codigo_punto"].nunique() * 0.10))
    )
    base_hist_valida = (
        df_base_hist is not None
        and not df_base_hist.empty
        and "codigo_punto" in df_base_hist.columns
        and df_base_hist["codigo_punto"].nunique() >= min_cb_hist
    )

    if base_hist_valida:
        base_cols = [
            "codigo_punto",
            "periodo",
            "fecha_corte",
            "codigo_corresponsal",
            "codigo_dane",
            "region",
            "departamento",
            "municipio",
            "corregimiento",
            "nombre_punto",
            "formato_punto",
            "categoria",
            "tipo_persona",
            "regimen",
            "numero_datafonos",
            "red_cb",
            "cupo_actual",
            "fecha_implementacion",
            "fecha_cierre",
            "estado_al_corte",
            "fuerza_comercial",
            "canal_comunicacion",
        ]
        base_cols = [c for c in base_cols if c in df_base_hist.columns]
        panel = df_base_hist[base_cols].copy()
        panel["fuente_base_historica"] = "base_funcionamiento_historica"
        # Alinear fecha_corte al fin de mes para cruzar con TRX mensual.
        panel["fecha_corte_original_base"] = panel["fecha_corte"]
        panel["fecha_corte"] = (
            panel["periodo"].dt.to_timestamp("M").clip(upper=fecha_fin_dt)
        )
    else:
        # Fallback: crea panel con la base actual, pero NO usa estado_actual como predictor.
        base_cols = [
            "codigo_punto",
            "codigo_corresponsal",
            "codigo_dane",
            "region",
            "departamento",
            "municipio",
            "corregimiento",
            "nombre_punto",
            "formato_punto",
            "categoria",
            "tipo_persona",
            "regimen",
            "numero_datafonos",
            "red_cb",
            "cupo_actual",
            "fecha_implementacion",
            "fecha_cierre",
            "estado_actual",
            "fuerza_comercial",
            "canal_comunicacion",
        ]
        base_cols = [c for c in base_cols if c in df_base_actual.columns]
        base = df_base_actual[base_cols].drop_duplicates("codigo_punto").copy()
        base["_k"] = 1
        df_periodos["_k"] = 1
        panel = base.merge(df_periodos, on="_k", how="outer").drop(columns="_k")
        panel["estado_al_corte"] = "NO_DISPONIBLE"
        panel["fuente_base_historica"] = "fallback_base_actual_no_usar_estado"

    trx_cols = [
        "codigo_punto",
        "periodo",
        "total_trx",
        "cnt_aprobada",
        "cnt_rechazada",
        "cnt_reversada",
        "valor_total",
        "ticket_promedio_mes",
        "fecha_primera_trx_mes",
        "fecha_ultima_trx_mes",
        "tasa_aprobada",
        "tasa_rechazada",
        "tasa_reversada",
    ]
    trx_cols = [c for c in trx_cols if c in df_trx_mensual.columns]
    panel = panel.merge(
        df_trx_mensual[trx_cols], on=["codigo_punto", "periodo"], how="left"
    )

    for col in [
        "total_trx",
        "cnt_aprobada",
        "cnt_rechazada",
        "cnt_reversada",
        "valor_total",
        "ticket_promedio_mes",
        "tasa_aprobada",
        "tasa_rechazada",
        "tasa_reversada",
    ]:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0)

    panel = panel.sort_values(["codigo_punto", "periodo"]).reset_index(drop=True)

    panel["fecha_ultima_trx_acum"] = panel.groupby("codigo_punto")[
        "fecha_ultima_trx_mes"
    ].ffill()
    panel["dias_sin_transacciones_limpio"] = (
        panel["fecha_corte"] - panel["fecha_ultima_trx_acum"]
    ).dt.days
    panel["dias_sin_transacciones"] = panel["dias_sin_transacciones_limpio"].fillna(999)
    panel["dias_operacion"] = (
        panel["fecha_corte"] - panel["fecha_implementacion"]
    ).dt.days

    panel["activo_en_corte"] = (
        panel["fecha_implementacion"].notna()
        & (panel["fecha_implementacion"] <= panel["fecha_corte"])
        & (
            panel["fecha_cierre"].isna()
            | (panel["fecha_cierre"] > panel["fecha_corte"])
        )
    ).astype(int)

    grp = panel.groupby("codigo_punto", group_keys=False)
    panel["total_trx_mes_anterior"] = grp["total_trx"].shift(1)
    panel["valor_mes_anterior"] = grp["valor_total"].shift(1)
    panel["trx_3m"] = (
        grp["total_trx"].rolling(3, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    panel["trx_6m"] = (
        grp["total_trx"].rolling(6, min_periods=1).sum().reset_index(level=0, drop=True)
    )
    panel["trx_prom_3m"] = (
        grp["total_trx"]
        .rolling(3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    panel["trx_prom_6m"] = (
        grp["total_trx"]
        .rolling(6, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    panel["valor_3m"] = (
        grp["valor_total"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    panel["valor_6m"] = (
        grp["valor_total"]
        .rolling(6, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    panel["var_trx_mom"] = np.where(
        panel["total_trx_mes_anterior"].fillna(0) > 0,
        (panel["total_trx"] - panel["total_trx_mes_anterior"])
        / panel["total_trx_mes_anterior"],
        np.nan,
    )

    panel["flag_inactivo_30d"] = panel["dias_sin_transacciones"].ge(30).astype(int)
    panel["flag_inactivo_60d"] = panel["dias_sin_transacciones"].ge(60).astype(int)
    panel["flag_inactivo_90d"] = panel["dias_sin_transacciones"].ge(90).astype(int)
    panel["flag_sin_trx_mes"] = panel["total_trx"].eq(0).astype(int)
    panel["flag_caida_trx_50"] = (
        panel["var_trx_mom"].le(-0.50).fillna(False).astype(int)
    )

    # Enriquecer con compensación histórica segura por snapshots.
    panel = enriquecer_panel_con_compensacion_historica(panel, snapshots_saldos)

    # Etiquetas de castigados y cierre con saldo.
    if df_castigados is not None and not df_castigados.empty:
        panel = panel.merge(df_castigados, on="codigo_punto", how="left")
    else:
        panel["fecha_castigo"] = pd.NaT
        panel["valor_castigado_original"] = 0
        panel["saldo_castigado_pendiente"] = 0
        panel["motivo_cierre_castigado"] = np.nan
        panel["flag_en_archivo_castigados"] = 0

    estado_actual_norm = (
        panel.get("estado_actual", pd.Series("", index=panel.index))
        .astype(str)
        .str.upper()
        .str.strip()
    )
    estado_corte_norm = (
        panel.get("estado_al_corte", pd.Series("", index=panel.index))
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # Para etiqueta sí podemos usar la fecha real conocida de cierre con saldo; no se usa como predictor.
    fecha_cierre_riesgo = pd.to_datetime(panel["fecha_cierre"], errors="coerce").where(
        estado_actual_norm.eq("CERRADO CON SALDO")
        | estado_corte_norm.eq("CERRADO CON SALDO")
    )
    fecha_castigo_riesgo = pd.to_datetime(panel["fecha_castigo"], errors="coerce")

    panel["fecha_evento_riesgo"] = pd.concat(
        [fecha_cierre_riesgo, fecha_castigo_riesgo], axis=1
    ).min(axis=1, skipna=True)
    panel["tipo_evento_riesgo"] = np.select(
        [
            fecha_cierre_riesgo.notna()
            & fecha_castigo_riesgo.notna()
            & (fecha_cierre_riesgo <= fecha_castigo_riesgo),
            fecha_cierre_riesgo.notna() & fecha_castigo_riesgo.isna(),
            fecha_castigo_riesgo.notna(),
        ],
        ["Cierre con saldo antes de castigo", "Cierre con saldo", "Castigo"],
        default="Sin evento",
    )
    panel["dias_hasta_evento_riesgo"] = (
        panel["fecha_evento_riesgo"] - panel["fecha_corte"]
    ).dt.days
    for h in [30, 60, 90]:
        panel[f"evento_riesgo_{h}d"] = (
            panel["dias_hasta_evento_riesgo"].gt(0)
            & panel["dias_hasta_evento_riesgo"].le(h)
        ).astype(int)
    panel["ya_evento_riesgo_al_corte"] = (
        panel["fecha_evento_riesgo"].notna()
        & (panel["fecha_evento_riesgo"] <= panel["fecha_corte"])
    ).astype(int)

    panel["score_riesgo_historico"] = 0
    panel["score_riesgo_historico"] += panel["flag_inactivo_30d"] * 20
    panel["score_riesgo_historico"] += panel["flag_inactivo_60d"] * 10
    panel["score_riesgo_historico"] += panel["flag_sin_trx_mes"] * 15
    panel["score_riesgo_historico"] += panel["flag_caida_trx_50"] * 15
    panel["score_riesgo_historico"] += panel["flag_sin_compensar_30d_hist"] * 25
    panel["score_riesgo_historico"] += panel["flag_sin_compensar_60d_hist"] * 15
    panel["score_riesgo_historico"] += panel["flag_saldo_expuesto_hist"] * 10
    panel["score_riesgo_historico"] += panel["flag_consumo_alto_hist"] * 10
    panel["score_riesgo_historico"] = panel["score_riesgo_historico"].clip(upper=100)

    panel["rango_inactividad"] = (
        pd.cut(
            pd.to_numeric(panel["dias_sin_transacciones"], errors="coerce"),
            bins=[-1, 5, 14, 30, 90, 180, 360, np.inf],
            labels=["0-5", "6-14", "15-30", "31-90", "91-180", "181-360", ">360"],
        )
        .astype("object")
        .fillna("Sin TRX histórica")
    )

    return panel


# ============================================================
# ALERTA ACTUAL
# ============================================================


def cargar_ultima_trx_compensacion_archivo(ruta: str | Path) -> pd.DataFrame:
    if not existe_archivo(ruta):
        print(f"No se encontró archivo opcional última TRX/compensación: {ruta}")
        return pd.DataFrame(columns=["codigo_punto"])
    try:
        xls = pd.ExcelFile(ruta)
    except Exception as e:
        print(f"No se pudo abrir archivo opcional {ruta}: {e}")
        return pd.DataFrame(columns=["codigo_punto"])

    salida = None

    def leer_sheet(nombre_sheet: str, nuevo_nombre: str) -> pd.DataFrame:
        if nombre_sheet not in xls.sheet_names:
            return pd.DataFrame(columns=["codigo_punto", nuevo_nombre])
        tmp = normalizar_columnas(pd.read_excel(ruta, sheet_name=nombre_sheet))
        if "codigo_punto" not in tmp.columns:
            return pd.DataFrame(columns=["codigo_punto", nuevo_nombre])
        posibles_fecha = [c for c in tmp.columns if "fecha" in c or "trx" in c]
        if not posibles_fecha:
            return pd.DataFrame(columns=["codigo_punto", nuevo_nombre])
        col_fecha = posibles_fecha[-1]
        tmp["codigo_punto"] = normalizar_codigo_punto(tmp["codigo_punto"])
        tmp = tmp.dropna(subset=["codigo_punto"]).copy()
        tmp["codigo_punto"] = tmp["codigo_punto"].astype(int)
        tmp[nuevo_nombre] = parse_fecha_flexible(tmp[col_fecha])
        return tmp[["codigo_punto", nuevo_nombre]].drop_duplicates("codigo_punto")

    for tmp in [
        leer_sheet("ult trx aprobada", "fecha_ult_trx_aprobada_archivo"),
        leer_sheet("ult trx", "fecha_ult_trx_general_archivo"),
        leer_sheet("ult compensacion", "fecha_ult_compensacion_archivo"),
    ]:
        salida = (
            tmp if salida is None else salida.merge(tmp, on="codigo_punto", how="outer")
        )
    return salida if salida is not None else pd.DataFrame(columns=["codigo_punto"])


def cargar_ultima_compensacion_simple(ruta: str | Path) -> pd.DataFrame:
    """
    Carga un archivo simple con una sola fecha final de compensación por punto.

    Formato esperado:
        codigo_punto | fecha
        10000        | 20251229

    Uso seguro:
    - Sí: alerta actual y gestión operativa.
    - No: reconstruir todo el histórico 2025, porque sería fuga de información.
    """
    if not existe_archivo(ruta):
        print(f"No se encontró archivo simple de última compensación: {ruta}")
        return pd.DataFrame(
            columns=["codigo_punto", "fecha_ultima_compensacion_final_archivo"]
        )

    df = pd.read_excel(ruta)
    df = normalizar_columnas(df)

    if "codigo_punto" not in df.columns:
        raise ValueError(
            "El archivo simple de compensación debe tener columna codigo_punto."
        )

    if "fecha" in df.columns:
        col_fecha = "fecha"
    else:
        posibles = [c for c in df.columns if "fecha" in c or "compensacion" in c]
        if not posibles:
            raise ValueError(
                "El archivo simple de compensación debe tener columna fecha."
            )
        col_fecha = posibles[0]

    df["codigo_punto"] = normalizar_codigo_punto(df["codigo_punto"])
    df["fecha_ultima_compensacion_final_archivo"] = parse_fecha_flexible(df[col_fecha])
    df = df.dropna(subset=["codigo_punto"]).copy()
    df["codigo_punto"] = df["codigo_punto"].astype(int)

    # Una sola fecha por punto: la máxima fecha conocida.
    df = df.sort_values(
        ["codigo_punto", "fecha_ultima_compensacion_final_archivo"]
    ).drop_duplicates("codigo_punto", keep="last")

    return df[["codigo_punto", "fecha_ultima_compensacion_final_archivo"]]


def construir_alerta_actual(
    df_base_actual: pd.DataFrame,
    df_trx_mensual: pd.DataFrame,
    df_saldos_actual: pd.DataFrame,
    df_castigados: pd.DataFrame,
    df_ultima_archivo: pd.DataFrame,
    df_ult_comp_simple: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    fecha_ref = pd.to_datetime(FECHA_FIN)
    ult_trx = (
        df_trx_mensual.sort_values(["codigo_punto", "fecha_ultima_trx_mes"])
        .groupby("codigo_punto", as_index=False)
        .agg(fecha_ultima_trx=("fecha_ultima_trx_mes", "max"))
    )
    df = df_base_actual.copy().merge(ult_trx, on="codigo_punto", how="left")
    df = df.merge(df_saldos_actual, on="codigo_punto", how="left")

    if df_ultima_archivo is not None and not df_ultima_archivo.empty:
        df = df.merge(df_ultima_archivo, on="codigo_punto", how="left")
        for col in ["fecha_ult_trx_aprobada_archivo", "fecha_ult_trx_general_archivo"]:
            if col in df.columns:
                df["fecha_ultima_trx"] = df["fecha_ultima_trx"].combine_first(df[col])
        if "fecha_ult_compensacion_archivo" in df.columns:
            df["fecha_ultima_compensacion_dt"] = df[
                "fecha_ultima_compensacion_dt"
            ].combine_first(df["fecha_ult_compensacion_archivo"])

    # Si existe archivo simple de última compensación, usarlo solo como respaldo actual.
    if df_ult_comp_simple is not None and not df_ult_comp_simple.empty:
        df = df.merge(df_ult_comp_simple, on="codigo_punto", how="left")
        if "fecha_ultima_compensacion_final_archivo" in df.columns:
            df["fecha_ultima_compensacion_dt"] = df[
                "fecha_ultima_compensacion_dt"
            ].combine_first(df["fecha_ultima_compensacion_final_archivo"])

    df["dias_sin_transacciones"] = (
        fecha_ref - pd.to_datetime(df["fecha_ultima_trx"], errors="coerce")
    ).dt.days.fillna(999)
    df["dias_sin_compensar"] = (
        fecha_ref - pd.to_datetime(df["fecha_ultima_compensacion_dt"], errors="coerce")
    ).dt.days
    df["dias_sin_compensar"] = df["dias_sin_compensar"].clip(lower=0).fillna(999)

    for col in ["saldo_punto_sql", "cupo_punto_saldo", "cupo_actual"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["cupo_modelo"] = np.where(
        df["cupo_punto_saldo"].fillna(0) > 0,
        df["cupo_punto_saldo"],
        df["cupo_actual"].fillna(0),
    )
    df["saldo_expuesto"] = df["saldo_punto_sql"].fillna(0).clip(lower=0)
    df["consumo_cupo_punto"] = np.where(
        df["cupo_modelo"] > 0, df["saldo_punto_sql"] / df["cupo_modelo"], 0
    )
    df["dias_mora_operativa"] = (df["dias_sin_transacciones"] - 6).clip(lower=0)
    df["bloque_mora_operativa"] = pd.cut(
        df["dias_mora_operativa"],
        bins=[-1, 0, 30, 60, np.inf],
        labels=["Óptimo", "1-30", "31-60", "Mayor a 60"],
    ).astype("object")

    df["flag_inactivo_30d"] = df["dias_sin_transacciones"].ge(30).astype(int)
    df["flag_inactivo_60d"] = df["dias_sin_transacciones"].ge(60).astype(int)
    df["flag_sin_compensar_30d"] = df["dias_sin_compensar"].ge(30).astype(int)
    df["flag_sin_compensar_60d"] = df["dias_sin_compensar"].ge(60).astype(int)
    df["flag_consumo_alto"] = df["consumo_cupo_punto"].ge(0.70).astype(int)
    df["flag_consumo_critico"] = df["consumo_cupo_punto"].ge(0.90).astype(int)
    df["flag_saldo_expuesto"] = df["saldo_expuesto"].gt(0).astype(int)
    df["flag_mora_31_60"] = df["bloque_mora_operativa"].eq("31-60").astype(int)
    df["flag_mora_mayor_60"] = df["bloque_mora_operativa"].eq("Mayor a 60").astype(int)

    df["score_riesgo_reglas"] = 0
    df["score_riesgo_reglas"] += df["flag_sin_compensar_30d"] * 30
    df["score_riesgo_reglas"] += df["flag_sin_compensar_60d"] * 15
    df["score_riesgo_reglas"] += df["flag_inactivo_30d"] * 20
    df["score_riesgo_reglas"] += df["flag_inactivo_60d"] * 10
    df["score_riesgo_reglas"] += df["flag_consumo_alto"] * 15
    df["score_riesgo_reglas"] += df["flag_consumo_critico"] * 10
    df["score_riesgo_reglas"] += df["flag_saldo_expuesto"] * 10
    df["score_riesgo_reglas"] += df["flag_mora_31_60"] * 25
    df["score_riesgo_reglas"] += df["flag_mora_mayor_60"] * 35
    df["score_riesgo_reglas"] = df["score_riesgo_reglas"].clip(upper=100)
    df["nivel_riesgo_reglas"] = pd.cut(
        df["score_riesgo_reglas"],
        bins=[-1, 39, 59, 79, 100],
        labels=["Bajo", "Medio", "Alto", "Crítico"],
    ).astype("object")
    df["perdida_esperada_reglas"] = (df["score_riesgo_reglas"] / 100) * df[
        "saldo_expuesto"
    ]

    if df_castigados is not None and not df_castigados.empty:
        df = df.merge(df_castigados, on="codigo_punto", how="left")
    else:
        df["flag_en_archivo_castigados"] = 0
        df["fecha_castigo"] = pd.NaT
        df["valor_castigado_original"] = 0
        df["saldo_castigado_pendiente"] = 0
        df["motivo_cierre_castigado"] = np.nan

    def crear_motivo(row):
        motivos = []
        if row["flag_sin_compensar_30d"]:
            motivos.append(f"{int(row['dias_sin_compensar'])} días sin compensar")
        if row["flag_inactivo_30d"]:
            motivos.append(
                f"{int(row['dias_sin_transacciones'])} días sin transacciones"
            )
        if row["flag_consumo_alto"]:
            motivos.append(f"consumo cupo {row['consumo_cupo_punto']:.0%}")
        if row["flag_saldo_expuesto"]:
            motivos.append(f"saldo expuesto ${row['saldo_expuesto']:,.0f}")
        if row["flag_mora_31_60"] or row["flag_mora_mayor_60"]:
            motivos.append(f"mora operativa {row['bloque_mora_operativa']}")
        if row.get("flag_en_archivo_castigados", 0) == 1:
            motivos.append(
                f"aparece en castigados: {row.get('motivo_cierre_castigado', '')}"
            )
        return " | ".join(motivos) if motivos else "Sin señal crítica"

    df["motivo_alerta"] = df.apply(crear_motivo, axis=1)
    df["accion_recomendada"] = np.select(
        [
            df["nivel_riesgo_reglas"].eq("Crítico"),
            df["nivel_riesgo_reglas"].eq("Alto"),
            df["nivel_riesgo_reglas"].eq("Medio"),
        ],
        [
            "Bloqueo/validación inmediata + gestión prioritaria",
            "Contacto el mismo día + seguimiento de compensación",
            "Monitoreo y recordatorio preventivo",
        ],
        default="Seguimiento normal",
    )
    orden = {"Crítico": 1, "Alto": 2, "Medio": 3, "Bajo": 4}
    df["orden_nivel"] = df["nivel_riesgo_reglas"].map(orden).fillna(9)
    df = df.sort_values(
        ["orden_nivel", "perdida_esperada_reglas", "saldo_expuesto"],
        ascending=[True, False, False],
    )
    return df.drop(columns=["orden_nivel"])


# ============================================================
# TARGET ROBUSTO SIN FUGA DE INFORMACIÓN
# ============================================================


def reconstruir_target_evento_riesgo(
    panel: pd.DataFrame,
    df_base_actual: pd.DataFrame,
    df_castigados: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruye el target por codigo_punto y lo propaga a todos los meses previos.

    Evento de riesgo = primera fecha conocida entre:
    1. primera aparición histórica como CERRADO CON SALDO,
    2. fecha_cierre de la base actual si el estado actual es CERRADO CON SALDO,
    3. fecha_castigo del archivo de castigados.

    Esta información se usa SOLO para construir la etiqueta, no como predictor.
    """
    panel = panel.copy()
    panel["fecha_corte"] = pd.to_datetime(panel["fecha_corte"], errors="coerce")
    if "fecha_cierre" in panel.columns:
        panel["fecha_cierre"] = pd.to_datetime(panel["fecha_cierre"], errors="coerce")

    # 1) Cierre con saldo observado en la historia mensual.
    estado_corte = (
        panel.get("estado_al_corte", pd.Series("", index=panel.index))
        .astype(str)
        .str.upper()
        .str.strip()
    )

    cierre_hist = panel.loc[
        estado_corte.eq("CERRADO CON SALDO"),
        ["codigo_punto", "fecha_cierre", "fecha_corte"],
    ].copy()

    if not cierre_hist.empty:
        cierre_hist["fecha_cierre_con_saldo_hist"] = cierre_hist["fecha_cierre"].where(
            cierre_hist["fecha_cierre"].notna(), cierre_hist["fecha_corte"]
        )
        cierre_hist = cierre_hist.groupby("codigo_punto", as_index=False).agg(
            fecha_cierre_con_saldo_hist=("fecha_cierre_con_saldo_hist", "min")
        )
    else:
        cierre_hist = pd.DataFrame(
            columns=["codigo_punto", "fecha_cierre_con_saldo_hist"]
        )

    # 2) Cierre con saldo desde la base actual. Solo etiqueta, no predictor.
    base = df_base_actual.copy()
    estado_actual = (
        base.get("estado_actual", pd.Series("", index=base.index))
        .astype(str)
        .str.upper()
        .str.strip()
    )
    cierre_actual = base.loc[
        estado_actual.eq("CERRADO CON SALDO"), ["codigo_punto", "fecha_cierre"]
    ].copy()

    if not cierre_actual.empty:
        cierre_actual["fecha_cierre"] = pd.to_datetime(
            cierre_actual["fecha_cierre"], errors="coerce"
        )
        cierre_actual = (
            cierre_actual.dropna(subset=["fecha_cierre"])
            .groupby("codigo_punto", as_index=False)
            .agg(fecha_cierre_con_saldo_actual=("fecha_cierre", "min"))
        )
    else:
        cierre_actual = pd.DataFrame(
            columns=["codigo_punto", "fecha_cierre_con_saldo_actual"]
        )

    # 3) Castigo.
    if df_castigados is not None and not df_castigados.empty:
        cast = df_castigados[["codigo_punto", "fecha_castigo"]].copy()
        cast["fecha_castigo"] = pd.to_datetime(cast["fecha_castigo"], errors="coerce")
        cast = (
            cast.dropna(subset=["fecha_castigo"])
            .groupby("codigo_punto", as_index=False)
            .agg(fecha_castigo_objetivo=("fecha_castigo", "min"))
        )
    else:
        cast = pd.DataFrame(columns=["codigo_punto", "fecha_castigo_objetivo"])

    eventos = panel[["codigo_punto"]].drop_duplicates()
    eventos = eventos.merge(cierre_hist, on="codigo_punto", how="left")
    eventos = eventos.merge(cierre_actual, on="codigo_punto", how="left")
    eventos = eventos.merge(cast, on="codigo_punto", how="left")

    eventos["fecha_evento_riesgo"] = pd.concat(
        [
            eventos["fecha_cierre_con_saldo_hist"],
            eventos["fecha_cierre_con_saldo_actual"],
            eventos["fecha_castigo_objetivo"],
        ],
        axis=1,
    ).min(axis=1, skipna=True)

    eventos["tipo_evento_riesgo"] = np.select(
        [
            eventos["fecha_cierre_con_saldo_hist"].notna()
            & (
                eventos["fecha_evento_riesgo"].eq(
                    eventos["fecha_cierre_con_saldo_hist"]
                )
            ),
            eventos["fecha_cierre_con_saldo_actual"].notna()
            & (
                eventos["fecha_evento_riesgo"].eq(
                    eventos["fecha_cierre_con_saldo_actual"]
                )
            ),
            eventos["fecha_castigo_objetivo"].notna()
            & (eventos["fecha_evento_riesgo"].eq(eventos["fecha_castigo_objetivo"])),
        ],
        ["Cierre con saldo histórico", "Cierre con saldo actual", "Castigo"],
        default="Sin evento",
    )

    panel = panel.drop(
        columns=[
            "fecha_evento_riesgo",
            "tipo_evento_riesgo",
            "dias_hasta_evento_riesgo",
            "evento_riesgo_30d",
            "evento_riesgo_60d",
            "evento_riesgo_90d",
            "ya_evento_riesgo_al_corte",
        ],
        errors="ignore",
    )

    panel = panel.merge(
        eventos[["codigo_punto", "fecha_evento_riesgo", "tipo_evento_riesgo"]],
        on="codigo_punto",
        how="left",
    )

    panel["dias_hasta_evento_riesgo"] = (
        panel["fecha_evento_riesgo"] - panel["fecha_corte"]
    ).dt.days

    for h in [30, 60, 90]:
        panel[f"evento_riesgo_{h}d"] = (
            panel["dias_hasta_evento_riesgo"].gt(0)
            & panel["dias_hasta_evento_riesgo"].le(h)
        ).astype(int)

    panel["ya_evento_riesgo_al_corte"] = (
        panel["fecha_evento_riesgo"].notna()
        & (panel["fecha_evento_riesgo"] <= panel["fecha_corte"])
    ).astype(int)

    return panel


# ============================================================
# LISTAS DE VARIABLES PARA MODELADO
# ============================================================


def generar_diccionario_variables(
    panel_entrenamiento: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Genera una lista controlada de variables candidatas para el Modelo V1.

    Principio:
    - Usar lista blanca para evitar fuga de información.
    - No usar identificadores, fechas crudas, estados de cierre, columnas de castigo
      ni variables que representen información posterior al evento.
    - No usar compensación/saldo histórico si la cobertura real de snapshots es baja.
    - No usar scores de reglas como predictor del primer modelo, para que el modelo
      aprenda desde señales base y el score quede como comparación operacional.
    """
    numericas_base = [
        "total_trx",
        "cnt_aprobada",
        "cnt_rechazada",
        "cnt_reversada",
        "valor_total",
        "ticket_promedio_mes",
        "tasa_aprobada",
        "tasa_rechazada",
        "tasa_reversada",
        "dias_sin_transacciones",
        "dias_operacion",
        "total_trx_mes_anterior",
        "valor_mes_anterior",
        "trx_3m",
        "trx_6m",
        "trx_prom_3m",
        "trx_prom_6m",
        "valor_3m",
        "valor_6m",
        "var_trx_mom",
        "flag_inactivo_30d",
        "flag_inactivo_60d",
        "flag_inactivo_90d",
        "flag_sin_trx_mes",
        "flag_caida_trx_50",
        "cupo_actual",
        "numero_datafonos",
    ]

    categoricas_base = [
        "region",
        "departamento",
        "municipio",
        "formato_punto",
        "tipo_persona",
        "regimen",
        "red_cb",
        "fuerza_comercial",
        "canal_comunicacion",
        "rango_inactividad",
    ]

    # Estas variables solo entran cuando hay snapshots históricos reales suficientes.
    # Si solo hay una foto actual, no deben usarse para entrenar historia 2025-2026.
    compensacion_hist = [
        "dias_sin_compensar_hist",
        "saldo_punto_hist",
        "cupo_punto_saldo_hist",
        "saldo_expuesto_hist",
        "consumo_cupo_hist",
        "flag_sin_compensar_30d_hist",
        "flag_sin_compensar_60d_hist",
        "flag_saldo_expuesto_hist",
        "flag_consumo_alto_hist",
    ]

    cobertura_comp = 0.0
    if (
        "flag_comp_historica_disponible" in panel_entrenamiento.columns
        and len(panel_entrenamiento) > 0
    ):
        cobertura_comp = panel_entrenamiento["flag_comp_historica_disponible"].mean()

    candidatas = [
        c for c in numericas_base + categoricas_base if c in panel_entrenamiento.columns
    ]

    if cobertura_comp >= 0.20:
        candidatas += [c for c in compensacion_hist if c in panel_entrenamiento.columns]
    else:
        print(
            f"Compensación histórica con cobertura {cobertura_comp:.2%}. "
            "No se usará como predictor del modelo histórico."
        )

    excluir_patrones = (
        "fecha_",
        "evento_",
        "dias_hasta_evento",
        "ya_evento",
        "castigado",
        "motivo",
        "nombre",
        "nit",
        "documento",
        "email",
        "celular",
        "direccion",
    )
    excluir_explicitas = {
        "codigo_punto",
        "codigo_corresponsal",
        "codigo_dane",
        "periodo",
        "estado_actual",
        "estado_al_corte",
        "fecha_cierre",
        "valor_castigado_original",
        "saldo_castigado_pendiente",
        "monto_recuperado_castigo",
        "pct_recuperado_castigo",
        "flag_en_archivo_castigados",
        "tipo_evento_riesgo",
        "fuente_base_historica",
        "fecha_corte_original_base",
        "fecha_snapshot_saldo_usada",
        "dias_sin_transacciones_limpio",
        "score_riesgo_historico",
        "categoria",
    }

    excluidas_por_fuga = [
        c
        for c in panel_entrenamiento.columns
        if c.startswith(excluir_patrones) or c in excluir_explicitas
    ]

    candidatas = [c for c in candidatas if c not in excluidas_por_fuga]

    # Quitar variables constantes, casi vacías o duplicadas dentro del dataset de entrenamiento.
    variables_finales = []
    variables_descartadas_calidad = []
    for c in sorted(set(candidatas)):
        s = panel_entrenamiento[c]
        nulos = float(s.isna().mean()) if len(s) else 1.0
        nunique = s.nunique(dropna=True)
        if nulos >= 0.95:
            variables_descartadas_calidad.append((c, f"descartar_nulos_{nulos:.2%}"))
            continue
        if nunique <= 1:
            variables_descartadas_calidad.append((c, "descartar_constante"))
            continue
        variables_finales.append(c)

    df_safe = pd.DataFrame(
        {
            "variable": variables_finales,
            "uso": "predictora_segura_modelo_v1",
        }
    )

    df_excl_fuga = pd.DataFrame(
        {
            "variable": sorted(set(excluidas_por_fuga)),
            "uso": "excluir_por_fuga_id_fecha_texto_o_info_posterior",
        }
    )
    df_excl_calidad = pd.DataFrame(
        variables_descartadas_calidad,
        columns=["variable", "uso"],
    )
    df_excl = pd.concat([df_excl_fuga, df_excl_calidad], ignore_index=True)

    return df_safe, df_excl


def construir_dataset_modelo_limpio(
    dataset_90d: pd.DataFrame,
    variables_seguras: pd.DataFrame,
    target: str = "evento_riesgo_90d",
) -> pd.DataFrame:
    """
    Construye el dataset final de entrenamiento limpio.

    Incluye columnas de trazabilidad, variables predictoras seguras y target.
    Las columnas de trazabilidad NO deben usarse como X del modelo; se dejan para auditoría.
    """
    if target not in dataset_90d.columns:
        raise ValueError(f"No existe el target requerido: {target}")

    variables = variables_seguras["variable"].dropna().astype(str).tolist()
    variables = [c for c in variables if c in dataset_90d.columns and c != target]

    trazabilidad = [
        c
        for c in ["codigo_punto", "fecha_corte", "periodo"]
        if c in dataset_90d.columns
    ]
    cols = trazabilidad + variables + [target]

    df = dataset_90d[cols].copy()
    df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    df = df[df[target].isin([0, 1])].copy()

    # Limpieza de infinitos y valores faltantes.
    for c in variables:
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
            df[c] = df[c].fillna(0)
        else:
            df[c] = (
                df[c]
                .astype("object")
                .where(df[c].notna(), "SIN_DATO")
                .astype(str)
                .str.strip()
                .replace({"": "SIN_DATO", "nan": "SIN_DATO", "None": "SIN_DATO"})
            )

    if "fecha_corte" in df.columns:
        df["fecha_corte"] = pd.to_datetime(df["fecha_corte"], errors="coerce")

    return df


def generar_resumen_calidad_modelo(
    dataset_modelo: pd.DataFrame,
    variables_seguras: pd.DataFrame,
    target: str = "evento_riesgo_90d",
) -> pd.DataFrame:
    """Genera métricas mínimas de calidad del dataset final de modelado."""
    eventos = (
        int(dataset_modelo[target].sum()) if target in dataset_modelo.columns else 0
    )
    filas = int(len(dataset_modelo))
    tasa = float(eventos / filas) if filas else 0.0
    cb_unicos = (
        int(dataset_modelo["codigo_punto"].nunique())
        if "codigo_punto" in dataset_modelo.columns
        else 0
    )
    meses = (
        int(dataset_modelo["periodo"].nunique())
        if "periodo" in dataset_modelo.columns
        else 0
    )
    n_variables = int(len(variables_seguras))

    estado = "OK_PARA_ENTRENAR"
    observacion = "Dataset listo para entrenar Modelo V1."
    if filas < 100000:
        estado = "REVISAR"
        observacion = "Pocas filas para un modelo robusto."
    if eventos < 500:
        estado = "NO_ENTRENAR"
        observacion = "Muy pocos eventos positivos; el modelo no aprendería bien."
    if tasa < 0.005 or tasa > 0.05:
        estado = "REVISAR" if estado == "OK_PARA_ENTRENAR" else estado
        observacion += (
            " Revisar tasa de evento; debería estar aproximadamente entre 0.5% y 5%."
        )

    return pd.DataFrame(
        [
            {"metrica": "filas", "valor": filas},
            {"metrica": "cb_unicos", "valor": cb_unicos},
            {"metrica": "periodos", "valor": meses},
            {"metrica": "eventos_positivos", "valor": eventos},
            {"metrica": "tasa_evento", "valor": tasa},
            {"metrica": "variables_predictoras", "valor": n_variables},
            {"metrica": "estado_calidad", "valor": estado},
            {"metrica": "observacion", "valor": observacion},
        ]
    )


# ============================================================
# MAIN
# ============================================================


def main():
    print("==============================================")
    print("INICIO PIPELINE ROBUSTO MODELO RIESGO CB v8 - DATASET LIMPIO")
    print(f"Fecha inicio histórico: {FECHA_INICIO}")
    print(f"Fecha fin histórico:    {FECHA_FIN}")
    print("==============================================")

    impala = None
    nacional = None
    try:
        impala = conectar_impala()
        df_base_actual = obtener_base_funcionamiento_actual(impala)
        print(f"Base funcionamiento actual: {df_base_actual.shape}")

        try:
            df_base_hist = obtener_base_funcionamiento_historica_mensual(
                impala, FECHA_INICIO, FECHA_FIN
            )
            print(f"Base funcionamiento histórica mensual: {df_base_hist.shape}")
        except Exception as e:
            print(
                f"No fue posible consultar base histórica mensual. Se usa fallback seguro: {e}"
            )
            df_base_hist = pd.DataFrame()

        lista_puntos = (
            df_base_actual["codigo_punto"].dropna().astype(int).unique().tolist()
        )

        df_trx_mensual = consultar_trx_mensual_por_batch(
            impala, lista_puntos, FECHA_INICIO, FECHA_FIN
        )
        print(f"TRX mensual: {df_trx_mensual.shape}")

        nacional = conectar_nacional()
        df_saldos_actual = consultar_saldos_actuales(nacional, lista_puntos)
        print(f"Saldos/compensación SQL actual: {df_saldos_actual.shape}")
        guardar_snapshot_saldos(df_saldos_actual)
        snapshots_saldos = cargar_snapshots_saldos()
        print(
            f"Snapshots históricos saldos/compensación disponibles: {snapshots_saldos.shape}"
        )

        df_castigados = cargar_cbs_castigados(RUTA_CBS_CASTIGADOS)
        print(f"CBs castigados limpio: {df_castigados.shape}")

        df_ultima_archivo = cargar_ultima_trx_compensacion_archivo(RUTA_ULTIMA_TRX_COMP)
        print(f"Archivo opcional última TRX/compensación: {df_ultima_archivo.shape}")

        df_ult_comp_simple = cargar_ultima_compensacion_simple(
            RUTA_ULT_COMPENSACION_SIMPLE
        )
        print(
            f"Archivo simple última compensación por punto: {df_ult_comp_simple.shape}"
        )

        panel_historico = construir_panel_historico_mensual(
            df_base_actual=df_base_actual,
            df_base_hist=df_base_hist,
            df_trx_mensual=df_trx_mensual,
            df_castigados=df_castigados,
            snapshots_saldos=snapshots_saldos,
            fecha_inicio=FECHA_INICIO,
            fecha_fin=FECHA_FIN,
        )

        # Reetiquetado robusto: propaga la primera fecha de cierre con saldo/castigo
        # a todos los meses anteriores del mismo CB.
        panel_historico = reconstruir_target_evento_riesgo(
            panel=panel_historico,
            df_base_actual=df_base_actual,
            df_castigados=df_castigados,
        )
        print(f"Panel histórico mensual: {panel_historico.shape}")

        panel_entrenamiento = panel_historico[
            (panel_historico["activo_en_corte"] == 1)
            & (panel_historico["ya_evento_riesgo_al_corte"] == 0)
        ].copy()
        fecha_limite_90d = pd.to_datetime(FECHA_FIN) - pd.Timedelta(days=90)
        dataset_90d = panel_entrenamiento[
            panel_entrenamiento["fecha_corte"] <= fecha_limite_90d
        ].copy()
        print(f"Dataset entrenamiento seguro 90d: {dataset_90d.shape}")

        alertas_actuales = construir_alerta_actual(
            df_base_actual=df_base_actual,
            df_trx_mensual=df_trx_mensual,
            df_saldos_actual=df_saldos_actual,
            df_castigados=df_castigados,
            df_ultima_archivo=df_ultima_archivo,
            df_ult_comp_simple=df_ult_comp_simple,
        )
        print(f"Alertas actuales: {alertas_actuales.shape}")

        # Comparativo directo base actual vs castigados.
        comparativo = df_base_actual.merge(df_castigados, on="codigo_punto", how="left")
        comparativo["flag_en_archivo_castigados"] = (
            comparativo["flag_en_archivo_castigados"].fillna(0).astype(int)
        )

        safe, excl = generar_diccionario_variables(dataset_90d)

        # Dataset final limpio para modelado.
        # Usar las columnas de variables_predictoras_seguras.csv como X
        # y evento_riesgo_90d como y. codigo_punto/fecha_corte/periodo son trazabilidad.
        dataset_modelo_limpio = construir_dataset_modelo_limpio(
            dataset_90d=dataset_90d,
            variables_seguras=safe,
            target="evento_riesgo_90d",
        )
        resumen_modelo = generar_resumen_calidad_modelo(
            dataset_modelo=dataset_modelo_limpio,
            variables_seguras=safe,
            target="evento_riesgo_90d",
        )
        print(f"Dataset modelo limpio: {dataset_modelo_limpio.shape}")
        print(resumen_modelo.to_string(index=False))

        # Exportar
        guardar_parquet_seguro(
            df_base_actual, SALIDA_DIR / "base_funcionamiento_actual.parquet"
        )
        guardar_parquet_seguro(
            df_base_hist, SALIDA_DIR / "base_funcionamiento_historica_mensual.parquet"
        )
        guardar_parquet_seguro(df_trx_mensual, SALIDA_DIR / "trx_mensual_cb.parquet")
        guardar_parquet_seguro(
            df_saldos_actual, SALIDA_DIR / "saldos_compensacion_actual.parquet"
        )
        guardar_parquet_seguro(
            snapshots_saldos,
            SALIDA_DIR / "saldos_compensacion_snapshots_historicos.parquet",
        )
        guardar_parquet_seguro(
            df_castigados, SALIDA_DIR / "cbs_castigados_limpio.parquet"
        )
        guardar_parquet_seguro(
            comparativo, SALIDA_DIR / "comparativo_base_vs_castigados.parquet"
        )
        guardar_parquet_seguro(
            panel_historico, SALIDA_DIR / "panel_historico_mensual_cb.parquet"
        )
        guardar_parquet_seguro(
            panel_entrenamiento,
            SALIDA_DIR / "dataset_entrenamiento_cb_evento_riesgo.parquet",
        )
        guardar_parquet_seguro(
            dataset_90d,
            SALIDA_DIR / "dataset_entrenamiento_cb_evento_riesgo_90d_seguro.parquet",
        )
        guardar_parquet_seguro(
            dataset_modelo_limpio,
            SALIDA_DIR / "dataset_modelo_cb_v1_limpio.parquet",
        )
        guardar_parquet_seguro(
            alertas_actuales, SALIDA_DIR / "alertas_actuales_riesgo_cb.parquet"
        )

        safe.to_csv(
            SALIDA_DIR / "variables_predictoras_seguras.csv",
            index=False,
            encoding="utf-8-sig",
        )
        excl.to_csv(
            SALIDA_DIR / "variables_excluidas_por_fuga.csv",
            index=False,
            encoding="utf-8-sig",
        )
        resumen_modelo.to_csv(
            SALIDA_DIR / "resumen_calidad_modelo_limpio.csv",
            index=False,
            encoding="utf-8-sig",
        )

        resumen = pd.DataFrame(
            [
                {
                    "tabla": "base_actual",
                    "filas": len(df_base_actual),
                    "cb_unicos": df_base_actual["codigo_punto"].nunique(),
                },
                {
                    "tabla": "trx_mensual",
                    "filas": len(df_trx_mensual),
                    "cb_unicos": (
                        df_trx_mensual["codigo_punto"].nunique()
                        if not df_trx_mensual.empty
                        else 0
                    ),
                },
                {
                    "tabla": "snapshots_saldos",
                    "filas": len(snapshots_saldos),
                    "cb_unicos": (
                        snapshots_saldos["codigo_punto"].nunique()
                        if not snapshots_saldos.empty
                        else 0
                    ),
                },
                {
                    "tabla": "panel_historico",
                    "filas": len(panel_historico),
                    "cb_unicos": panel_historico["codigo_punto"].nunique(),
                },
                {
                    "tabla": "dataset_90d",
                    "filas": len(dataset_90d),
                    "cb_unicos": dataset_90d["codigo_punto"].nunique(),
                    "eventos_90d": (
                        int(dataset_90d["evento_riesgo_90d"].sum())
                        if "evento_riesgo_90d" in dataset_90d
                        else 0
                    ),
                },
                {
                    "tabla": "dataset_modelo_limpio",
                    "filas": len(dataset_modelo_limpio),
                    "cb_unicos": (
                        dataset_modelo_limpio["codigo_punto"].nunique()
                        if "codigo_punto" in dataset_modelo_limpio
                        else 0
                    ),
                    "eventos_90d": (
                        int(dataset_modelo_limpio["evento_riesgo_90d"].sum())
                        if "evento_riesgo_90d" in dataset_modelo_limpio
                        else 0
                    ),
                },
                {
                    "tabla": "alertas_actuales",
                    "filas": len(alertas_actuales),
                    "cb_unicos": alertas_actuales["codigo_punto"].nunique(),
                },
            ]
        )
        resumen.to_csv(
            SALIDA_DIR / "resumen_calidad_dataset.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if EXPORTAR_EXCEL_ALERTAS:
            cols_alerta = [
                "codigo_punto",
                "nombre_punto",
                "estado_actual",
                "red_cb",
                "departamento",
                "municipio",
                "cupo_modelo",
                "saldo_punto_sql",
                "saldo_expuesto",
                "consumo_cupo_punto",
                "fecha_ultima_trx",
                "fecha_ultima_compensacion_dt",
                "dias_sin_transacciones",
                "dias_sin_compensar",
                "bloque_mora_operativa",
                "fecha_castigo",
                "motivo_cierre_castigado",
                "valor_castigado_original",
                "saldo_castigado_pendiente",
                "monto_recuperado_castigo",
                "pct_recuperado_castigo",
                "score_riesgo_reglas",
                "nivel_riesgo_reglas",
                "perdida_esperada_reglas",
                "motivo_alerta",
                "accion_recomendada",
            ]
            cols_alerta = [c for c in cols_alerta if c in alertas_actuales.columns]
            alertas_actuales[cols_alerta].to_excel(
                SALIDA_DIR / "alertas_actuales_riesgo_cb.xlsx", index=False
            )
            comparativo.to_excel(
                SALIDA_DIR / "comparativo_base_vs_castigados.xlsx", index=False
            )

        print("==============================================")
        print("FIN PIPELINE ROBUSTO MODELO RIESGO CB v8 - DATASET LIMPIO")
        print("==============================================")

    finally:
        if impala is not None:
            impala.close()
        if nacional is not None:
            nacional.close()


if __name__ == "__main__":
    main()
