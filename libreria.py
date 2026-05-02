import pandas as pd

base = "_modelo_riesgo_cb/salidas_produccion"

control = pd.read_csv(f"{base}/control_calidad_scoring.csv")
print("CONTROL CALIDAD")
print(control)

preventivo = pd.read_parquet(f"{base}/scoring_preventivo_activos.parquet")
recuperacion = pd.read_parquet(f"{base}/scoring_recuperacion_cerrados.parquet")

print("\nPREVENTIVO:", preventivo.shape)
print(preventivo["estado_actual"].value_counts(dropna=False))

print("\nRECUPERACION:", recuperacion.shape)
print(recuperacion["estado_actual"].value_counts(dropna=False))

print("\nNulos dias_sin_transacciones preventivo:")
print(preventivo["dias_sin_transacciones"].isna().sum())

print("\nConsumo visual negativo preventivo:")
print((preventivo["consumo_cupo_visual"] < 0).sum())

print("\nNuevos Top 500 preventivo:")
print(preventivo["flag_nuevo_top500_preventivo"].sum())
