#!/usr/bin/env python
# coding: utf-8

# ## Preproceso para la prediccion mensual

import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from pyspark.sql import Window
from pyspark.sql import functions as F


def _get_spark_session():
    try:
        return spark  # type: ignore[name-defined]
    except Exception:
        try:
            from pyspark.sql import SparkSession

            return SparkSession.getActiveSession()
        except Exception:
            return None


def _normalize_local_path(path_value):
    if path_value.startswith("dbfs:/"):
        return "/dbfs/" + path_value[len("dbfs:/") :].lstrip("/")
    return path_value


def _to_spark_path(path_value):
    path_str = str(path_value)
    if path_str.startswith("dbfs:/"):
        return path_str
    if path_str.startswith("/dbfs/"):
        return "dbfs:/" + path_str[len("/dbfs/") :]
    return "file://" + path_str


def _resolve_repo_dir():
    try:
        return Path(__file__).resolve().parent
    except Exception:
        try:
            nb_path = (  # type: ignore[name-defined]
                dbutils.notebook.entry_point.getDbutils()
                .notebook()
                .getContext()
                .notebookPath()
                .get()
            )
            return Path(f"/Workspace{nb_path}").resolve().parent
        except Exception:
            return Path.cwd()


def _get_periodo_prediccion_param():
    raw_value = ""
    try:
        dbutils.widgets.text("periodo_prediccion", "")  # type: ignore[name-defined]
        raw_value = dbutils.widgets.get("periodo_prediccion").strip()  # type: ignore[name-defined]
    except Exception:
        raw_value = os.getenv("PERIODO_PREDICCION", "").strip()

    if not raw_value:
        return None
    if not (raw_value.isdigit() and len(raw_value) == 6):
        raise ValueError("periodo_prediccion debe tener formato YYYYMM (ejemplo: 202501).")
    return int(raw_value)


def _read_sql_table(spark_session, table_name):
    print(f"Leyendo {table_name} desde Spark/Databricks...")
    return spark_session.table(table_name)


def _require_columns(df, required_columns, table_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError(f"La tabla {table_name} no contiene columnas requeridas: {missing}")


def _find_column_case_insensitive(columns, *candidates):
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _to_lowercase_columns(df):
    return df.select(*[F.col(c).alias(c.lower()) for c in df.columns])


def _drop_existing_columns(df, columns_to_drop):
    to_drop = [c for c in columns_to_drop if c in df.columns]
    if not to_drop:
        return df
    return df.drop(*to_drop)


def _write_single_csv(spark_df, target_path, compression=None):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = target_path.parent / f".tmp_{target_path.stem}_{uuid.uuid4().hex}"

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    writer = spark_df.coalesce(1).write.mode("overwrite").option("header", "true")
    if compression:
        writer = writer.option("compression", compression)
    writer.csv(_to_spark_path(tmp_dir))

    part_files = list(tmp_dir.glob("part-*"))
    if not part_files:
        raise RuntimeError(f"No se genero archivo csv temporal en {tmp_dir}")
    part_file = part_files[0]

    if target_path.exists():
        target_path.unlink()
    shutil.move(str(part_file), str(target_path))
    shutil.rmtree(tmp_dir, ignore_errors=True)


def _resolve_ltv_file(inputs_dir, periodo):
    override = os.getenv("GES_LTV_INPUT_FILE", "").strip()
    if override:
        override_path = Path(_normalize_local_path(override))
        if not override_path.is_absolute():
            override_path = inputs_dir / override_path
        if not override_path.exists():
            raise RuntimeError(
                f"GES_LTV_INPUT_FILE apunta a un archivo inexistente: {override_path}"
            )
        return override_path

    search_dirs = [inputs_dir, inputs_dir.parent]
    matched_period = []
    matched_any = []

    for base_dir in search_dirs:
        if not base_dir.exists():
            continue
        for root, _, files in os.walk(str(base_dir)):
            for filename in files:
                low = filename.lower()
                if "ltv" not in low:
                    continue
                path = Path(root) / filename
                matched_any.append(path)
                if periodo in filename:
                    matched_period.append(path)

    candidates = matched_period if matched_period else matched_any
    if not candidates:
        raise RuntimeError(
            "No se encontro archivo LTV de entrada. "
            "Copie un archivo con 'ltv' en el nombre en input/preproceso "
            "o configure GES_LTV_INPUT_FILE con la ruta exacta."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _previous_month_start(dt):
    first_day_current = dt.replace(day=1)
    previous_month_last = first_day_current - timedelta(days=1)
    return previous_month_last.replace(day=1)


spark_session = _get_spark_session()
if spark_session is None:
    raise RuntimeError("Este script requiere una sesion Spark activa de Databricks.")

# Periodo de ejecucion
hoy = datetime.today()
periodo_prediccion_param = _get_periodo_prediccion_param()
if periodo_prediccion_param is None:
    fecha_prediccion = _previous_month_start(hoy)
    periodo_prediccion = fecha_prediccion.year * 100 + fecha_prediccion.month
else:
    periodo_prediccion = periodo_prediccion_param
    year = int(str(periodo_prediccion)[:4])
    month = int(str(periodo_prediccion)[4:6])
    fecha_prediccion = datetime(year=year, month=month, day=1)

pre_ges = 0
hay_ges = 0

num_a_mes = {
    1: "ene",
    2: "feb",
    3: "mar",
    4: "abr",
    5: "may",
    6: "jun",
    7: "jul",
    8: "ago",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dic",
}
periodo = f"{num_a_mes[fecha_prediccion.month]}{fecha_prediccion:%y}"

# Paths y tablas
repo_dir = _resolve_repo_dir()
default_storage_root = os.getenv("GES_STORAGE_ROOT", "/dbfs/tmp/modelo_ltv")
inputs = Path(
    _normalize_local_path(
        os.getenv(
            "GES_PREPROCESO_INPUT_DIR",
            os.path.join(default_storage_root, "input", "preproceso"),
        )
    )
)
outputs = Path(
    _normalize_local_path(os.getenv("GES_OUTPUT_DIR", os.path.join(default_storage_root, "output")))
)
in_pred = Path(
    _normalize_local_path(
        os.getenv(
            "GES_PREDICCION_INPUT_DIR",
            os.path.join(default_storage_root, "input", "prediccion"),
        )
    )
)

inputs.mkdir(parents=True, exist_ok=True)
outputs.mkdir(parents=True, exist_ok=True)
in_pred.mkdir(parents=True, exist_ok=True)

aux_schema = os.getenv("GES_AUX_SQL_SCHEMA", "EST.P_DDV_EST").strip() or "EST.P_DDV_EST"
table_prm_categoria = os.getenv("GES_SQL_TABLE_PRM_CATEGORIA", f"{aux_schema}.PRM_CATEGORIA")
table_division_regiones = os.getenv(
    "GES_SQL_TABLE_DIVISION_REGIONES", f"{aux_schema}.DIVISION_REGIONES"
)
table_cod_comuna = os.getenv("GES_SQL_TABLE_COD_COMUNA", f"{aux_schema}.COD_COMUNA")
table_nse_y_pobreza = os.getenv("GES_SQL_TABLE_NSE_Y_POBREZA", f"{aux_schema}.NSE_Y_POBREZA")
table_ltv_input = os.getenv("GES_SQL_TABLE_LTV_INPUT", f"{aux_schema}.JC_PRED_LTV_INPUT")

# GES base desde Spark
df_ges_spark = _read_sql_table(spark_session, "EST.P_DDV_EST.JC_GES_PRED")
if "fld_pertermino" in df_ges_spark.columns:
    df_ges_spark = df_ges_spark.drop("fld_pertermino")
if "fld_termino" not in df_ges_spark.columns:
    raise RuntimeError("No se encontro columna fld_termino para reconstruir fld_pertermino.")

numeric_string_targets = {
    "id_titular": "bigint",
    "periodo": "int",
    "n_ges_mva_2m": "double",
}
current_dtypes = dict(df_ges_spark.dtypes)
for col_name, target_type in numeric_string_targets.items():
    if current_dtypes.get(col_name) == "string":
        df_ges_spark = df_ges_spark.withColumn(
            col_name, F.expr(f"try_cast(`{col_name}` as {target_type})")
        )

df_ges_spark = df_ges_spark.withColumn(
    "fld_pertermino",
    F.coalesce(
        F.expr("cast(date_format(try_to_date(`fld_termino`, 'MM/dd/yyyy'), 'yyyyMM') as int)"),
        F.expr("cast(date_format(try_to_date(`fld_termino`, 'dd/MM/yyyy'), 'yyyyMM') as int)"),
        F.expr("cast(date_format(try_to_date(`fld_termino`, 'yyyy/MM/dd'), 'yyyyMM') as int)"),
    ),
)

nombre_archivo = (
    f"{hoy:%y.%m.%d}_GES_{fecha_prediccion:%Y%m}_{num_a_mes[fecha_prediccion.month]}"
    f"{fecha_prediccion:%y}.gz"
)
ges_file = inputs / nombre_archivo
_write_single_csv(df_ges_spark, ges_file, compression="gzip")

# LTV input desde Spark a archivo local solo si no existe
nombre_archivo_ltv = (
    f"{hoy:%y.%m.%d}_LTV_{fecha_prediccion:%Y%m}_{num_a_mes[fecha_prediccion.month]}"
    f"{fecha_prediccion:%y}.gz"
)
ltv_file = inputs / nombre_archivo_ltv
if not ltv_file.exists():
    ltv_source = _read_sql_table(spark_session, table_ltv_input)
    _write_single_csv(ltv_source, ltv_file, compression="gzip")

# Lee LTV para el periodo
print("Buscando los datos de ltv del periodo...")
ltv_path = _resolve_ltv_file(inputs, periodo)
print(f"Se utilizara el archivo {ltv_path.name} para el preproceso")

df_ltv = (
    spark_session.read.option("header", "true")
    .option("inferSchema", "true")
    .csv(_to_spark_path(ltv_path))
)
df_ltv = _to_lowercase_columns(df_ltv)
_require_columns(df_ltv, ["id_titular", "periodo"], str(ltv_path))
df_ltv = df_ltv.withColumn("periodo", F.to_date(F.col("periodo").cast("string"), "yyyyMM"))

df_ges = _to_lowercase_columns(df_ges_spark)
df_ges = df_ges.withColumn(
    "huerfano",
    (F.col("fld_pertermino") < F.col("periodo")) & (F.col("fld_pertermino") != F.lit(190001)),
)
df_ges = df_ges.withColumn("periodo", F.to_date(F.col("periodo").cast("string"), "yyyyMM"))
df_ges = df_ges.filter(F.col("fld_pertermino").isNotNull()).drop("fld_pertermino")

# Merge principal
df = df_ltv.join(df_ges, on=["periodo", "id_titular"], how="left")

# Shift periodo para fugas
df = df.withColumn(
    "periodo",
    F.when(F.col("tipo_transac").isin(["D2", "D3"]), F.add_months(F.col("periodo"), -1)).otherwise(
        F.col("periodo")
    ),
)
df = df.filter(F.col("renta_imponible") >= 0)
df = _drop_existing_columns(df, ["anno", "mes", "fechaingreso"])

# Datos preferente
alto = [
    "CLINICA ALEMANA",
    "CLINICA LAS CONDES",
    "LOS ANDES",
    "SAN CARLOS",
    "ALEMANA-SC-LA-CSM / CLC",
    "SAN CARLOS - LOS ANDES",
    "SAN CARLOS - UC - LOS ANDES",
    "BALTICO",
    "SAN CARLOS - STA. MARIA - LOS ANDES",
    "ADRIATICO",
    "SAN CARLOS-UC",
]
medio = [
    "SANTA MARIA",
    "HOSP. CLINICO UNIVERSIDAD CATOLICA",
    "CLINICA UNIVERSIDAD CATOLICA",
    "TABANCURA",
    "MEDITERRANEO",
    "SAN CARLOS-UC",
    "HOSP.CLINICO UNIVERSIDAD DE CHILE",
    "INDISA INSTITUCIONAL",
    "CLINICA INDISA",
]
bajo = [
    "DAVILA",
    "BICENTENARIO-VESPUCIO-LAS LILAS",
    "UC SJ 110 - BICENTENARIO - VESPUCIO",
    "CORDILL.-BICENT.-AVANSALUD-SERVET",
    "BICENTENARIO-TABANCURA-DAVILA",
    "CITY COLMENA",
    "BICENTENARIO-VESPUCIO",
    "CLINICA CORDILLERA - CLINICA DAVILA",
    "UC SJ - BICENTENARIO - VESPUCIO",
]
le = ["LIBRE ELECCION"]

categoria = _to_lowercase_columns(_read_sql_table(spark_session, table_prm_categoria))
_require_columns(categoria, ["cod_categoria", "preferente"], table_prm_categoria)
df = df.join(
    categoria.select("cod_categoria", "preferente"),
    df["categoria_cod"] == categoria["cod_categoria"],
    how="left",
)

df = df.withColumn(
    "cat_linea_plan",
    F.when(F.col("preferente").isin(alto), F.lit("alto"))
    .when(F.col("preferente").isin(medio), F.lit("medio"))
    .when(F.col("preferente").isin(bajo), F.lit("bajo"))
    .when(F.col("preferente").isin(le), F.lit("le"))
    .otherwise(F.lit(None)),
)

replace_pref = {
    "UC ": "UNIVERSIDAD CATOLICA",
    "CSM": "STA. MARIA",
    "CLC": "CLINICA LAS CONDES",
    "BICENT.": "BICENTENARIO",
    "CORDILL.": "CLINICA CORDILLERA",
}
for old, new in replace_pref.items():
    df = df.withColumn(
        "preferente",
        F.when(F.col("preferente").isNotNull(), F.replace(F.col("preferente"), old, new)).otherwise(
            F.col("preferente")
        ),
    )

# Datos demograficos extras
division_raw = _read_sql_table(spark_session, table_division_regiones)
cod_region_col = _find_column_case_insensitive(division_raw.columns, "COD_REGION", "cod_region")
gls_region_col = _find_column_case_insensitive(
    division_raw.columns, "GLS_WEB_REGION", "gls_web_region"
)
norte_col = _find_column_case_insensitive(
    division_raw.columns, "norte_centro_sur", "NORTE_CENTRO_SUR", "Unnamed: 3"
)
if not cod_region_col or not gls_region_col or not norte_col:
    raise RuntimeError(
        f"La tabla {table_division_regiones} no contiene columnas esperadas para regiones."
    )

division_reg = (
    division_raw.select(
        F.col(cod_region_col).cast("int").alias("COD_REGION"),
        F.col(gls_region_col).alias("GLS_WEB_REGION"),
        F.col(norte_col).alias("norte_centro_sur"),
    )
    .filter(F.col("COD_REGION").isNotNull())
    .dropDuplicates(["COD_REGION"])
)

cod_comuna_raw = _read_sql_table(spark_session, table_cod_comuna)
con_comuna_col = _find_column_case_insensitive(cod_comuna_raw.columns, "con_comuna_gls")
cod_comuna_col = _find_column_case_insensitive(cod_comuna_raw.columns, "cod_comuna")
if not con_comuna_col or not cod_comuna_col:
    raise RuntimeError(f"La tabla {table_cod_comuna} no contiene con_comuna_gls/cod_comuna.")
cod_comuna = cod_comuna_raw.select(
    F.col(con_comuna_col).alias("con_comuna_gls"),
    F.col(cod_comuna_col).alias("cod_comuna"),
)

nse_raw = _read_sql_table(spark_session, table_nse_y_pobreza)
comuna_col = _find_column_case_insensitive(nse_raw.columns, "COMUNA", "comuna")
if not comuna_col:
    raise RuntimeError(f"La tabla {table_nse_y_pobreza} no contiene columna COMUNA/comuna.")
if comuna_col != "COMUNA":
    nse_raw = nse_raw.withColumnRenamed(comuna_col, "COMUNA")

nse_y_pobreza = nse_raw.join(
    cod_comuna, nse_raw["COMUNA"] == cod_comuna["con_comuna_gls"], how="inner"
).withColumn(
    "COD_REGION",
    F.expr("cast(substr(cast(cod_comuna as string), 1, length(cast(cod_comuna as string)) - 3) as int)"),
)
full_comuna = nse_y_pobreza.join(division_reg, on="COD_REGION", how="left")
df = df.join(full_comuna, df["comuna_cod"] == full_comuna["cod_comuna"], how="left")

# Limpieza
if "categoria_gls" in df.columns:
    df = df.withColumnRenamed("categoria_gls", "categor__a_gls")

drop_cols = [
    "region_cod",
    "comuna_cod",
    "fecha_nacimiento",
    "cod_sucursal",
    "centrocostos_cod",
    "detalle_producto",
    "cod_categoria",
    "con_comuna_gls",
    "cod_comuna",
    "COD_REGION",
    "GLS_WEB_REGION",
    "comuna_gls",
]
df = _drop_existing_columns(df, drop_cols)

cols_strip = [
    "region_gls",
    "sucursal_gls",
    "categor__a_gls",
    "serie",
    "tipo_plan",
    "tipo_producto",
    "actividad",
]
for col_name in cols_strip:
    if col_name in df.columns:
        df = df.withColumn(col_name, F.rtrim(F.col(col_name)))

# Features
tamaño_ventana = 5
df = df.withColumn(f"hay_preges_{tamaño_ventana}m", F.lit(pre_ges))
df = df.withColumn(f"hay_ges_{tamaño_ventana}m", F.lit(hay_ges))
df = df.withColumn("fuga_5m", F.lit(None).cast("double"))
df = df.withColumn("ant_m12_y_ges", F.col("antiguedad") <= F.lit(12 - tamaño_ventana))

window_hist = (
    Window.partitionBy("id_titular")
    .orderBy("periodo")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)
df = df.withColumn(
    "hubo_cie_complejo",
    F.max(F.coalesce(F.col("cie_complejo"), F.lit(0))).over(window_hist),
)
df = df.withColumn(
    "hay_gasto_licencia",
    ((F.col("gasto_licencias") != 0) | (F.col("gasto_licencias_excl") != 0)).cast("int"),
)
df = df.withColumn("hubo_licencia", F.max(F.col("hay_gasto_licencia")).over(window_hist))
datos = df

# Export a csv consumible por el script de prediccion
f = f"ready_to_pred_{periodo}.csv"
ready_path = in_pred / f
index_window = Window.orderBy(F.monotonically_increasing_id())
df_to_write = df.withColumn("index", F.row_number().over(index_window) - F.lit(1))
ordered_cols = ["index"] + [c for c in df_to_write.columns if c != "index"]
_write_single_csv(df_to_write.select(*ordered_cols), ready_path)

print((df.count(), len(df.columns)))
if "n_ges_mva_2m" in df.columns:
    df.select("n_ges_mva_2m").distinct().show(50, truncate=False)
    df.select("n_ges_mva_2m").show(5, truncate=False)
