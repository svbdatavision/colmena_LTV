#!/usr/bin/env python
# coding: utf-8

# #  Template para Cálculo del LTV
# 
# En este notebook se calcula el **LTV estimado o predicho** para medir el rendimiento del modelo, usando los modelos antes entrenados para cada uno de los módulos.
# 
# Para ello, a partir de las predicciones de percentil de ingresos y costos, se calcula su valor correspondiente, y sumado a la probabilidad de fuga se tiene todo para calcular el LTV.
# 
# $$ LTV =  \sum_{k=1}^3 \frac{1}{(1+t)^k} (1-P_{fuga}(k)) \widehat{margen}(k)$$
# 
# 
# 
# Para ello se calcula el 'margen_estimado' como la resta entre el 'ingreso_estimado' y el 'costo_estimado'
# 
# * Input: 
#     - LTV_{mes}.csv
#     - `LTV.LTV_features_resumen_pasado_PIT`. Dataframe con información de todos los afiliados, con historial hacia el pasado, para varios PIT, todo calculado desde bigQuery, reducido de categorías tras limpieza en preprocesing.
#     - Cada uno de los modelos de h2o.
# * Output:
#    - LTV.PREDICCION_MODULOS_LTV_{mes}{año}
#    - LTV.PREDICCION_LTV_CATLTV_{mes}{año}
#     

# In[ ]:


import time
import os
import sys
import shutil
from pathlib import Path
from numba import njit, prange
import pandas as pd
import json
try:
    import great_expectations as ge
except Exception:
    ge = None
import numpy as np
import h2o
import re
from sklearn.preprocessing import QuantileTransformer
try:
    from google.cloud import bigquery
except Exception:
    bigquery = None
np.random.seed(54321)

try:
    import snowflake.connector as snow_con
except Exception:
    snow_con = None
import getpass as gp

# import feather
try:
    from IPython.display import display, HTML
except Exception:
    display = None
    HTML = None


def _run_ipython_magic(name, value):
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        if ip is not None:
            ip.run_line_magic(name, value)
    except Exception:
        pass


def _normalize_local_path(path_value):
    if path_value.startswith("dbfs:/"):
        return "/dbfs/" + path_value[len("dbfs:/"):].lstrip("/")
    return path_value


def _resolve_repo_dir():
    try:
        return Path(__file__).resolve().parent
    except Exception:
        try:
            nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # type: ignore[name-defined]
            return Path(f"/Workspace{nb_path}").resolve().parent
        except Exception:
            return Path.cwd()


def _get_spark_session():
    try:
        return spark  # type: ignore[name-defined]
    except Exception:
        try:
            from pyspark.sql import SparkSession
            return SparkSession.getActiveSession()
        except Exception:
            return None


def _get_secret(scope, key):
    if not scope or not key:
        return None
    try:
        return dbutils.secrets.get(scope=scope, key=key)  # type: ignore[name-defined]
    except Exception:
        return None


def _get_credential(env_name, prompt_label):
    value = os.getenv(env_name)
    if value:
        return value

    secret_scope = os.getenv("SNOWFLAKE_SECRET_SCOPE")
    secret_key = os.getenv(f"{env_name}_KEY")
    secret_value = _get_secret(secret_scope, secret_key)
    if secret_value:
        return secret_value

    if sys.stdin and sys.stdin.isatty():
        return gp.getpass(prompt_label)

    raise RuntimeError(
        f"No se encontro {env_name}. Configura variable de entorno o secreto en Databricks."
    )


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
        raise ValueError(
            "periodo_prediccion debe tener formato YYYYMM (ejemplo: 202501)."
        )
    return int(raw_value)


def _load_h2o_model_compatible(model_base_path, model_format="auto", model_label="modelo"):
    model_format = (model_format or "auto").strip().lower()
    if model_format not in {"auto", "binary", "mojo"}:
        raise ValueError(
            f"model_format invalido ({model_format}) para {model_label}. "
            "Usar: auto, binary o mojo."
        )

    base_path = Path(_normalize_local_path(str(model_base_path)))
    mojo_candidates = []
    binary_candidates = []

    if base_path.suffix.lower() in {".zip", ".mojo"}:
        mojo_candidates = [base_path]
    elif base_path.suffix:
        binary_candidates = [base_path]
    else:
        binary_candidates = [base_path]
        mojo_candidates = [
            Path(str(base_path) + ".zip"),
            Path(str(base_path) + ".mojo"),
            base_path / "mojo.zip",
            base_path / "model.zip",
        ]

    # Dedup conservando orden
    seen = set()
    mojo_candidates = [p for p in mojo_candidates if not (str(p) in seen or seen.add(str(p)))]
    seen = set()
    binary_candidates = [p for p in binary_candidates if not (str(p) in seen or seen.add(str(p)))]

    attempts = []
    if model_format in {"auto", "mojo"}:
        attempts.extend([("mojo", p) for p in mojo_candidates if p.exists()])
    if model_format in {"auto", "binary"}:
        attempts.extend([("binary", p) for p in binary_candidates if p.exists()])

    if not attempts:
        expected = [str(p) for p in (mojo_candidates + binary_candidates)]
        raise FileNotFoundError(
            f"No se encontro artefacto para {model_label}. "
            f"Paths evaluados: {expected}"
        )

    last_error = None
    for artifact_type, artifact_path in attempts:
        try:
            if artifact_type == "mojo":
                print(f"Cargando {model_label} como MOJO: {artifact_path}")
                return h2o.import_mojo(str(artifact_path))
            print(f"Cargando {model_label} como binario H2O: {artifact_path}")
            return h2o.load_model(path=str(artifact_path))
        except Exception as exc:
            last_error = exc
            err_text = str(exc)
            if "Found version" in err_text and "running version" in err_text:
                raise RuntimeError(
                    f"No se pudo cargar {model_label} por incompatibilidad de version H2O. "
                    f"Detalle: {err_text}. "
                    "Recomendacion: convertir este modelo a MOJO en un entorno legacy "
                    "(H2O 3.20.x + Java 8) y volver a ejecutar con LTV_H2O_MODEL_FORMAT=mojo."
                ) from exc

    raise RuntimeError(
        f"No se pudo cargar {model_label}. Ultimo error: {last_error}"
    ) from last_error


def _load_json_if_exists(json_path):
    p = Path(json_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_first_available_model(model_candidates, model_format="auto", model_label="modelo"):
    if not model_candidates:
        raise ValueError(f"Sin candidatos para {model_label}")

    missing = []
    last_error = None
    for candidate in model_candidates:
        try:
            return _load_h2o_model_compatible(
                candidate,
                model_format=model_format,
                model_label=model_label,
            )
        except FileNotFoundError:
            missing.append(str(candidate))
        except Exception as exc:
            last_error = exc
            break

    if last_error is not None:
        raise last_error
    raise FileNotFoundError(
        f"No se encontro modelo para {model_label}. Paths evaluados: {missing}"
    )


_run_ipython_magic('load_ext', 'autoreload')
_run_ipython_magic('autoreload', '2')

if display is not None and HTML is not None:
    display(HTML("<style>.container { width:95% !important; }</style>"))
pd.set_option('display.max_columns',2000)


# In[ ]:


#1. Ubicación de datos y nombre de proyecto (Varía entre Colmena y Spike)
###########################################################
pid = "estudios-242917"
spark_session = _get_spark_session()
default_ltv_base_candidate = Path("/Volumes/bigdata/default/ml_models/modelo_ltv/ltv")
if default_ltv_base_candidate.exists():
    default_ltv_base = str(default_ltv_base_candidate)
else:
    default_ltv_base = "/dbfs/mnt/modelos/ltv" if spark_session is not None else "/estudio/data/ltv"
carpeta_ltv_path = Path(_normalize_local_path(os.getenv("LTV_BASE_DIR", default_ltv_base)))
ltv_h2o_model_format = os.getenv("LTV_H2O_MODEL_FORMAT", "auto")
carpeta_ltv = f"{carpeta_ltv_path.as_posix()}/"
datos_ltv_folder = f"{(carpeta_ltv_path / 'Input').as_posix()}/"
#asume que existen las carpetas "Greats_expectations_LTV", "models", "Output"
for required_dir in [
    Path(datos_ltv_folder),
    carpeta_ltv_path / "Greats_expectations_LTV",
    carpeta_ltv_path / "Output",
]:
    required_dir.mkdir(parents=True, exist_ok=True)

#2. Cambiar: Qué periodo se va a predecir?
###########################################################
#descomentar si se quiere definir el periodo manualmente
#periodo_de_prediccion = '202201' #añomes. p.ej'201905'
periodo_prediccion_param = _get_periodo_prediccion_param()
if periodo_prediccion_param is None:
    fecha_prediccion = pd.to_datetime("today") - pd.DateOffset(months=1, day=1)
    periodo_prediccion = fecha_prediccion.year * 100 + fecha_prediccion.month
else:
    periodo_prediccion = periodo_prediccion_param
    fecha_prediccion = pd.to_datetime(str(periodo_prediccion), format="%Y%m")


#La base debe tener el periodo que se va a predecir en su nombre ej. -> marzo: 20.04.16 LTV_202003.csv

# descomentar si se sube el archivo manualmente
# for _, _, files in os.walk(datos_ltv_folder):
#     for file in files:
#         if periodo_de_prediccion in file and 'ltv' in file.lower():  
#             nombre_base = file
#             break
# print(nombre_base)


## 22.01.18_LTV_202112_dic21



###########################################################

num_a_mes = {
    1: 'ene', 2: 'feb', 3: 'mar',  4: 'abr',  5: 'may',  6: 'jun',
    7: 'jul', 8: 'ago', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dic'
}

mes_actual = fecha_prediccion.month
año = f"{fecha_prediccion:%y}"
nombre_fancy = f"{num_a_mes[fecha_prediccion.month]}{fecha_prediccion:%y}" #P ej: "may19"
                
nombre_base = f"{fecha_prediccion:%y.%m.%d}_LTV_{fecha_prediccion:%Y%m}_{num_a_mes[fecha_prediccion.month]}{fecha_prediccion:%y}.gz"
    

if mes_actual == 1:
    mes_anterior = 12
    nombre_fancy_mes_anterior = num_a_mes[mes_anterior] + str((int(año)-1))
else:
    mes_anterior = mes_actual - 1
    nombre_fancy_mes_anterior = num_a_mes[mes_anterior] + año

nombre_tabla = f"LTV.ltv_{nombre_fancy}"
nombre_base_consolidada = "LTV_NEW_" + nombre_fancy
nombre_ltv_new_mes_anterior = "LTV_NEW_" + nombre_fancy_mes_anterior


start = time.time()
print(time.asctime(time.localtime(start)))


# In[ ]:


print(nombre_tabla, nombre_base_consolidada, nombre_ltv_new_mes_anterior, nombre_base)


# In[ ]:


snow_user = os.getenv("SNOWFLAKE_USER")
snow_pass = os.getenv("SNOWFLAKE_PASSWORD")
snow_account = os.getenv("SNOWFLAKE_ACCOUNT", "isapre_colmena.us-east-1")
snow_warehouse = "P_OPX"


# In[ ]:




if spark_session is not None:
    print("Leyendo EST.P_DDV_EST.JC_PRED_LTV_INPUT desde Spark/Databricks...")
    data_ltv_snow = spark_session.table("EST.P_DDV_EST.JC_PRED_LTV_INPUT").toPandas()
else:
    if snow_con is None:
        raise ImportError(
            "snowflake-connector-python no esta disponible. Ejecuta en Databricks con Spark o instala la libreria."
        )
    if not snow_user:
        snow_user = _get_credential("SNOWFLAKE_USER", "snow_user")
    if not snow_pass:
        snow_pass = _get_credential("SNOWFLAKE_PASSWORD", "snow_pass")

    snow = snow_con.connect(user=snow_user,
                            password=snow_pass,
                            account=snow_account,
                            warehouse=snow_warehouse
                            )
    with snow.cursor() as cur:
        print(f"{periodo_prediccion}")
        cur.execute(f"SET (per_desde, per_hasta) = ({periodo_prediccion}, {periodo_prediccion}); ")
        cur.execute('SELECT * FROM EST.P_DDV_EST.JC_PRED_LTV_INPUT;')
        data_ltv_snow = cur.fetch_pandas_all()

data_ltv_snow.to_csv(datos_ltv_folder + nombre_base, index=False, compression='gzip')


# In[ ]:


data_ltv_snow.describe()


# In[ ]:


data_ltv_snow.head()


# ### Guardar Columnas de modelos de h2o

# In[ ]:


cols_mod1 = set(["comuna_gls", "sucursal_gls",
"actividad", "detalle_producto", "region_gls", "tipo_transac", "tipo_trabajador", "serie",
"vigente", "tipo_plan", "tipo_producto", "fechaingreso", "antiguedad", "num_cargas",
"edad", "num_empleadores", "seg_muerte", "cambio_plan", "costo_final", "precio_base", "factor_riesgo",
"costo_total", "costo_benef_adic", "pagado", "excesos", "excedentes", "gasto_Ambulatorios",
"gasto_Hospitalarios", "gasto_Hospitalarios_excl", "gasto_ges", "gasto_caec", "gasto_pharma",
"recuperacion_gastos", "iva_cotizaciones", "iva_recuperado", "gasto_Licencias", "gasto_Licencias_excl", "cie_complejo",
"prestacion_amb_compleja", "renta_imponible","gasto_parto"])

cols_mod2 = set(["fechaingreso", "antiguedad", "num_cargas", "vigente", "region_gls", "comuna_gls",
"sucursal_gls", "edad", "num_empleadores", "actividad", "tipo_trabajador", "tipo_transac", "serie",
"tipo_plan", "tipo_producto", "detalle_producto", "seg_muerte", "cambio_plan", "costo_final", "precio_base",
"factor_riesgo", "costo_total", "costo_benef_adic", "pagado", "excesos", "excedentes", "gasto_Ambulatorios",
"gasto_Hospitalarios", "gasto_Hospitalarios_excl", "gasto_ges", "gasto_caec", "gasto_pharma", "recuperacion_gastos", "iva_cotizaciones",
"iva_recuperado", "gasto_Licencias", "gasto_Licencias_excl", "cie_complejo", "prestacion_amb_compleja",
"renta_imponible", "gasto_parto"])

cols_mod3 = set(["fechaingreso", "antiguedad", "num_cargas", "vigente", "region_gls", "comuna_gls",
"sucursal_gls", "edad", "num_empleadores", "actividad", "tipo_trabajador", "tipo_transac", "serie",
"tipo_plan", "tipo_producto", "detalle_producto", "seg_muerte", "cambio_plan", "costo_final",
"precio_base", "factor_riesgo", "costo_total", "costo_benef_adic", "pagado", "excesos", "excedentes",
"gasto_Ambulatorios", "gasto_Hospitalarios", "gasto_Hospitalarios_excl", "gasto_ges", "gasto_caec", "gasto_pharma",
"recuperacion_gastos", "iva_cotizaciones", "iva_recuperado", "gasto_Licencias", "gasto_Licencias_excl", "cie_complejo",
"prestacion_amb_compleja", "renta_imponible", "gasto_parto"])

(pd.DataFrame({'lista_variables': list(cols_mod1.union(cols_mod2).union(cols_mod3))})
   .to_csv(datos_ltv_folder + "nombres_variables_usadas_por_ltv"))

print("Ok")


# ## Correr update de LTV

# ### H2O Cargamos las tablas

# In[ ]:


h2o.init(port=54321, min_mem_size="2g", max_mem_size="4g")
h2o.remove_all()


# In[ ]:


nombre_base


# In[ ]:


data_ltv_snow.head()


# In[ ]:


# data_ltv2_old = pd.read_csv(datos_ltv_folder + '22.01.18_LTV_202112_dic21.csv.zip', sep=';', compression='zip').rename(columns = lambda x: x.lower()).rename(columns={'categoria_gls': 'categor__a_gls'})
data_ltv2 = data_ltv_snow.rename(columns = lambda x: x.lower()).rename(columns={'categoria_gls': 'categor__a_gls'})
drop = ['agencia_vn', 'agente_vn', 'clasif_morosidad',
       'clasif_riesgo', 'costo_final_mva_1to6m', 'per', 'periodo_anual',
       'periodo_ult_vig', 'rut', 'rut_titular']
data_ltv2.drop(columns=drop, inplace=True)
data_ltv2.head()


# In[ ]:


# mapeo_col

mapeo_col = {'detalle_producto': 'detalle_producto',
 'id_titular': 'id_titular',
 'gasto_caec': 'gasto_caec',
 'region_gls': 'region_gls',
 'anno': 'anno',
 'actividad': 'actividad',
 'costo_total': 'costo_total',
 'benef_adic': 'costo_benef_adic',
 'seg_muerte': 'seg_muerte',
 'gasto_parto': 'gasto_parto',
 'fechaingreso': 'fechaingreso',
 'categoria_cod': 'categoria_cod',
 'recuperacion_gastos': 'recuperacion_gastos',
 'excesos': 'excesos',
 'linea_plan': 'linea_plan',
 'factor_riesgo': 'factor_riesgo',
 'tipo_plan': 'tipo_plan',
 'gasto_ges': 'gasto_ges',
 'contrato': 'contrato',
 'categor__a_gls': 'categor__a_gls',
 'cie_complejo': 'cie_complejo',
 'centrocostos_cod': 'centrocostos_cod',
 'edad': 'edad',
 'cambio_plan': 'cambio_plan',
 'gasto_hospitalarios': 'gasto_Hospitalarios',
 'tipo_transac': 'tipo_transac',
 'iva_recuperado': 'iva_recuperado',
 'gasto_pharma': 'gasto_pharma',
 'sucursal_gls': 'sucursal_gls',
 'tipo_producto': 'tipo_producto',
 'centro_costo_gls': 'centro_costo_gls',
 'precio_base': 'precio_base',
 'periodo': 'periodo',
 'antiguedad': 'antiguedad',
 'fecha_nacimiento': 'fecha_nacimiento',
 'vigente': 'vigente',
 'gasto_hospitalarios_excl': 'gasto_Hospitalarios_excl',
 'iva_cotizaciones': 'iva_cotizaciones',
 'region_cod': 'region_cod',
 'valor_uf': 'valor_uf',
 'comuna_gls': 'comuna_gls',
 'excedentes': 'excedentes',
 'mes': 'mes',
 'gasto_licencias': 'gasto_Licencias',
 'gasto_licencias_excl': 'gasto_Licencias_excl',
 'gasto_ambulatorios': 'gasto_Ambulatorios',
 'tipo_trabajador': 'tipo_trabajador',
 'comuna_cod': 'comuna_cod',
 'num_empleadores': 'num_empleadores',
 'correlativo': 'correlativo',
 'sucursal_cod': 'sucursal_cod',
 'pagado': 'pagado',
 'costo_final': 'costo_final',
 'prestacion_amb_compleja': 'prestacion_amb_compleja',
 'serie': 'serie',
 'renta_imponible': 'renta_imponible',
 'sexo': 'sexo',
 'num_cargas': 'num_cargas',
 'costo_total_mva_1to6m': 'Costo_Total_mva_1to6m',
 'precio_base_mva_1to6m': 'precio_base_mva_1to6m',
 'excesos_mva_1to6m': 'Excesos_mva_1to6m',
 'excedentes_mva_1to6m': 'Excedentes_mva_1to6m',
 'factor_riesgo_mva_1to6m': 'factor_riesgo_mva_1to6m',
 'iva_recuperado_mva_1to6m': 'iva_recuperado_mva_1to6m',
 'iva_cotizaciones_mva_1to6m': 'iva_cotizaciones_mva_1to6m',
 'recuperacion_gastos_mva_1to6m': 'recuperacion_gastos_mva_1to6m',
 'gasto_ambulatorios_mva_1to6m': 'gasto_Ambulatorios_mva_1to6m',
 'gasto_caec_mva_1to6m': 'Gasto_Caec_mva_1to6m',
 'gasto_ges_mva_1to6m': 'Gasto_Ges_mva_1to6m',
 'gasto_hospitalarios_mva_1to6m': 'gasto_Hospitalarios_mva_1to6m',
 'gasto_hospitalarios_excl_mva_1to6m': 'gasto_Hospitalarios_Excl_mva_1to6m',
 'gasto_licencias_mva_1to6m': 'gasto_Licencias_mva_1to6m',
 'gasto_licencias_excl_mva_1to6m': 'gasto_Licencias_Excl_mva_1to6m',
 'gasto_pharma_mva_1to6m': 'gasto_pharma_mva_1to6m',
 'gasto_parto_mva_1to6m': 'Gasto_Parto_mva_1to6m',
 'renta_imponible_mva_1to6m': 'renta_imponible_mva_1to6m',
 'costos_mva_1to6m': 'Costos_mva_1to6m',
 'ingresos_mva_1to6m': 'Ingresos_mva_1to6m',
 'margen_mva_1to6m': 'Margen_mva_1to6m',
 'cie_complejo_mva_1to6m': 'cie_complejo_mva_1to6m',
 'prestacion_amb_compleja_mva_1to6m': 'prestacion_amb_compleja_mva_1to6m',
 'costo_total_mva_7to12m': 'Costo_Total_mva_7to12m',
 'precio_base_mva_7to12m': 'precio_base_mva_7to12m',
 'excesos_mva_7to12m': 'Excesos_mva_7to12m',
 'excedentes_mva_7to12m': 'Excedentes_mva_7to12m',
 'factor_riesgo_mva_7to12m': 'factor_riesgo_mva_7to12m',
 'iva_recuperado_mva_7to12m': 'iva_recuperado_mva_7to12m',
 'iva_cotizaciones_mva_7to12m': 'iva_cotizaciones_mva_7to12m',
 'recuperacion_gastos_mva_7to12m': 'recuperacion_gastos_mva_7to12m',
 'gasto_ambulatorios_mva_7to12m': 'gasto_Ambulatorios_mva_7to12m',
 'gasto_caec_mva_7to12m': 'Gasto_Caec_mva_7to12m',
 'gasto_ges_mva_7to12m': 'Gasto_Ges_mva_7to12m',
 'gasto_hospitalarios_mva_7to12m': 'gasto_Hospitalarios_mva_7to12m',
 'gasto_hospitalarios_excl_mva_7to12m': 'gasto_Hospitalarios_Excl_mva_7to12m',
 'gasto_licencias_mva_7to12m': 'gasto_Licencias_mva_7to12m',
 'gasto_licencias_excl_mva_7to12m': 'gasto_Licencias_Excl_mva_7to12m',
 'gasto_pharma_mva_7to12m': 'gasto_pharma_mva_7to12m',
 'gasto_parto_mva_7to12m': 'Gasto_Parto_mva_7to12m',
 'renta_imponible_mva_7to12m': 'renta_imponible_mva_7to12m',
 'costos_mva_7to12m': 'Costos_mva_7to12m',
 'ingresos_mva_7to12m': 'Ingresos_mva_7to12m',
 'margen_mva_7to12m': 'Margen_mva_7to12m',
 'cie_complejo_mva_7to12m': 'cie_complejo_mva_7to12m',
 'prestacion_amb_compleja_mva_7to12m': 'prestacion_amb_compleja_mva_7to12m',
 'costo_total_mva_13to24m': 'Costo_Total_mva_13to24m',
 'precio_base_mva_13to24m': 'precio_base_mva_13to24m',
 'excesos_mva_13to24m': 'Excesos_mva_13to24m',
 'excedentes_mva_13to24m': 'Excedentes_mva_13to24m',
 'factor_riesgo_mva_13to24m': 'factor_riesgo_mva_13to24m',
 'iva_recuperado_mva_13to24m': 'iva_recuperado_mva_13to24m',
 'iva_cotizaciones_mva_13to24m': 'iva_cotizaciones_mva_13to24m',
 'recuperacion_gastos_mva_13to24m': 'recuperacion_gastos_mva_13to24m',
 'gasto_ambulatorios_mva_13to24m': 'gasto_Ambulatorios_mva_13to24m',
 'gasto_caec_mva_13to24m': 'Gasto_Caec_mva_13to24m',
 'gasto_ges_mva_13to24m': 'Gasto_Ges_mva_13to24m',
 'gasto_hospitalarios_mva_13to24m': 'gasto_Hospitalarios_mva_13to24m',
 'gasto_hospitalarios_excl_mva_13to24m': 'gasto_Hospitalarios_Excl_mva_13to24m',
 'gasto_licencias_mva_13to24m': 'gasto_Licencias_mva_13to24m',
 'gasto_licencias_excl_mva_13to24m': 'gasto_Licencias_Excl_mva_13to24m',
 'gasto_pharma_mva_13to24m': 'gasto_pharma_mva_13to24m',
 'gasto_parto_mva_13to24m': 'Gasto_Parto_mva_13to24m',
 'renta_imponible_mva_13to24m': 'renta_imponible_mva_13to24m',
 'costos_mva_13to24m': 'Costos_mva_13to24m',
 'ingresos_mva_13to24m': 'Ingresos_mva_13to24m',
 'margen_mva_13to24m': 'Margen_mva_13to24m',
 'cie_complejo_mva_13to24m': 'cie_complejo_mva_13to24m',
 'prestacion_amb_compleja_mva_13to24m': 'prestacion_amb_compleja_mva_13to24m',
 'costos_sum_future_1y': 'Costos_sum_future_1y',
 'ingresos_sum_future_1y': 'Ingresos_sum_future_1y',
 'margen_sum_future_1y': 'Margen_sum_future_1y',
 'costos_avg_future_1y': 'Costos_avg_future_1y',
 'ingresos_avg_future_1y': 'Ingresos_avg_future_1y',
 'margen_avg_future_1y': 'Margen_avg_future_1y',
 'costos_sum_future_2y': 'Costos_sum_future_2y',
 'ingresos_sum_future_2y': 'Ingresos_sum_future_2y',
 'margen_sum_future_2y': 'Margen_sum_future_2y',
 'costos_avg_future_2y': 'Costos_avg_future_2y',
 'ingresos_avg_future_2y': 'Ingresos_avg_future_2y',
 'margen_avg_future_2y': 'Margen_avg_future_2y',
 'costos_sum_future_3y': 'Costos_sum_future_3y',
 'ingresos_sum_future_3y': 'Ingresos_sum_future_3y',
 'margen_sum_future_3y': 'Margen_sum_future_3y',
 'costos_avg_future_3y': 'Costos_avg_future_3y',
 'ingresos_avg_future_3y': 'Ingresos_avg_future_3y',
 'margen_avg_future_3y': 'Margen_avg_future_3y'}

# for col in data_ltv2.columns:
#     for col2 in data_ltv.columns:
#         if col.lower() == col2.lower():
#             mapeo_col[col] = col2
            
#mapeo_col


# In[ ]:


data_ltv  = data_ltv2.rename(columns=mapeo_col)


# In[ ]:


df = data_ltv
#df.drop(labels=['periodo','ID_PIT'], axis=1, inplace = True)
#df.rename(columns = {'keyPIT_Date':'periodo'}, inplace = True)
df['periodo'] = pd.to_datetime(df['periodo'], format ='%Y%m')
df['fechaingreso'] = pd.to_datetime(df['fechaingreso'])


# In[ ]:


df.head()


# # Arreglamos las categorías con demasiados puntos

# ### Categoria_a

# In[ ]:


df['categ_simplificada'] = df['categor__a_gls'].apply(lambda x: (re.findall('^\D+',x)[0]))
print('Pasamos de tener {} categorías, a tener {} después de simplificar'.format(
             len(set(df['categor__a_gls'])),len(set(df['categ_simplificada']))))


# ### Linea plan

# In[ ]:


df['linea_plan_simplificada'] = df['linea_plan'].apply(lambda x: (re.findall('^\D+',x)[0]) if len(x) > 0 else x)
print('Pasamos de tener {} categorías, a tener {} después de simplificar'.format(
len(set(df['linea_plan'])),len(set(df['linea_plan_simplificada']))))


# ### Borramos las categorías entonces

# In[ ]:


[m for m in df.columns if 'cat' in m]
df.drop(labels = ['categoria_cod', 'categor__a_gls','linea_plan'], axis = 1, inplace = True)

borramos = ['contrato','correlativo','fecha_nacimiento'] + [col for col in df.columns if 'fut' in col]
df.drop(labels = borramos, axis = 1, inplace = True)

df[[m for m in df.columns if 'cod' in m]].head()
df.drop(labels = [m for m in df.columns if 'cod' in m], axis = 1, inplace = True)


# # Categorizamos el dataframe

# In[ ]:


df.head()


# In[ ]:


categoricals =  ['region_gls', 'comuna_gls', 'sucursal_gls', 'centro_costo_gls', 'vigente', 
                 'actividad','Tipo_Trab', 'Region_cod', 'tipo_transac', 'serie',
                 'tipo_plan','tipo_producto',
                 'detalle_producto','linea_plan_simplificada',
                 'tipo_trabajador', 'categ_simplificada']

for tipo in df.columns:
    if tipo in categoricals:
        df[tipo] = df[tipo].astype('category')
        
df.drop(labels=['sexo'], axis=1, inplace=True)


# In[ ]:


# Acá se hace check de Great Expectations intermedio
df.to_csv(f'{carpeta_ltv}Greats_expectations_LTV/ltv_{nombre_fancy}_inputs.csv')
expectations_inputs_path = f'{carpeta_ltv}Greats_expectations_LTV/ltv_expectations_marzo.json'
my_expectations_config = _load_json_if_exists(expectations_inputs_path)
if ge is not None and my_expectations_config is not None:
    #El nombre fancy corresponde al mes de ltv predicho que se quiere validar, entregado al principio del script
    my_df = ge.read_csv(
        f'{carpeta_ltv}Greats_expectations_LTV/ltv_{nombre_fancy}_inputs.csv',
        expectations_config=my_expectations_config,
    )

    # Por ahora se eliminan acá los valores negativos en estas variables, que no deberían estar
    my_df.loc[my_df['gasto_Licencias_excl'] < 0., 'gasto_Licencias_excl'] = 0 # Tiene un valor con (-)
    my_df.loc[my_df['costo_total'] < 0., 'costo_total'] = 0 # Tiene 213 valores con (-)
    my_df.loc[my_df['factor_riesgo'] < 0., 'factor_riesgo'] = 0 # Tiene 158 valores con (-)
    my_df.loc[my_df['gasto_Licencias_mva_13to24m'] < 0., 'gasto_Licencias_mva_13to24m'] = 0 # Tiene dos valores con (-)
    my_df.loc[my_df['gasto_Licencias_Excl_mva_7to12m'] < 0., 'gasto_Licencias_Excl_mva_7to12m'] = 0 # Tiene un valor con (-)

    resultado = my_df.validate(result_format='BASIC', only_return_failures=True)

    with open(f'{carpeta_ltv}Greats_expectations_LTV/resultado_GE_{nombre_fancy}_inputs.json', 'w') as outfile:
        json.dump(resultado, outfile)
    print("Guardado!")
else:
    print(
        f"Great Expectations intermedio omitido. "
        f"ge_disponible={ge is not None}, config_encontrada={my_expectations_config is not None}, "
        f"path={expectations_inputs_path}"
    )

#assert resultado['success']


# ### Cargar última versión de los modelos predictivos

# In[ ]:


# original spike
# modelos_ingresos = {}
# modelos_costos = {}
# modelos_fuga = {}

# carpeta_models_automl = f"{carpeta_ltv}models/h2o_models/modelsautoml/"
# print(carpeta_models_automl)
# modelos_ingresos['1y'] = h2o.load_model(f'{carpeta_models_automl}ingresos_avg/1y/StackedEnsemble_AllModels_0_AutoML_20190103_170253')
# modelos_ingresos['2y'] = h2o.load_model(f'{carpeta_models_automl}ingresos_avg/2y/StackedEnsemble_AllModels_0_AutoML_20190103_190356')
# modelos_ingresos['3y'] = h2o.load_model(f'{carpeta_models_automl}ingresos_avg/3y/StackedEnsemble_AllModels_0_AutoML_20190103_210454')

# modelos_costos['1y'] = h2o.load_model(f'{carpeta_models_automl}costos_avg/1y/StackedEnsemble_AllModels_0_AutoML_20190103_170925')
# modelos_costos['2y'] = h2o.load_model(f'{carpeta_models_automl}costos_avg/2y/StackedEnsemble_AllModels_0_AutoML_20190103_191041')
# modelos_costos['3y'] = h2o.load_model(f'{carpeta_models_automl}costos_avg/3y/StackedEnsemble_AllModels_0_AutoML_20190103_211140')

# modelos_fuga['1y'] = h2o.load_model(f'{carpeta_models_automl}fuga/y1/StackedEnsemble_AllModels_0_AutoML_20181211_175510')
# modelos_fuga['2y'] = h2o.load_model(f'{carpeta_models_automl}fuga/y2/StackedEnsemble_AllModels_0_AutoML_20181211_195722')
# modelos_fuga['3y'] = h2o.load_model(f'{carpeta_models_automl}fuga/y3/StackedEnsemble_AllModels_0_AutoML_20181211_215935')


# In[ ]:


modelos_ingresos = {}
modelos_costos = {}
modelos_fuga = {}
keys = {'1y', '2y','3y'}
folders = {'fuga', 'costos', 'ingresos'}
carpeta_models_automl = f"{carpeta_ltv}models/h2o_models/modelsautoml/"
print(carpeta_models_automl)

legacy_stack_map = {
    ("ingresos", "1y"): "StackedEnsemble_AllModels_0_AutoML_20190103_170253",
    ("ingresos", "2y"): "StackedEnsemble_AllModels_0_AutoML_20190103_190356",
    ("ingresos", "3y"): "StackedEnsemble_AllModels_0_AutoML_20190103_210454",
    ("costos", "1y"): "StackedEnsemble_AllModels_0_AutoML_20190103_170925",
    ("costos", "2y"): "StackedEnsemble_AllModels_0_AutoML_20190103_191041",
    ("costos", "3y"): "StackedEnsemble_AllModels_0_AutoML_20190103_211140",
    ("fuga", "1y"): "StackedEnsemble_AllModels_0_AutoML_20181211_175510",
    ("fuga", "2y"): "StackedEnsemble_AllModels_0_AutoML_20181211_195722",
    ("fuga", "3y"): "StackedEnsemble_AllModels_0_AutoML_20181211_215935",
}

for folder in folders:
    for key in keys:
        print(folder + ' - ' + key)
        if folder == 'ingresos':
            model_candidates = [
                f'{carpeta_models_automl}ingresos_avg/'+key+'/retrain_ltv_ingresos_' + key,
                f'{carpeta_models_automl}ingresos_avg/'+key+'/' + legacy_stack_map[(folder, key)],
            ]
            modelos_ingresos[key] = _load_first_available_model(
                model_candidates,
                model_format=ltv_h2o_model_format,
                model_label=f"ingresos_{key}",
            )
        elif folder == 'costos':
            model_candidates = [
                f'{carpeta_models_automl}costos_avg/' +key+'/retrain_ltv_costos_' + key,
                f'{carpeta_models_automl}costos_avg/' +key+'/' + legacy_stack_map[(folder, key)],
            ]
            modelos_costos[key] = _load_first_available_model(
                model_candidates,
                model_format=ltv_h2o_model_format,
                model_label=f"costos_{key}",
            )
        else:
            model_candidates = [
                f'{carpeta_models_automl}fuga/' +key[1]+key[0]+'/retrain_ltv_fuga_' + key[1]+key[0],
                f'{carpeta_models_automl}fuga/' +key+'/retrain_ltv_fuga_' + key,
                f'{carpeta_models_automl}fuga/' +key[1]+key[0]+'/' + legacy_stack_map[(folder, key)],
                f'{carpeta_models_automl}fuga/' +key+'/' + legacy_stack_map[(folder, key)],
            ]
            modelos_fuga[key] = _load_first_available_model(
                model_candidates,
                model_format=ltv_h2o_model_format,
                model_label=f"fuga_{key}",
            )

# modelos_ingresos['1y'] = h2o.load_model(f'{carpeta_models_automl}ingresos_avg/1y/StackedEnsemble_AllModels_0_AutoML_20190103_170253')
# modelos_ingresos['2y'] = h2o.load_model(f'{carpeta_models_automl}ingresos_avg/2y/StackedEnsemble_AllModels_0_AutoML_20190103_190356')
# modelos_ingresos['3y'] = h2o.load_model(f'{carpeta_model_automl}ingresos_avg/3y/StackedEnsemble_AllModels_0_AutoML_20190103_210454')

# modelos_costos['1y'] = h2o.load_model(f'{carpeta_models_automl}costos_avg/1y/StackedEnsemble_AllModels_0_AutoML_20190103_170925')
# modelos_costos['2y'] = h2o.load_model(f'{carpeta_models_automl}costos_avg/2y/StackedEnsemble_AllModels_0_AutoML_20190103_191041')
# modelos_costos['3y'] = h2o.load_model(f'{carpeta_models_automl}costos_avg/3y/StackedEnsemble_AllModels_0_AutoML_20190103_211140')

# modelos_fuga['1y'] = h2o.load_model(f'{carpeta_models_automl}fuga/y1/StackedEnsemble_AllModels_0_AutoML_20181211_175510')
# modelos_fuga['2y'] = h2o.load_model(f'{carpeta_models_automl}fuga/y2/StackedEnsemble_AllModels_0_AutoML_20181211_195722')
# modelos_fuga['3y'] = h2o.load_model(f'{carpeta_models_automl}fuga/y3/StackedEnsemble_AllModels_0_AutoML_20181211_215935')


# ### Generamos el h2o frame

# In[ ]:


h2o.__version__


# In[ ]:


mva_col_types = {col:'numeric' for col in df.columns if 'mva_' in col}
df_h2o = h2o.H2OFrame(df, destination_frame='dataframe_con_todo', column_types=mva_col_types)


# In[ ]:


df_h2o.head()


# ### Predecimos

# In[ ]:


df_h2o.columns


# In[ ]:


for key in modelos_ingresos:
    df_h2o['prediccion_percentil_ingresos_'+key] = modelos_ingresos[key].predict(df_h2o)
for key in modelos_costos:
    df_h2o['prediccion_percentil_costos_'+key] = modelos_costos[key].predict(df_h2o)
for key in modelos_fuga:
    df_h2o['prediccion_probabilidad_fuga_'+key] = modelos_fuga[key].predict(df_h2o)['p1']


# # Actualizar distribuciones 

# Dejar los percentiles entre 0 y 1

# In[ ]:


for key in modelos_ingresos:
    print(key)
    df_h2o[df_h2o['prediccion_percentil_ingresos_'+key] < 0, 'prediccion_percentil_ingresos_'+key] = 0
    df_h2o[df_h2o['prediccion_percentil_costos_'+key] < 0, 'prediccion_percentil_costos_'+key] = 0

    df_h2o[df_h2o['prediccion_percentil_ingresos_'+key] > 1, 'prediccion_percentil_ingresos_'+key] = 1
    df_h2o[df_h2o['prediccion_percentil_costos_'+key] > 1, 'prediccion_percentil_costos_'+key] = 1


# In[ ]:


df_h2o[[m for m in df_h2o.columns if 'perc' in m]].head(5)


# In[ ]:


df = df_h2o.as_data_frame()

df['Margen_t'] = 6*df['Margen_mva_1to6m'] + 6*df['Margen_mva_7to12m']
df['Margen_t-1'] = 12*df['Margen_mva_13to24m'] 

df['Costos_t'] = 6*df['Costos_mva_1to6m'] + 6*df['Costos_mva_7to12m']
df['Costos_t-1'] = 12*df['Costos_mva_13to24m'] 

df['Ingresos_t'] = 6*df['Ingresos_mva_1to6m'] + 6*df['Ingresos_mva_7to12m']
df['Ingresos_t-1'] = 12*df['Ingresos_mva_13to24m'] 

#ESTE TARDA MUCHO


# Le agregamos el percentil al que pertenecen:

# In[ ]:


columnas_a_transformar = ['Costos_t-1','Costos_t','Ingresos_t-1','Ingresos_t']

qtransformer = QuantileTransformer(100) #De nuevo seed?

for col in columnas_a_transformar:
    #df[col].fillna(0, inplace=True)
    qtransformer.fit(df[col].dropna().values.reshape(-1, 1))
    df['perc_' + col] = qtransformer.transform(df[col].values.reshape(-1, 1)).flatten()


df.head(2)


# In[ ]:


df.perc_Costos_t.agg(['min','max', 'std'])


# In[ ]:


def calcular_transformacion(df, ano1='Margen_t-1', ano2='Margen_t'):
    """
    Calcula el delta (cambio en $) por percentil entre dos distribuciones
    """
    perc_ano1 = 'perc_'+ano1
    perc_ano2 = 'perc_'+ano2
    
    delta = {}
    candidatos_ano1 = {}
    candidatos_ano2 = {}
    candidatos = {}
    num_candidatos = {}
    for i in np.linspace(0, 1, 101):
        i_round = np.round(i, 2)
        #Identifica filas que están en el grupo i (+- 4 percentiles)
        candidatos_ano1[i] = ((df[perc_ano1].round(decimals=2) >= (i_round - 0.02)) &  (df[perc_ano1].round(decimals = 2) <= (i_round + 0.02)))
                                     
        candidatos_ano2[i] = ((df[perc_ano2].round(decimals=2) >= (i_round - 0.02)) & (df[perc_ano2].round(decimals=2) <=(i_round + 0.02)))
                                     
        
        #Cumplieron estar en ese grupo de percentil dos años seguidos
        candidatos[i] = candidatos_ano1[i] & candidatos_ano2[i]
        num_candidatos[i] = candidatos[i].sum()
        if num_candidatos[i] == 0:
            delta[i] = 0
        if num_candidatos[i] > 0:
            #Delta promedio (en $) de margen de gente que no cambió (casi) su percentil
            delta[i] = (df[candidatos[i]][ano2] - df[candidatos[i]][ano1]).mean()
    return delta#, num_candidatos


# In[ ]:



@njit(parallel=True)
def par_nb_transformar_distribucion(costos_o_ingresos_ano1_array, predicted_percentiles: np.array, keys_delta: np.array, values_delta: np.array, num_anos=1):
    """
    Calcula el valor correspondiente al percentil p del siguiente año.
    ValueNextYear[percentil] = VPrevYear[percentil] + num_años*delta[percentil]
    
    p: percentil predicho de ingresos o costos (tomado de, p ej, 'prediccion_percentil_costos_1y')
    delta: diccionario de diferencias promedio por percentil
    costos_o_ingresos_ano1_array: array de 'Costos_t-1', 'Ingresos_t-1', 'Margen_t-1'
    """
    value_next_year = np.empty_like(costos_o_ingresos_ano1_array)
    
    for i in prange(predicted_percentiles.shape[0]):
        p = predicted_percentiles[i]
        value_prev_year = np.percentile(costos_o_ingresos_ano1_array[~np.isnan(costos_o_ingresos_ano1_array)], 100*p) #interpolation='midpoint'
        #Identifico el value delta más cercano a p
        nearest_perc_index = np.abs(keys_delta - p).argmin()
        value_next_year[i] = value_prev_year + num_anos*values_delta[nearest_perc_index]
    return value_next_year


# ## Percentiles Costos del Futuro

# In[ ]:


#Transformar percentiles predicho a pesos ($) (costos)
ano1 = 'Costos_t-1'
ano2 = 'Costos_t'

delta = calcular_transformacion(df, ano1=ano1, ano2=ano2)


# ## FOR LOOP para los tres costos con numba parallel

# In[ ]:


keys_delta, values_delta = np.array(list(delta.keys())), np.array(list(delta.values()))
costos_o_ingresos_ano1_array = df['Costos_t-1'].values #.copy()

opciones = ['prediccion_percentil_costos_1y', 'prediccion_percentil_costos_2y',
            'prediccion_percentil_costos_3y']
num=1
for m in opciones:
    predicted_percentiles = df[m].values #.copy()
    df['predicted_Costos_t+'+str(num)] = par_nb_transformar_distribucion(costos_o_ingresos_ano1_array, predicted_percentiles, 
                                                keys_delta, values_delta, num_anos=num)
    num+=1


# ## Percentiles Ingresos del Futuro

# In[ ]:


#Transformar percentile predicho a $pesos (ingresos)
ano1 = 'Ingresos_t-1'
ano2 = 'Ingresos_t'

delta = calcular_transformacion(df, ano1=ano1, ano2=ano2)


# ## FOR LOOP para los tres ingresos con numba parallel

# In[ ]:


keys_delta, values_delta = np.array(list(delta.keys())), np.array(list(delta.values()))
costos_o_ingresos_ano1_array = df['Ingresos_t-1'].values #.copy()

opciones = ['prediccion_percentil_ingresos_1y', 'prediccion_percentil_ingresos_2y',
            'prediccion_percentil_ingresos_3y']
num=1
for m in opciones:
    predicted_percentiles = df[m].values #.copy()
    df['predicted_Ingresos_t+'+str(num)] = par_nb_transformar_distribucion(costos_o_ingresos_ano1_array, predicted_percentiles, 
                                                keys_delta, values_delta, num_anos=num)
    num+=1


# In[ ]:


pd.set_option('display.max_columns', 500)
df.head(2)


# In[ ]:


#Export y subir a BigQuery
# ES necesario pasar por H2o?
# 
# h2o_df = h2o.H2OFrame(df)
csv_name = f'{carpeta_ltv}Output/output_modelos_avg_{nombre_fancy}.csv'
# h2o.export_file(h2o_df,
#                 csv_name,
#                 force=True, parts=1)
df.to_csv(csv_name)
ltv_table_name = f"LTV.PREDICCION_MODULOS_LTV_{nombre_fancy}"
# !bq --location=US load --autodetect --source_format=CSV {ltv_table_name} "{csv_name}"


# ### Creamos la tabla de LTV:

# In[ ]:


#Liberamos memoria de los procesos de h2o
h2o.remove_all()
h2o.shutdown()
h2o.init(port=54321, min_mem_size="2g", max_mem_size="4g")


# In[ ]:


df_ltv = df[[m for m in df.columns if 'predicted' in m or 'probabilidad' in m or 'id_' in m or 'prediccion_percentil' in m]].copy()
df_ltv.head()


# In[ ]:


df_ltv = df[[m for m in df.columns if 'predicted' in m or 'probabilidad' in m or 'id_' in m or 'prediccion_percentil' in m]].copy()
df_ltv.set_index('id_titular', inplace = True)
#df_ltv.drop_duplicates(subset='id_titular', inplace = True)
df_ltv = df_ltv[~df_ltv.index.duplicated(keep='first')]

df_ltv.head(3)


# In[ ]:


df_ltv.describe()


# In[ ]:


def calculo_ltv(df, cols_margen, cols_fuga, r=0.1):
    """
    Toma un df con columnas predichas de margen, columnas predichas de fuga
    y calcula el LTV con una tasa de descuento r
    
    df: dataframe con la info necesaria: columnas de margen y columnas de fuga.
    cols_margen: lista de las columnas usadas de margen.
    cols_fuga: lista de las columnas usadas de probabilidad de fuga. Deben ir en el 
                mismo orden que cols_margen.
    r: tasa de descuento.
    
    returns:
        Devuelve un datatrace con el mismo índice que df, con el LTV predicho para cada ID.
    
    """
    def desc(r, periodo):
        return 1/(1+r)**(periodo+1)
    
    LTV_predicted = pd.DataFrame(0,index = df.index, columns = ['LTV_predicted'])
    
    # Importante, suponemos que vienen ordenadas las listas:
    
    for i in np.arange(0,len(cols_margen)):
        LTV_predicted['LTV_predicted'] += desc(r, i)*df[cols_margen[i]]*(1 - df[cols_fuga[i]])
    
    return LTV_predicted


def calculo_ltv_fuga0(df, cols_margen, r=0.1):
    """
    Toma un df con columnas predichas de margen, columnas predichas de fuga
    y calcula el LTV con una tasa de descuento r
    
    df: dataframe con la info necesaria: columnas de margen y columnas de fuga.
    cols_margen: lista de las columnas usadas de margen.
    cols_fuga: lista de las columnas usadas de probabilidad de fuga. Deben ir en el 
                mismo orden que cols_margen.
    r: tasa de descuento.
    
    returns:
        Devuelve un datatrace con el mismo índice que df, con el LTV predicho para cada ID.
    
    """
    def desc(r, periodo):
        return 1/(1+r)**(periodo+1)
    
    LTV_predicted = pd.DataFrame(0,index = df.index, columns = ['LTV_predicted'])
    
    # Importante, suponemos que vienen ordenadas las listas:
    for i in np.arange(0, len(cols_margen)):
        LTV_predicted['LTV_predicted'] += desc(r, i)*df[cols_margen[i]]
    return LTV_predicted


# In[ ]:


df_ltv['predicted_Margen_t+1'] = df_ltv['predicted_Ingresos_t+1'] - df_ltv['predicted_Costos_t+1']
df_ltv['predicted_Margen_t+2'] = df_ltv['predicted_Ingresos_t+2'] - df_ltv['predicted_Costos_t+2']
df_ltv['predicted_Margen_t+3'] = df_ltv['predicted_Ingresos_t+3'] - df_ltv['predicted_Costos_t+3']

cols_margen = ['predicted_Margen_t+1', 'predicted_Margen_t+2', 'predicted_Margen_t+3']
cols_fuga = ['prediccion_probabilidad_fuga_1y', 'prediccion_probabilidad_fuga_2y', 'prediccion_probabilidad_fuga_3y']

ltv = calculo_ltv(df_ltv, cols_margen, cols_fuga)
ltv_fuga0 = calculo_ltv_fuga0(df_ltv, cols_margen)


# In[ ]:


quant_dict_nueva = {}
for column in [m for m in df_ltv.columns if 'predicted_Margen' in m]:
    df_ltv[column + '_perc'] = np.nan
    quant_dict_nueva[column + '_perc'] = QuantileTransformer(100) #seed?
    df_ltv.loc[~df_ltv[column].isna(), column + '_perc'] = (quant_dict_nueva[column + '_perc'].fit_transform(
        (df_ltv.loc[~df_ltv[column].isna()][column].values).reshape([-1, 1])).ravel())

df_ltv.head(3)


# In[ ]:


df_ltv['ltv_predicted'] = ltv['LTV_predicted']
df_ltv['ltv_fuga0_predicted'] = ltv_fuga0['LTV_predicted']
#LTV en percentiles

quant_dict_ltv = {}
for column in [m for m in df_ltv.columns if 'ltv' in m]:
    df_ltv[column + '_perc'] = np.nan
    quant_dict_nueva[column + '_perc'] = QuantileTransformer(100)
    df_ltv.loc[~df_ltv[column].isna(), column + '_perc'] = (quant_dict_nueva[column + '_perc'].fit_transform(
        (df_ltv.loc[~df_ltv[column].isna()][column].values).reshape([-1, 1])).ravel())

df_ltv.head(3)


# In[ ]:


df_ltv.reset_index(inplace=True)
df_ltv.head(3)


# In[ ]:


df_ltv.describe()


# In[ ]:


import requests

def slack_code_ready(message, username, webhook_num):
    """
    Sends a message to a username on slack
    """
    webhook = "https://hooks.slack.com/services/" + webhook_num
    message = "<@U{0}> {1}".format(username, message)
    payload = {"text": message}
    headers = {'Content-type': 'application/json'}

    return requests.post(url=webhook, data=json.dumps(payload), headers=headers)

username =  "6FK1CFBJ" 
webhook = "T6FJW727J/BBY4DLYGP/whY7uk7I6DRJ53VWUJrSYtXp"


#slack_code_ready(f"""Ya terminó la parte de df_ltv. Ahora paso a df_p""",
 #                username, webhook)


# ## Le agregamos el cluster al que pertenece:

# In[ ]:


df_p = pd.DataFrame()
df_p['promedio_ingresos'] = df_ltv[['predicted_Ingresos_t+3','predicted_Ingresos_t+2','predicted_Ingresos_t+1']].mean(axis =1)
df_p['promedio_costos'] = df_ltv[['predicted_Costos_t+3','predicted_Costos_t+2','predicted_Costos_t+1']].mean(axis =1)
df_p['promedio_fuga'] = df_ltv[['prediccion_probabilidad_fuga_1y','prediccion_probabilidad_fuga_2y','prediccion_probabilidad_fuga_3y']].mean(axis =1)
df_p['ltv_predicted'] = df_ltv[['ltv_predicted']]
df_p['ltv_fuga0_predicted'] = df_ltv[['ltv_fuga0_predicted']]

df_p['id_titular'] = df_ltv[['id_titular']]

df_p.head(2)


# In[ ]:


df_p.to_csv("../../df_p.csv")


# In[ ]:


import gc
h2o.cluster().shutdown()
del df_p
time.sleep(15)
gc.collect()
time.sleep(25)


# In[ ]:


# De nuevo para asegurarse de que la memoria esté disponible
h2o.init(port=54321, min_mem_size="2g", max_mem_size="4g")
h2o.remove_all()

model = _load_h2o_model_compatible(
    f'{carpeta_ltv}Output/modelos_categorias_clustering_predicting_average/kmeans_5',
    model_format=ltv_h2o_model_format,
    model_label="kmeans_5",
)
df_ltv.reset_index(inplace = True)


# In[ ]:


df_p_h2o = h2o.import_file(path='../../df_p.csv', destination_frame="dp_p")
cluster_predicted = model.predict(df_p_h2o).as_data_frame() + 1

diccionario_clusters_LTV = {5: 'muy alto', 1: 'alto', 3:'medio', 2:'bajo',
                            4:'muy bajo (alta fuga)'}


# In[ ]:


df_ltv['categoria_ltv'] = cluster_predicted.replace(diccionario_clusters_LTV)
df_ltv.head(2)


# In[ ]:


nombre_fancy


# In[ ]:


csv_name = f'{carpeta_ltv}Output/prediccion_ltv_{nombre_fancy}.csv'
#df_ltv.to_pickle(f'/home/ubuntu/spike/LTV3/Output/prediccion_ltv_con_categoria_{fancy_name}.pkl')
df_ltv.to_csv(csv_name, index = False)
repo_dir = _resolve_repo_dir()
ges_pred_input_dir = Path(_normalize_local_path(
    os.getenv("GES_PREDICCION_INPUT_DIR", str(repo_dir / "input" / "prediccion"))
))
ges_pred_input_dir.mkdir(parents=True, exist_ok=True)
target_prediccion_ltv = ges_pred_input_dir / Path(csv_name).name
if Path(csv_name).resolve() != target_prediccion_ltv.resolve():
    shutil.copy2(csv_name, target_prediccion_ltv)
    print(f"Archivo prediccion_ltv copiado a {target_prediccion_ltv}")
else:
    print(f"Archivo prediccion_ltv ya esta en {target_prediccion_ltv}")

ltv_table_name = f"LTV.PREDICCION_LTV_CATLTV_{nombre_fancy}"
#!bq --location=US load --autodetect --replace --source_format=CSV {ltv_table_name} "{csv_name}"


# In[ ]:


df_ltv.head()


# In[ ]:


end = time.time()

tiempo_transcurrido = (end - start)/60


# In[ ]:


## Acá se hace check de Great Expectations final (output)
expectations_outputs_path = f'{carpeta_ltv}Greats_expectations_LTV/ltv_expectations_marzo_output.json'
my_expectations_config = _load_json_if_exists(expectations_outputs_path)
if ge is not None and my_expectations_config is not None:
    my_df = ge.read_csv(
        f'{carpeta_ltv}Output/prediccion_ltv_{nombre_fancy}.csv',
        expectations_config=my_expectations_config,
    )
    resultado_output = my_df.validate(result_format='BASIC', only_return_failures=True)

    with open(
        f'{carpeta_ltv}Greats_expectations_LTV/resultado_GE_{nombre_fancy}_outputs.json',
        'w'
    ) as outfile:
        json.dump(resultado_output, outfile)
    print("Guardado!")
else:
    print(
        f"Great Expectations final omitido. "
        f"ge_disponible={ge is not None}, config_encontrada={my_expectations_config is not None}, "
        f"path={expectations_outputs_path}"
    )
print("Guardado!")

#assert resultado_output['success']


# In[ ]:


import requests
def slack_code_ready(message, username, webhook_num):
    """
    Sends a message to a username on slack
    """
    webhook = "https://hooks.slack.com/services/" + webhook_num
    message = "<@U{0}> {1}".format(username, message)
    payload = {"text": message}
    headers = {'Content-type': 'application/json'}

    return requests.post(url=webhook, data=json.dumps(payload), headers=headers)

username =  "J2R9275K" #Emi. CD: "6FK1CFBJ"
webhook = "T6FJW727J/BBY4DLYGP/whY7uk7I6DRJ53VWUJrSYtXp"


#slack_code_ready(f"""Terminó la predicción de LTV para {nombre_fancy}.
 #                 Bien, no? Anda a LTV.PREDICCION_LTV_CATLTV_{nombre_fancy}
  #                  a cachar si se ve gonito.
  #                  Se demoró {tiempo_transcurrido} mins \X|C/ """,
  #               username, webhook)


# In[ ]:


print(tiempo_transcurrido)
print(tiempo_transcurrido/60)


# In[ ]:


for col in df_ltv.columns:
    if df_ltv[col].isna().any():
        print(f'{col} tiene {df_ltv[col].isna().sum()} vacios')


# In[ ]:


data_ltv.shape, df_ltv.shape


# In[ ]:


df_ltv.head()


# In[ ]:


df_ltv.describe()


# In[ ]:


df_ltv.ltv_fuga0_predicted.describe()

