#!/usr/bin/env python
# coding: utf-8

# ## Preproceso para la predicción mensual
# 

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import collections
from IPython.display import display
import seaborn as sns
import re
from tqdm import tqdm_notebook as progress_bar
from difflib import SequenceMatcher
from itertools import combinations
import os
from pathlib import Path

# sys.path.append('../../src')
# from comparar_distribuciones import *

# sys.path.append("../../../spike")
# import SpikePy as sp

def _run_ipython_magic(name, value):
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        if ip is not None:
            ip.run_line_magic(name, value)
    except Exception:
        pass


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


def _read_sql_table_as_pandas(spark_session, table_name):
    print(f"Leyendo {table_name} desde Spark/Databricks...")
    # Si el volumen crece, considerar filtrar por periodo antes de pasar a pandas.
    return spark_session.table(table_name).toPandas()


def _require_columns(df, required_columns, table_name):
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"La tabla {table_name} no contiene columnas requeridas: {missing}"
        )


_run_ipython_magic('matplotlib', 'inline')
_run_ipython_magic('load_ext', 'autoreload')
_run_ipython_magic('autoreload', '2')

plt.style.use('fivethirtyeight')
#Usar kernel spikelabs_env_3


# ## Setear periodo 
# Debiese ser el único input

# In[2]:


#periodo a predecir 'mes(3letras)año'
hoy =  pd.to_datetime("today")
periodo_prediccion_param = _get_periodo_prediccion_param()
if periodo_prediccion_param is None:
    fecha_prediccion =  hoy - pd.DateOffset(months=1, day=1)
    periodo_prediccion = fecha_prediccion.year*100 + fecha_prediccion.month
else:
    periodo_prediccion = periodo_prediccion_param
    fecha_prediccion = pd.to_datetime(str(periodo_prediccion), format="%Y%m")

pre_ges = 0  # hubo cambio ges 5 meses antes del periodo de cierre 0 -> si no hubo cambio | 1 -> hubo cambio
hay_ges = 0  # hay /habrá cambio ges en los 5 meses siguientes 0 -> si no habrá camnbio | 1 -> habrá cambio


num_a_mes = {
    1: 'ene', 2: 'feb', 3: 'mar',  4: 'abr',  5: 'may',  6: 'jun',
    7: 'jul', 8: 'ago', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dic'
}
periodo = f"{num_a_mes[fecha_prediccion.month]}{fecha_prediccion:%y}"


# In[3]:


spark_session = _get_spark_session()


# In[4]:


if spark_session is None:
    raise RuntimeError(
        "Este script requiere una sesion Spark activa de Databricks."
    )
print("Leyendo EST.P_DDV_EST.JC_GES_PRED desde Spark/Databricks...")
# Evita evaluar el CAST invalido del view sobre fld_pertermino y lo reconstruye de forma segura.
from pyspark.sql import functions as F
df_ges_spark = spark_session.table("EST.P_DDV_EST.JC_GES_PRED").drop("fld_pertermino")
if "fld_termino" not in df_ges_spark.columns:
    raise RuntimeError("No se encontro columna fld_termino para reconstruir fld_pertermino.")

# Casteo seguro para columnas numericas que pueden venir tipadas como string.
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
# Si el volumen crece, considerar filtrar por periodo antes de pasar a pandas.
df_ges = df_ges_spark.toPandas()


# In[5]:


repo_dir = _resolve_repo_dir()
default_storage_root = str(repo_dir)
inputs = Path(_normalize_local_path(
    os.getenv("GES_PREPROCESO_INPUT_DIR", str(repo_dir / "input" / "preproceso"))
))
outputs = Path(_normalize_local_path(
    os.getenv("GES_OUTPUT_DIR", str(repo_dir / "output"))
))
in_pred = Path(_normalize_local_path(
    os.getenv("GES_PREDICCION_INPUT_DIR", str(repo_dir / "input" / "prediccion"))
))

inputs.mkdir(parents=True, exist_ok=True)
outputs.mkdir(parents=True, exist_ok=True)
in_pred.mkdir(parents=True, exist_ok=True)

aux_schema = os.getenv("GES_AUX_SQL_SCHEMA", "EST.P_DDV_EST").strip() or "EST.P_DDV_EST"
table_prm_categoria = os.getenv(
    "GES_SQL_TABLE_PRM_CATEGORIA", f"{aux_schema}.PRM_CATEGORIA"
)
table_division_regiones = os.getenv(
    "GES_SQL_TABLE_DIVISION_REGIONES", f"{aux_schema}.DIVISION_REGIONES"
)
table_cod_comuna = os.getenv(
    "GES_SQL_TABLE_COD_COMUNA", f"{aux_schema}.COD_COMUNA"
)
table_nse_y_pobreza = os.getenv(
    "GES_SQL_TABLE_NSE_Y_POBREZA", f"{aux_schema}.NSE_Y_POBREZA"
)

nombre_archivo = f'{hoy:%y.%m.%d}_GES_{fecha_prediccion:%Y%m}_{num_a_mes[fecha_prediccion.month]}{fecha_prediccion:%y}.gz'
ges_file = inputs / nombre_archivo
df_ges.to_csv(ges_file, index=False, compression='gzip')


# In[6]:


#shift periodo
p = {'ene':'01','feb':'02','mar':'03','abr':'04','may':'05','jun':'06',
     'jul':'07','ago':'08','sep':'09','oct':'10','nov':'11','dic':'12'}

per = p[periodo[:3]]

t = str(int(per)%12+1)

if len(t) == 1:
    next_per = '0'+t
else:
    next_per = t        

if periodo[:3] == 'dic':
    agno = str(int(periodo[3:])+1)
else:
    agno = periodo[3:]


# In[7]:


#todos los archivos necesarios
# Los directorios ya fueron definidos de forma portable para Databricks.

# print('Buscando los datos de ges del periodo...')

# temp_ges = []

# for root, subdirs, files in os.walk(inputs):
#      for filename in files:
#             if 'ges' in filename.lower():
#                 if periodo in filename:
#                     temp_ges.append(filename)
# try:
#     print('Los archivos encontrados son:...')

#     print(*temp_ges,sep='\n')
    
#     print(f'Se utilizara el archivo {temp_ges[0]} para el preproceso')
    
#     ges = temp_ges[0]
    
# except IndexError:
#     print('No hay archivos en el periodo, buscando en todos los periodos...')

#     for root, subdirs, files in os.walk('input'):
#          for filename in files:
#                 if 'ges' in filename.lower():
#                         temp_ges.append(filename)
#     print('Los archivos son:')
    
#     print(*temp_ges,sep='\n')
    
#     print('Modifique más abajo el archivo que desea utilizar')

ges = str(ges_file)


# In[8]:


print('Buscando los datos de ltv del periodo...')

temp_ltv = []

for root, subdirs, files in os.walk(str(inputs)):
     for filename in files:
            if 'ltv' in filename.lower():
                if periodo in filename:
                    temp_ltv.append(filename)
try:
    print('Los archivos encontrados son:...')

    print(*temp_ltv,sep='\n')
    
    print(f'Se utilizara el archivo {temp_ltv[0]} para el preproceso')
    
    ltv = temp_ltv[0]
    
except IndexError:
    print('No hay archivos en el periodo, buscando en todos los periodos...')

    for root, subdirs, files in os.walk(str(inputs.parent)):
         for filename in files:
                if 'ges' in filename.lower():
                        temp_ltv.append(filename)
    print('Los archivos son:')
    
    print(*temp_ltv,sep='\n')
    
    print('Modifique más abajo el archivo que desea utilizar')


# In[9]:


#datos auxiliares (migrados desde Excel a tablas SQL)


# In[10]:


#cargamos datos ltv
df_ltv =  pd.read_csv(inputs / ltv, sep=",", compression='gzip').rename(columns = lambda x: x.lower())
df_ltv['periodo'] = pd.to_datetime(df_ltv['periodo'], format='%Y%m')
# df_ltv.head()


# In[11]:


df_ltv.head()


# In[12]:


#cargamos datos ges, solo algunas columnas
cols_to_use = ['id_titular','periodo','fld_pertermino','n_ges_mva_2m',
               'preexistencia_afi','preexistencia_car','gls_isapreant']

# df_ges = df pd.read_csv(inputs+ges, compression='zip') #usecols=cols_to_use,delimiter=';',


# In[13]:


#generamos la variable huerfano y formateamos
df_ges['huerfano'] = ((df_ges['fld_pertermino'] < df_ges['periodo']) & (df_ges['fld_pertermino'] != 190001))
df_ges['periodo'] = pd.to_datetime(df_ges['periodo'], format='%Y%m')  #
# df_ges['periodo'] = (df_ges['periodo'] + pd.Timedelta('32 day')).apply(lambda f: f.replace(day=1))
df_ges = df_ges.rename(columns=lambda x: x.lower())
df_ges = df_ges.dropna(subset=['fld_pertermino'])
df_ges = df_ges.drop(['fld_pertermino'], axis=1)


# In[14]:


df_ges.head()


# In[15]:


#merge
df = pd.merge(df_ltv, df_ges, how='left', on=['periodo', 'id_titular'])


# #### Shiftear periodos Fuga y VN

# In[16]:


fugas = df.tipo_transac.isin(['D2', 'D3'])
index_fugas = fugas[fugas].index
periodos_fuga = df.loc[index_fugas, 'periodo']
df.loc[index_fugas, 'periodo'] = (periodos_fuga - pd.Timedelta('1 day')).apply(lambda f: f.replace(day=1))
df = df[df.renta_imponible >= 0]
df = df.drop(['anno', 'mes', 'fechaingreso'], axis=1)


# ### Datos preferente

# In[17]:


def cat_linea_plan(x):
    for c, lst in categorias.items():
        if x in lst:
            return c
        
alto = ['CLINICA ALEMANA', 'CLINICA LAS CONDES', 'LOS ANDES', 'SAN CARLOS', 'ALEMANA-SC-LA-CSM / CLC', 
        'SAN CARLOS - LOS ANDES', 'SAN CARLOS - UC - LOS ANDES', 'BALTICO', 'SAN CARLOS - STA. MARIA - LOS ANDES',
        'ADRIATICO', 'SAN CARLOS-UC']
medio = ['SANTA MARIA', 'HOSP. CLINICO UNIVERSIDAD CATOLICA', 'CLINICA UNIVERSIDAD CATOLICA', 'TABANCURA', 
         'MEDITERRANEO', 'SAN CARLOS-UC', 'HOSP.CLINICO UNIVERSIDAD DE CHILE', 'INDISA INSTITUCIONAL', 'CLINICA INDISA']
bajo = ['DAVILA', 'BICENTENARIO-VESPUCIO-LAS LILAS',  
        'UC SJ 110 - BICENTENARIO - VESPUCIO', 'CORDILL.-BICENT.-AVANSALUD-SERVET', 'BICENTENARIO-TABANCURA-DAVILA', 
        'CITY COLMENA', 'BICENTENARIO-VESPUCIO', 'CLINICA CORDILLERA - CLINICA DAVILA', 
        'UC SJ - BICENTENARIO - VESPUCIO']
le = ['LIBRE ELECCION']

categorias = {'alto': alto, 'medio': medio, 'bajo': bajo, 'le': le}


# In[18]:


#demora 1 min
categoria = _read_sql_table_as_pandas(spark_session, table_prm_categoria)
categoria = categoria.rename(columns=lambda x: str(x).strip().lower())
_require_columns(categoria, ['cod_categoria', 'preferente'], table_prm_categoria)
df = pd.merge(df, categoria[['cod_categoria', 'preferente']], how='left', 
                 left_on='categoria_cod', right_on='cod_categoria')
df['cat_linea_plan'] = df.preferente.apply(cat_linea_plan)

titulares = df.id_titular.unique()


# In[19]:


replace_pref = {'UC ': 'UNIVERSIDAD CATOLICA',
                'CSM': 'STA. MARIA',
                'CLC': 'CLINICA LAS CONDES',
                'BICENT.': 'BICENTENARIO',
                'CORDILL.': 'CLINICA CORDILLERA'}


# In[20]:


def limpiar_preferente(x):
    y = x
    for old, new in replace_pref.items():
        try:
            y = y.replace(old, new)
        except:
            pass
    return y


# In[21]:


df.preferente = df.preferente.apply(limpiar_preferente)


# ### Datos Democráficos Extras

# In[22]:


# demora 
division_reg = _read_sql_table_as_pandas(spark_session, table_division_regiones)
division_reg = division_reg.rename(columns=lambda x: str(x).strip())
if 'Unnamed: 3' in division_reg.columns and 'norte_centro_sur' not in division_reg.columns:
    division_reg = division_reg[19:].rename(columns={'Unnamed: 3': 'norte_centro_sur'})
division_reg = division_reg.rename(
    columns={
        'cod_region': 'COD_REGION',
        'gls_web_region': 'GLS_WEB_REGION',
        'NORTE_CENTRO_SUR': 'norte_centro_sur',
    }
)
_require_columns(
    division_reg,
    ['COD_REGION', 'GLS_WEB_REGION', 'norte_centro_sur'],
    table_division_regiones,
)
division_reg = division_reg[['COD_REGION', 'GLS_WEB_REGION', 'norte_centro_sur']]
division_reg['COD_REGION'] = division_reg.COD_REGION.astype('int')

cod_comuna = _read_sql_table_as_pandas(spark_session, table_cod_comuna)
cod_comuna = cod_comuna.rename(columns=lambda x: str(x).strip().lower())
_require_columns(cod_comuna, ['con_comuna_gls', 'cod_comuna'], table_cod_comuna)

nse_y_pobreza = _read_sql_table_as_pandas(spark_session, table_nse_y_pobreza)
nse_y_pobreza = nse_y_pobreza.rename(columns=lambda x: str(x).strip())
if 'comuna' in nse_y_pobreza.columns and 'COMUNA' not in nse_y_pobreza.columns:
    nse_y_pobreza = nse_y_pobreza.rename(columns={'comuna': 'COMUNA'})
_require_columns(nse_y_pobreza, ['COMUNA'], table_nse_y_pobreza)

nse_y_pobreza = pd.merge(nse_y_pobreza, cod_comuna, how='inner',
                        left_on='COMUNA', right_on='con_comuna_gls')
nse_y_pobreza['COD_REGION'] = nse_y_pobreza.cod_comuna.astype('str').str[:-3].astype('int')
full_comuna = pd.merge(nse_y_pobreza, division_reg, how='left',
                      on='COD_REGION')

#Asumiendo que tu base se llama df_res y tiene la variable `comuna_cod`
df = pd.merge(df, full_comuna, how='left',
                      left_on='comuna_cod', right_on='cod_comuna')


# ### Drop y limpieza 

# In[23]:


df = df.rename(columns={'categoria_gls': 'categor__a_gls'})


# In[24]:


drop = ['region_cod', 'comuna_cod', 'fecha_nacimiento', 'cod_sucursal', 'centrocostos_cod', 
        'detalle_producto', 'cod_categoria', 'con_comuna_gls', 'cod_comuna', 'COD_REGION',
        'GLS_WEB_REGION', 'comuna_gls']
df = df.drop(drop, axis=1)


# In[25]:


cols_strip = ['region_gls', 'sucursal_gls', 'categor__a_gls', 'serie', 'tipo_plan', 'tipo_producto', 'actividad']
for col in cols_strip:
    df.loc[:, col] = df[col].str.rstrip()


# ## Generar Features

# ### Variable categórica de ges y pre-ges en ventana de tiempo
# Setear en función de si habrá annuncio de alza por precio ges y (pre_ges)
# y apertura de cartera por cambio (ges) en una ventana de tiempo (meses)

# In[26]:


tamaño_ventana = 5 #meses a futuro
df[f'hay_preges_{tamaño_ventana}m'] = pre_ges
df[f'hay_ges_{tamaño_ventana}m'] = hay_ges


# ### Variable categórica de Fuga Voluntaria en ventana de tiempo
# variable de entrenamient

# In[27]:


df['fuga_5m'] = np.NaN


# ### Antigedad < 12 y GES = 1

# In[28]:


df['ant_m12_y_ges'] = (df.antiguedad <= 12 - tamaño_ventana)


# ### Tuvo CIE complejo, tuvo gasto licencia

# In[29]:


#propagar variables 
var = 'cie_complejo'
var_new = 'hubo_cie_complejo'
df_aux = df[['id_titular', 'periodo', var]]
df_aux = pd.pivot_table(df_aux, index='periodo', values=var, 
                               columns='id_titular', aggfunc=np.max)
df_aux = df_aux.sort_index().fillna(0)
df_aux = df_aux.rolling(window=100, min_periods=1).max()
df_aux = df_aux.stack().reset_index().rename(columns={0: var_new})
df = pd.merge(df, df_aux, how='left', on=['id_titular', 'periodo'])


# In[30]:


df['hay_gasto_licencia'] = ((df.gasto_licencias != 0) | (df.gasto_licencias_excl != 0)) + 0

var = 'hay_gasto_licencia'
var_new = 'hubo_licencia'
df_aux = df[['id_titular', 'periodo', var]]
df_aux = pd.pivot_table(df_aux, index='periodo', values=var, 
                               columns='id_titular', aggfunc=np.max)
df_aux = df_aux.sort_index().fillna(0)
df_aux = df_aux.rolling(window=100, min_periods=1).max()
df_aux = df_aux.stack().reset_index().rename(columns={0: var_new})
datos = pd.merge(df, df_aux, how='left', on=['id_titular', 'periodo'])


# In[31]:


f = f'ready_to_pred_{periodo}.csv'
#df.to_csv(outputs + f) dejo solamente la copia para la predicción, no tiene sentido estar duplicando este archivo
df.to_csv(in_pred / f) #apunta la carperta de inputs para la predicción


# In[32]:


df.shape


# In[33]:


print(df.n_ges_mva_2m.unique())
df.n_ges_mva_2m.head()


# In[ ]:




