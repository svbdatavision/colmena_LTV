#!/usr/bin/env python
# coding: utf-8

# ## Predección de fuga durante periodo GES

# In[1]:


# usar kernel spikelabs_env_3
import pandas as pd
#import feather
from itertools import product
import numpy as np
import h2o
import seaborn as sns
import os
from pathlib import Path
from scipy.stats import wasserstein_distance as w_dist
import warnings
warnings.filterwarnings('ignore')
import json
import tqdm

import getpass as gp

import sys
try:
    from snowflake.connector.pandas_tools import pd_writer
except Exception:
    pd_writer = None

try:
    from sqlalchemy import create_engine
except Exception:
    create_engine = None

try:
    from IPython.core.display import display, HTML
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
        return gp.getpass(prompt=prompt_label)

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


def _resolve_period_file(base_dir, period_tag, contains_token):
    for _, _, files in os.walk(str(base_dir)):
        for file_name in files:
            if period_tag in file_name and contains_token in file_name.lower():
                return file_name
    return None


_run_ipython_magic('load_ext', 'autoreload')
_run_ipython_magic('autoreload', '2')

#Cambie la ruta al PD; versión anterior:  estudio/data/
spark_session = _get_spark_session()
default_storage_root = "/dbfs/mnt/modelos" if spark_session is not None else "/mnt/disks/modelos"
disco = _normalize_local_path(os.getenv("GES_STORAGE_ROOT", default_storage_root)).rstrip("/")

script_graficos = Path(disco) / "Proyecto_GES" / "script_graficos"
if script_graficos.exists():
    sys.path.append(str(script_graficos))
try:
    from plot_helpers import compare_cont_dists
    from plot_helpers import compare_categorical_dists
except Exception:
    compare_cont_dists = None
    compare_categorical_dists = None


if display is not None and HTML is not None:
    display(HTML("<style>.container { width:95% !important; }</style>"))
# para poder ver todas las columnas
pd.set_option('display.max_columns',2000)


# In[2]:


snow_account = os.getenv("SNOWFLAKE_ACCOUNT", "isapre_colmena.us-east-1")


# In[3]:


# las demas rutas dependen de esto, por lo que es necesario verificar
#assert os.getcwd() == '/estudio/data/Proyecto_GES/Prediccion'


periodo_prediccion_param = _get_periodo_prediccion_param()
if periodo_prediccion_param is None:
    hoy = pd.to_datetime("today")
    fecha_prediccion = hoy - pd.DateOffset(months=1, day=1)
    periodo_prediccion = fecha_prediccion.year*100 + fecha_prediccion.month
else:
    periodo_prediccion = periodo_prediccion_param
    fecha_prediccion = pd.to_datetime(str(periodo_prediccion), format="%Y%m")

num_a_mes = {
    1: 'ene', 2: 'feb', 3: 'mar',  4: 'abr',  5: 'may',  6: 'jun',
    7: 'jul', 8: 'ago', 9: 'sep', 10: 'oct', 11: 'nov', 12: 'dic'
}
periodo = f"{num_a_mes[fecha_prediccion.month]}{fecha_prediccion:%y}"  # periodo en formato abreviado usado por nombres de archivos
renta_tope = 84.3                                               #modifcar al tope en el periodo a predecir

#datos del training
train_file = 'retrain.csv'
#carpeta de inputs
repo_dir = Path(__file__).resolve().parent
inputs = Path(_normalize_local_path(
    os.getenv("GES_PREDICCION_INPUT_DIR", str(repo_dir / "input" / "prediccion"))
))

#especial para el archivo original sin pre-proceso
#aca va ltv del periodo (la tabla no predicciones) y base ges
input2 = Path(_normalize_local_path(
    os.getenv("GES_PREPROCESO_INPUT_DIR", f"{disco}/Proyecto_GES/Prediccion/input/preproceso")
))

#carpeta de output
outputs = Path(_normalize_local_path(
    os.getenv("GES_OUTPUT_DIR", str(repo_dir / "output"))
))

#path total desde root, para el h2o load
#aca va ltv predicciones del periodo y el archivo que se genera en el notebook de preproceso
input_total = Path(_normalize_local_path(
    os.getenv("GES_PREDICCION_SCAN_DIR", str(inputs))
))

inputs.mkdir(parents=True, exist_ok=True)
outputs.mkdir(parents=True, exist_ok=True)

#datos del periodo a predecir
#ges_file = f'ready_to_pred_{periodo}.csv'
#datos del ltv
#Automatizar la selección de este archivo.
#ltv_file = f'02. prediccion_ltv_con_categoria_{periodo}.csv'
ges_file = _resolve_period_file(input_total, periodo, "ready")
ltv_file = _resolve_period_file(input_total, periodo, "prediccion_ltv")



#ges sin prep  20.04.17_GES_mar20
#ges_r_file = f'2021.03.19_GES_{periodo}.csv.zip'

ges_r_file = _resolve_period_file(input2, periodo, "ges")

if ges_file is None:
    raise FileNotFoundError(
        f"No se encontro archivo ready_to_pred para periodo {periodo} en {input_total}"
    )
if ltv_file is None:
    raise FileNotFoundError(
        f"No se encontro archivo prediccion_ltv para periodo {periodo} en {input_total}"
    )
            
print(f'archivo ges: {ges_file}')
print(f'archivo ltv: {ltv_file}')
print(f'archivo ges preproceso: {ges_r_file}')
  


# ## **Carga de datos:**
# Se cargan los datos del ltv del mes, del ges del mes y del train del modelo ges para comparar y validar

# In[4]:


#cargamos datos ges
df_ges = pd.read_csv(inputs / ges_file, index_col=0)


# In[5]:


ges_file


# In[6]:


#chequeamos que las columnas no hayan variado

categs = ['sucursal_gls', 'region_gls', 'actividad', 'tipo_trabajador', 'tipo_producto', 
          'tipo_plan', 'linea_plan', 'COMUNA', 'sexo', 'hay_ges_5m', 'hay_preges_5m',
          'ant_m12_y_ges' , 'norte_centro_sur', 'preferente', 'cat_linea_plan', 'huerfano',
          'gls_isapreant', 'preexistencia_afi', 'preexistencia_car']


numeric = ['antiguedad', 'edad', 'costo_final', 'precio_base', 'factor_riesgo',
           'costo_total', 'benef_adic', 'excesos', 'excedentes',
           'gasto_ambulatorios', 'gasto_hospitalarios', 'gasto_hospitalarios_excl',
           'gasto_ges', 'gasto_caec', 'gasto_pharma', 'recuperacion_gastos',
           'iva_cotizaciones', 'iva_recuperado', 'gasto_licencias',
           'gasto_licencias_excl', 'cie_complejo', 'prestacion_amb_compleja',
           'gasto_parto', 'avg_income', 'pobreza_multi_ptje']

total_cols = categs + numeric

c=0

for col in total_cols:
    try:
        df_ges[col].shape
        c+=1
    except KeyError:
        print(f'La columna {col} no existe')

if c == len(total_cols):
    print('Todas las columnas existen')


# In[7]:


#botamos duplicados
n_unique = df_ges.id_titular.nunique()
df_ges.drop_duplicates(subset='id_titular', keep='first', inplace=True)
assert len(df_ges) == n_unique


# In[8]:


#feature generate
df_ges['renta_imponible_uf'] = df_ges['renta_imponible']/df_ges['valor_uf']
df_ges.loc[df_ges.renta_imponible_uf >= renta_tope, 'renta_imponible_uf'] = renta_tope
df_ges['periodo'] = pd.to_datetime(df_ges['periodo'])


# In[9]:


#cargamos datos de entrenamiento
df_train = pd.read_csv(inputs / train_file, index_col=0)
target = 'fuga_5m'


# In[10]:


#cargamos datos ltv que se usaran posteriormente
df_ltv = pd.read_csv(inputs / ltv_file) #.drop(columns = ['Unnamed: 0', 'index'])


# In[11]:


df_ltv.head()


# ## **Predicciones**

# In[12]:


#iniciamos h2o
H2O_server = h2o.init(port=54321, nthreads=-1, max_mem_size =60) #max_mem_size considera GB si no se espefica
h2o.remove_all() 


# In[13]:


#cargamos el modelo
modelo_fuga = h2o.load_model(path=str(inputs / "modelos" / "may20_modelo_proyecto_ges"))


# In[14]:


df_ges.head()


# In[15]:


#clasificamos las columnas
df_ges_h2o = h2o.H2OFrame(df_ges[total_cols], destination_frame="datosh2o")

for col in numeric:
    df_ges_h2o[col] = df_ges_h2o[col].asnumeric()
for col in categs:
    df_ges_h2o[col] = df_ges_h2o[col].asfactor()

df_train_h2o = h2o.H2OFrame(df_train, destination_frame="dfh2o")
df_train_h2o = df_train_h2o.drop('fuga_5m')


# In[16]:


#predecimos
predict_new = modelo_fuga.predict(df_ges_h2o)
predict_new = predict_new.as_data_frame()

predict_train = modelo_fuga.predict(df_train_h2o)
predict_train = predict_train.as_data_frame()


# In[17]:


#adjuntamos la probabilidad a los datos ges
df_ges['prob_fuga'] = predict_new['p1'].values
df_train['prob_fuga'] = predict_train['p1'].values


# In[18]:


df_ges.prob_fuga.mean()


# In[19]:


#chequeamos que tanto cambio la probabilidad de fuga. No debiese superar el 0.05, si no, chequear los datos.
w_dist(df_train['prob_fuga'].values,df_ges['prob_fuga'].values)


# In[20]:


#descomentar solo si el número de arriba salio extraño
#_ = compare_cont_dists([predict_new, predict_train], ['p1'], labels=[periodo, 'train'])
#_ = compare_cont_dists([df_ges, df_train], numeric, ['new', 'train'])
#_ = compare_categorical_dists(df_ges, df_train, categs, ['new', 'train'])


# In[21]:


guardar_cols = [col for col in df_ltv.columns]
guardar_cols += ['prob_fuga']

#merge datos ges-ltv
df_gl = df_ges.copy()
df_gl = pd.merge(df_ges, df_ltv, on='id_titular', how='left')

#Desduplicar
n_unique = df_gl.id_titular.nunique()
df_gl.drop_duplicates(subset='id_titular', keep='first', inplace=True)
assert len(df_gl) == n_unique


# In[22]:


df_gl[guardar_cols].to_csv(outputs / f'base_entregable_{periodo}_Prob_Fuga.csv')


# In[23]:


#traemos datos de bigquery para generar la columna cie_complejo. Desde marzo hasta junio se usa la tabla de enero, 
#de ahí hacia adelante se usa la que corresponde al periodo

# if periodo in ['mar19','abr19','may19']:
#     periodo_bq = 'ene19'
# else:
#     periodo_bq = periodo

# query = f"""
# SELECT id_titular, periodo, cie_complejo
# FROM `LTV.LTV_NEW_oct20`

# """

# pid = 'estudios-242917' #'estudio-dev' #'estudios-242917' #

# id_periodo = pd.read_gbq(query, project_id=pid, dialect='standard', private_key = inputs+"modelos/Estudios-bfe21c8b729c.json", 
#                  configuration={"allow_large_results": True}) #progress_bar_type = 'tqdm_notebook'


# In[24]:


# se calcula en snow. 
# id_periodo = pd.read_csv(inputs+'20.12.21_id_periodo_cie_complejo_nov20.csv', sep=';', usecols=['id_titular', 'periodo', 'cie_complejo'])


# In[25]:


#script para crear los moving averages
def create_mov_avgs(datos, shift_m, cols, window=6, min_periods=3):
    """
    Crea medias móviles para variables `cols`.
    shift_m: el número de meses para ir hacia atrás
    Window: qué tan larga es la ventana para tomar el promedio
    Min_periods: número mínimo de promedios para calcular el promedio
    """
    return (datos.groupby('id_titular')[cols]
                 .shift(periods=shift_m)
                 .rolling(window=window, min_periods=min_periods)
                 .max())


# In[26]:


#periodo str to int
p = {'ene':1,'feb':2,'mar':3,'abr':4,'may':5,'jun':6,'jul':7,'ago':8,'sep':9,'oct':10,'nov':11,'dic':12}


# In[27]:


# titulares = df_gl.id_titular.unique().tolist()
# id_periodo['periodo'] = pd.to_datetime(id_periodo['periodo'], format='%Y%m')
# cie_complejo = pd.concat([id_periodo, df_gl[['id_titular', 'cie_complejo', 'periodo']]])
# cie_complejo = cie_complejo[cie_complejo.id_titular.isin(titulares)]
# cie_complejo.sort_values(['id_titular', 'periodo'], inplace=True)
# cie_complejo['cie_complejo_mva_6m'] = create_mov_avgs(cie_complejo, shift_m=0, window=6, cols='cie_complejo')
# cie_complejo = cie_complejo[(cie_complejo.periodo == pd.to_datetime(f'2021-{p[periodo[:3]]}-1'))][['id_titular', 'cie_complejo_mva_6m']].sort_values(by=['id_titular', 'cie_complejo_mva_6m']).drop_duplicates(subset=['id_titular'], keep = 'last')


# In[28]:


# cie_complejo[cie_complejo.id_titular == 76010006].sort_values(by=['id_titular', 'periodo', 'cie_complejo_mva_6m']).drop_duplicates(subset=['id_titular', 'periodo'], keep='last')


# In[29]:


#cie_complejo.cie_complejo_mva_6m.agg(['min','max', 'mean'])


# In[30]:


df_gl.shape #--, id_periodo.shape, cie_complejo.shape


# In[31]:


#pegamos la información
#df_gl_cie =  df_gl   # viene incluido en el archivo ; pd.merge(df_gl, cie_complejo, how='left', on='id_titular')


# In[32]:


ges_r_file


# In[33]:


# #se llama para recuperar algunos id
# # N_GES -> se utiliza para los filtros Colmena 
# cols_to_use = ['id_titular', 'periodo', 'n_ges']

# df_ges_r = pd.read_csv(input2+ges_r_file, usecols=cols_to_use,delimiter=';', compression='zip')
# df_ges_r['periodo'] = pd.to_datetime(df_ges_r['periodo'], format='%Y%m')
# df_ges_r = df_ges_r.rename(columns=lambda x: x.lower())
# df_ges_r.sort_values(['id_titular', 'periodo'], inplace=True)
# df_ges_r['n_ges_mva_2m'] = create_mov_avgs(df_ges_r, shift_m=0, window=2, cols='n_ges', min_periods=0)
# df_ges_r = df_ges_r[df_ges_r.periodo == pd.to_datetime(f'2021-{p[periodo[:3]]}-1')]  #CAMBIAR EL AÑO QUE ESTA ESCRITO EN DURO, GENERA PROBLEMA AL CAMBIAR EL AÑO
# df = df_gl_cie.merge(df_ges_r,how='left',on=['id_titular', 'periodo'])
df  = df_gl


# In[34]:


#output con todos los afiliados 
#df.to_csv(outputs+f'prediccion_total_ges_{periodo}.csv')


# ## **Clustering**: 
# A partir de ahora generamos un clustering para el 20% superior en cuanto a probabilidad de fuga

# In[35]:


df.columns


# In[36]:


df.shape


# In[37]:


def clusters(df,df_new,perc_fuga = [.2, .1, .05, 0],perc_ltv = [0, .33, .66, 1]):
    
    interven  = pd.DataFrame()
    interven_afiliados  = pd.DataFrame()
    interven_monto  = pd.DataFrame()

    columns = ['prob_baja', 'prob_media', 'prob_alta']

    quant_ltv = np.quantile(df['predicted_Margen_t+1'], perc_ltv)
    quant_prob = np.quantile(df_new.prob_fuga, 1-np.array(perc_fuga))
    res_ltv_list = [(df['predicted_Margen_t+1'] >= quant_ltv[i]) & (df['predicted_Margen_t+1'] <= quant_ltv[i+1]) for i in range(3)]
    res_fuga_list = [(df.prob_fuga >= quant_prob[i]) & (df.prob_fuga <= quant_prob[i+1]) for i in range(3)]
    res = list(product(res_ltv_list, res_fuga_list))
    
    for i, r in enumerate(res):
        r_ltv, r_fuga = r
        d_res = df[r_ltv & r_fuga]
        df.loc[r_ltv & r_fuga, 'cluster'] = int(1+i)
        N = d_res.id_titular.nunique()
        ltv_mean = .9*d_res['predicted_Margen_t+1'].mean()
        ltv_sum = .9*d_res['predicted_Margen_t+1'].sum()
        interven_afiliados.loc[f'ltv {ltv_mean:.1E}', columns[i%3]] = N
        interven_monto.loc[f'ltv {ltv_mean:.1E}', columns[i%3]] = ltv_sum
        interven.loc[f'ltv {ltv_mean:.1E}', columns[i%3]] = str(N) + ' | ' + f'{ltv_sum:.2E}'
    
    return df,interven,interven_afiliados,interven_monto


# In[38]:


def entregar(df,columns,name):
    
    df_entregable = df[columns].reset_index().drop(['index'], axis=1)
    df_entregable = df_entregable.rename(columns={'predicted_Margen_t+1': 'ltv_predicted_1y', 'prediccion_probabilidad_fuga_1y': 'prob_fuga_1y'})
    df_entregable['ltv_predicted_1y'] = .9 * df_entregable['ltv_predicted_1y']
    df_entregable.drop_duplicates(subset=['id_titular'],inplace=True)
    df_entregable.to_csv(outputs / name)
    
    return df_entregable


# In[39]:


#filtrar outliers ltv
lower_ltv = 0
upper_ltv = 1e7

res_ltv = (df.ltv_fuga0_predicted >= lower_ltv) & (df.ltv_fuga0_predicted <= upper_ltv)
res_ltv2 = (df.ltv_predicted > lower_ltv) & (df.ltv_predicted <= upper_ltv)
filtro_colmena = ((df.edad <= 65) & (df.cie_complejo_mva_6m == 0) & (df.n_ges_mva_2m == 0))
res_fuga = df.prob_fuga >= np.quantile(df.prob_fuga, 0.8)
titulares = df.id_titular.unique()


# In[40]:


(df.cie_complejo_mva_6m).unique()


# In[41]:


filtro_colmena.sum()


# In[42]:


res_ltv.sum(), res_fuga.sum(), filtro_colmena.sum()


# In[43]:


#Con filtros
df_fil_clus = clusters(df[res_ltv & res_fuga & filtro_colmena],df)[0]

columns = ['id_titular', 'sexo', 'edad', 'antiguedad', 'num_cargas', 
           'COMUNA', 'region_gls', 'clasif_riesgo', 'clasif_morosidad', 
           'prob_fuga', 'ltv_fuga0_predicted', 'prediccion_probabilidad_fuga_1y',
           'costo_final', 'costo_total', 'predicted_Margen_t+1',
           'cluster']

df_fil_ent = entregar(df_fil_clus,columns=columns,name=f'base_entregable_{periodo}_2.csv')


# In[44]:


#Sin filtros
df['cumple_con_filtros'] = (res_ltv & res_fuga & filtro_colmena).astype('int')
df_sfil_clus = clusters(df,df)[0]

columns = ['id_titular', 'sexo', 'edad', 'antiguedad', 'num_cargas', 
           'COMUNA', 'region_gls', 'clasif_riesgo', 'clasif_morosidad', 
           'prob_fuga', 'ltv_fuga0_predicted', 'prediccion_probabilidad_fuga_1y', 'costo_final', 'costo_total', 'predicted_Margen_t+1','cumple_con_filtros']

df_sfil_ent = entregar(df_sfil_clus,columns=columns,name=f'base_entregable_{periodo}.csv')


# In[45]:


try:
    df_fil_clus.drop(columns=['index'])
except:
    pass


# In[46]:


df_sfil_ent_to_write = df_sfil_ent.rename(columns=lambda x: x.upper())
df_fil_clus_to_write = df_fil_clus.drop(columns=['index'], errors='ignore').rename(columns=lambda x: x.upper())
df_ltv_to_write = df_ltv.drop(columns=['index'], errors='ignore').rename(columns=lambda x: x.upper())

if spark_session is not None:
    print('Comienzo carga en tablas Databricks')
    spark_session.createDataFrame(df_sfil_ent_to_write).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("EST.P_DDV_EST.JC_ML_GES_PRED")
    print('JC_ML_GES_PRED OK')
    spark_session.createDataFrame(df_fil_clus_to_write).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("EST.P_DDV_EST.JC_ML_GES_CLUSTER_PRED")
    print('JC_ML_GES_CLUSTER_PRED OK')
    spark_session.createDataFrame(df_ltv_to_write).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("EST.P_DDV_EST.JC_ML_LTV_PRED")
    print('JC_ML_LTV_PRED OK')
    print('Carga completa')
else:
    if create_engine is None or pd_writer is None:
        raise ImportError(
            "Dependencias de Snowflake no disponibles. Ejecuta en Databricks con Spark o instala snowflake/sqlalchemy."
        )

    snow_user = _get_credential("SNOWFLAKE_USER", "Usuario SnowFlake")
    snow_pass = _get_credential("SNOWFLAKE_PASSWORD", "Password SnowFlake")
    engine = create_engine(
        'snowflake://{user}:{password}@{account}/EST/P_DDV_EST'.format(
            user=snow_user,
            password=snow_pass,
            account=snow_account
        )
    )
    conn = None
    try:
        conn = engine.connect()
    #     results = conn.execute('select CURRENT_DATABASE(), CURRENT_SCHEMA()').fetchall()
        
        #Truncamos
        conn.execute('TRUNCATE TABLE IF EXISTS EST.P_DDV_EST.JC_ML_LTV_PRED')
        conn.execute('TRUNCATE TABLE IF EXISTS EST.P_DDV_EST.JC_ML_GES_CLUSTER_PRED')
        conn.execute('TRUNCATE TABLE IF EXISTS EST.P_DDV_EST.JC_ML_GES_PRED')
        
        print('Tablas truncadas')
            
        #Cargamos
        print('Comienzo carga')
        print('JC_ML_GES_PRED ', end ='')
        df_sfil_ent_to_write.to_sql('JC_ML_GES_PRED', conn, index=False, if_exists='append', method=pd_writer) # , chunksize=12000,
        print('Ok')
        print('JC_ML_GES_CLUSTER_PRED ', end='')
        df_fil_clus_to_write.to_sql('JC_ML_GES_CLUSTER_PRED', conn, index=False, if_exists='append', method=pd_writer)
        print('OK')
        print('JC_ML_LTV_PRED ', end='')
        df_ltv_to_write.to_sql('JC_ML_LTV_PRED', conn, index=False, if_exists='append', method=pd_writer)
        print('OK')
        print('Carga completa')
    finally:
        if conn is not None:
            conn.close()
        engine.dispose()


# In[47]:


df_ltv.columns


# In[48]:


df_sfil_ent.shape


# In[49]:


df_fil_clus.shape


# In[50]:


df_fil_clus.head()


# In[51]:


df.shape


# In[ ]:





# In[ ]:




