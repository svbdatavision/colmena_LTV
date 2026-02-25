#!/usr/bin/env python
# coding: utf-8

# ## Predección de fuga durante periodo GES

# In[1]:


# usar kernel spikelabs_env_3
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

#Cambie la ruta al PD; versión anterior:  estudio/data/

disco = '/mnt/disks/modelos/' #al bucket (original): /estudio/data/

import pandas as pd
#import feather
from itertools import product
import numpy as np
import h2o
import seaborn as sns
import os
from scipy.stats import wasserstein_distance as w_dist
import warnings
warnings.filterwarnings('ignore')
import json
import tqdm

import getpass as gp

import sys
sys.path.append(disco +'Proyecto_GES/script_graficos/')
from plot_helpers import compare_cont_dists
from plot_helpers import compare_categorical_dists

from snowflake import connector
from snowflake.connector.pandas_tools import pd_writer, write_pandas
from sqlalchemy import create_engine


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
# para poder ver todas las columnas
pd.set_option('display.max_columns',2000)


# In[2]:


snow_user = gp.getpass(prompt='Usuario SnowFlake')
snow_pass = gp.getpass(prompt = 'Password SnowFlake')


# In[3]:


# las demas rutas dependen de esto, por lo que es necesario verificar
#assert os.getcwd() == '/estudio/data/Proyecto_GES/Prediccion'


periodo = 'jun24'                                               #Modificar cada mes el periodo en español, en proceso 1 estan las abreviaciones
renta_tope = 84.3                                               #modifcar al tope en el periodo a predecir

#datos del training
train_file = 'retrain.csv'
#carpeta de inputs
inputs = 'input/prediccion/'

#especial para el archivo original sin pre-proceso
#aca va ltv del periodo (la tabla no predicciones) y base ges
input2 = disco+'Proyecto_GES/Prediccion/input/preproceso/'

#carpeta de output
outputs = 'output/'

#path total desde root, para el h2o laod
#aca va ltv predicciones del periodo y el archivo que se genera en el notebook de preproceso
input_total = disco+'Proyecto_GES/Prediccion/input/prediccion/'

#datos del periodo a predecir
#ges_file = f'ready_to_pred_{periodo}.csv'
#datos del ltv
#Automatizar la selección de este archivo.
#ltv_file = f'02. prediccion_ltv_con_categoria_{periodo}.csv'
for folder,_, files in os.walk(input_total):
    for file in files:
        if periodo in file:
            if 'ready' in file.lower():
                ges_file = file
            elif 'prediccion_ltv' in file.lower():
                ltv_file = file



#ges sin prep  20.04.17_GES_mar20
#ges_r_file = f'2021.03.19_GES_{periodo}.csv.zip'

for folder,_, files in os.walk(input2):
    for file in files:
        if periodo in file and 'ges' in file.lower():
            ges_r_file = file
            
print(f'archivo ges: {ges_file}')
print(f'archivo ltv: {ltv_file}')
print(f'archivo ges preproceso: {ges_r_file}')
  


# ## **Carga de datos:**
# Se cargan los datos del ltv del mes, del ges del mes y del train del modelo ges para comparar y validar

# In[4]:


#cargamos datos ges
df_ges = pd.read_csv(inputs+ges_file,index_col=0)


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
df_train = pd.read_csv(inputs+train_file,index_col=0)
target = 'fuga_5m'


# In[10]:


#cargamos datos ltv que se usaran posteriormente
df_ltv = pd.read_csv(inputs+ltv_file) #.drop(columns = ['Unnamed: 0', 'index'])


# In[11]:


df_ltv.head()


# ## **Predicciones**

# In[12]:


#iniciamos h2o
H2O_server = h2o.init(port=54321, nthreads=-1, max_mem_size =60) #max_mem_size considera GB si no se espefica
h2o.remove_all() 


# In[13]:


#cargamos el modelo
modelo_fuga = h2o.load_model(path=inputs+'modelos/may20_modelo_proyecto_ges')


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


df_gl[guardar_cols].to_csv(outputs+f'base_entregable_{periodo}_Prob_Fuga.csv')


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
    df_entregable.to_csv(outputs+name)
    
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


engine = create_engine(
    'snowflake://{user}:{password}@{account}/EST/P_DDV_EST'.format(
        user=snow_user,
        password=snow_pass,
        account='isapre_colmena.us-east-1'
    )
)
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
    df_sfil_ent.rename(columns = lambda x: x.upper()).to_sql('JC_ML_GES_PRED', conn, index = False, if_exists='append', method = pd_writer) # , chunksize=12000, 
    print('Ok')
    print('JC_ML_GES_CLUSTER_PRED ', end = '')
    df_fil_clus.drop(columns=['index']).rename(columns = lambda x: x.upper()).to_sql('JC_ML_GES_CLUSTER_PRED', conn, index = False, if_exists='append', method = pd_writer)
    print('OK')
    print('JC_ML_LTV_PRED')
    df_ltv.drop(columns=['index']).rename(columns = lambda x: x.upper()).to_sql('JC_ML_LTV_PRED', conn, index = False, if_exists='append', method = pd_writer)
    print('OK')
    print('Carga completa')

finally:
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




