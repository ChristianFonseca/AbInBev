import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from numpy import inf
from datetime import datetime

pd.set_option('max_columns',200)

path_data = '../data'

### Funciones de ayuda

# Funciones de Conversion
def convertirMesAnumero(fecha):
    anio = fecha//100
    mes = fecha%100
    return anio*12+mes

def convertirNumeroAMes(fecha):
    anio = fecha//12
    mes = fecha%12
    if mes == 0:
        mes_1 = 12
        anio=anio-1
    else:
        mes_1 = mes
        anio=anio
    return anio*100+mes_1

def agregarMeses(fecha,cntMes):
    fechaNumero=convertirMesAnumero(fecha)
    fechaNumero = fechaNumero + cntMes
    return convertirNumeroAMes(fechaNumero)

# Funcion de validacion cruzada usando LightGBM
def model_training(train, test, features, cat_ind, target, parameters, nfolds, seed=256):
    kf_previo=StratifiedKFold(n_splits=nfolds,random_state=seed,shuffle=True)

    i=1
    r=[]
    importancias=pd.DataFrame()
    importancias['variable']=features

    for train_index,test_index in kf_previo.split(train,train[target]):

        lgb_train = lgb.Dataset(train.loc[train_index,features].values,train.loc[train_index,target].values.ravel())
        lgb_eval = lgb.Dataset(train.loc[test_index,features].values,train.loc[test_index,target].values.ravel(), reference=lgb_train)

        params = parameters 

        lgbm3 = lgb.train(params,lgb_train,num_boost_round=20000,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=25,categorical_feature=cat_ind) # 
        test["TARGET_FOLD"+str(i)]=lgbm3.predict(test[features].values, num_iteration=lgbm3.best_iteration)

        importancias['gain_'+str(i)]=lgbm3.feature_importance(importance_type="gain")


        print ("Fold_"+str(i))
        a= (roc_auc_score(train.loc[test_index,target],lgbm3.predict(train.loc[test_index,features].values, num_iteration=lgbm3.best_iteration)))
        r.append(a)
        print (a)
        print ("")

        i=i+1

    print ("mean: "+str(np.mean(np.array(r))))
    print ("std: "+str(np.std(np.array(r))))
    
    return lgbm3, importancias, test

## Contrucción del TARGET

#### Base Executed Promos

df_ep = pd.read_csv(f'{path_data}/executed_promos.csv',encoding="ISO-8859-1") # executed promos
df_ep.columns = [x.upper() for x in df_ep.columns]

# El target sale de la data historica de promociones realizadas, las cuales se asumen que son efectivas
df_ep['TARGET'] = 1

df_ep.shape

df_ep.head()

#### Base Active Promotions

df_ap = pd.read_csv(f'{path_data}/active_promos.csv',encoding="ISO-8859-1") # active promos
df_ap.columns = [x.upper() for x in df_ap.columns]
df_ap = df_ap[['CODIGODC','CLIENTE','MARCA','CUPO','FECHA_DESDE','FECHA_HASTA']]

df_ap = df_ap.sort_values(by='CODIGODC').reset_index(drop=True)

df_ap.shape

df_ap

#### Base Client Attributes

df_ca = pd.read_csv(f'{path_data}/clients_attributes.csv',encoding="ISO-8859-1") # clients attributes
df_ca.columns = [x.upper() for x in df_ca.columns]

df_ca['CODMES_ALTA'] = df_ca['FECHAALTACLIENTE'].str[0:4].map(int) * 100 + df_ca['FECHAALTACLIENTE'].str[5:7].map(int)

# creacion de dummies de la region
df_ca = df_ca.join(pd.get_dummies(df_ca['REGION'],prefix='region')[['region_1','region_3','region_4','region_5']])
df_ca.drop(columns=['REGION'],inplace=True)

df_ca.columns = [k.upper() for k in df_ca.columns]

# datos que usaremos
df_ca_final = df_ca[['CLIENTE','CODMES_ALTA','GERENCIA','SUBCANAL','TIPOPOBLACION','ESTRATO','EF','REGION_1','REGION_3','REGION_4','REGION_5']]

df_ca_final

#### Base de Sales

# Lectura de datos de Sales
df_sa = pd.read_csv(f'{path_data}/sales.csv',encoding="ISO-8859-1") # sales
# Conversion de nombres a mayuscula
df_sa.columns = [x.upper() for x in df_sa.columns]
# Ordenamiento de data en funcion de cliente, año y mes
df_sa = df_sa.sort_values(by=['CLIENTE','AÑO','MES']).reset_index(drop=True)

print(f'Dimensiones de la data de sales: {df_sa.shape}')

# creacion del codmes
df_sa['CODMES'] = df_sa['AÑO']*100 + df_sa['MES']
df_sa.drop(columns=['AÑO','MES'],inplace=True)

df_sa[(df_sa['CLIENTE']==15069) & (df_sa['CLASEENVASE']==1) & (df_sa['MARCA']==44) & (df_sa['CUPO']==30.0)]

# creacion de dummys en función de la clase del envase
df_sa = df_sa.join(pd.get_dummies(df_sa['CLASEENVASE'],prefix='ENVASE_CLASE')[['ENVASE_CLASE_1','ENVASE_CLASE_2']])
#df_sa.drop(columns=['CLASEENVASE'],inplace=True)
# creacion de dummys en función del segmentos de Precios
df_sa = df_sa.join(pd.get_dummies(df_sa['SEGMENTOPRECIO'],prefix='SEGMENTO'))
#df_sa.drop(columns=['SEGMENTOPRECIO'],inplace=True)

df_sa.rename(columns={'SEGMENTO_1.0':'SEGMENTO_1',
                            'SEGMENTO_2.0':'SEGMENTO_2',
                            'SEGMENTO_3.0':'SEGMENTO_3',
                            'SEGMENTO_4.0':'SEGMENTO_4'},
                   inplace=True)

df_sa = df_sa.groupby(['CLIENTE','CODMES','MARCA','CUPO']).agg({'NR':'sum','HL':'sum','DCTO':'sum',
                        'ENVASE_CLASE_1':'max','ENVASE_CLASE_2':'max',
                        'SEGMENTO_1':'max','SEGMENTO_2':'max','SEGMENTO_3':'max','SEGMENTO_4':'max',
                        'CLASEENVASE':'first','SEGMENTOPRECIO':'first'}).reset_index()

df_sa['FLG_DCTO'] = ( df_sa['DCTO'] < 0 ) * 1

df_sa['BR'] = df_sa['NR'] + df_sa['DCTO']

from numpy import inf
df_sa['VAR_BR_HL'] = df_sa['BR']/df_sa['HL']
df_sa.loc[df_sa['VAR_BR_HL'] == -inf,'VAR_BR_HL'] = np.nan
df_sa.loc[df_sa['VAR_BR_HL'] == inf,'VAR_BR_HL'] = np.nan

##### Construccion de variables historicas de Sales

df_sa1 = df_sa[['CLIENTE','MARCA','CUPO']].drop_duplicates()

df_sa2 = df_sa[(df_sa['CODMES'] > 201905) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'NR':'max', 'BR':'max', 'HL':'max', 'DCTO':'max'}).reset_index()
df_sa2.rename(columns={'NR':'NR_3', 'BR':'BR_3', 'HL':'HL_3', 'DCTO':'DCTO_3'},inplace=True)
df_sa3 = df_sa[(df_sa['CODMES'] > 201902) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'NR':'max', 'BR':'max', 'HL':'max', 'DCTO':'max'}).reset_index()
df_sa3.rename(columns={'NR':'NR_6', 'BR':'BR_6', 'HL':'HL_6', 'DCTO':'DCTO_6'},inplace=True)
df_sa4 = df_sa[(df_sa['CODMES'] > 201812) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'NR':'max', 'BR':'max', 'HL':'max', 'DCTO':'max'}).reset_index()
df_sa4.rename(columns={'NR':'NR_9', 'BR':'BR_9', 'HL':'HL_9', 'DCTO':'DCTO_9'},inplace=True)
df_sa5 = df_sa[(df_sa['CODMES'] > 201808) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'NR':'max', 'BR':'max', 'HL':'max', 'DCTO':'max'}).reset_index()
df_sa5.rename(columns={'NR':'NR_12', 'BR':'BR_12', 'HL':'HL_12', 'DCTO':'DCTO_12'},inplace=True)
df_sa6 = df_sa[(df_sa['CODMES'] >= 201801) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'NR':'max', 'BR':'max', 'HL':'max', 'DCTO':'max'}).reset_index()
df_sa6.rename(columns={'NR':'NR_ALL', 'BR':'BR_ALL', 'HL':'HL_ALL', 'DCTO':'DCTO_ALL'},inplace=True)


df_sa7  = df_sa[(df_sa['CODMES'] > 201905) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'ENVASE_CLASE_1':['max','nunique','size','std'],'ENVASE_CLASE_2':['max','nunique','size','std'], 'SEGMENTO_1':['max','nunique','size','std'], 'SEGMENTO_2':['max','nunique','size','std'], 'SEGMENTO_3':['max','nunique','size','std'], 'SEGMENTO_4':['max','nunique','size','std'], 'FLG_DCTO':['max','nunique','size','std']}).reset_index()
df_sa7.columns = [n1 +'_' + n2 for n1, n2 in df_sa7.columns]
df_sa7.rename(columns={'CLIENTE_':'CLIENTE', 'MARCA_':'MARCA', 'CUPO_':'CUPO'},inplace=True)
df_sa7.rename(columns=dict([ (k, k + '_3') for k in df_sa7.columns if k not in ['CLIENTE','MARCA','CUPO']]),inplace=True)

df_sa8  = df_sa[(df_sa['CODMES'] > 201902) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'ENVASE_CLASE_1':['max','nunique','size','std'],'ENVASE_CLASE_2':['max','nunique','size','std'], 'SEGMENTO_1':['max','nunique','size','std'], 'SEGMENTO_2':['max','nunique','size','std'], 'SEGMENTO_3':['max','nunique','size','std'], 'SEGMENTO_4':['max','nunique','size','std'], 'FLG_DCTO':['max','nunique','size','std']}).reset_index()
df_sa8.columns = [n1 +'_' + n2 for n1, n2 in df_sa8.columns]
df_sa8.rename(columns={'CLIENTE_':'CLIENTE', 'MARCA_':'MARCA', 'CUPO_':'CUPO'},inplace=True)
df_sa8.rename(columns=dict([ (k, k + '_6') for k in df_sa8.columns if k not in ['CLIENTE','MARCA','CUPO']]),inplace=True)

df_sa9 = df_sa[(df_sa['CODMES'] > 201812) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'ENVASE_CLASE_1':['max','nunique','size','std'],'ENVASE_CLASE_2':['max','nunique','size','std'], 'SEGMENTO_1':['max','nunique','size','std'], 'SEGMENTO_2':['max','nunique','size','std'], 'SEGMENTO_3':['max','nunique','size','std'], 'SEGMENTO_4':['max','nunique','size','std'], 'FLG_DCTO':['max','nunique','size','std']}).reset_index()
df_sa9.columns = [n1 +'_' + n2 for n1, n2 in df_sa9.columns]
df_sa9.rename(columns={'CLIENTE_':'CLIENTE', 'MARCA_':'MARCA', 'CUPO_':'CUPO'},inplace=True)
df_sa9.rename(columns=dict([ (k, k + '_9') for k in df_sa9.columns if k not in ['CLIENTE','MARCA','CUPO']]),inplace=True)

df_sa10 = df_sa[(df_sa['CODMES'] > 201808) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'ENVASE_CLASE_1':['max','nunique','size','std'],'ENVASE_CLASE_2':['max','nunique','size','std'], 'SEGMENTO_1':['max','nunique','size','std'], 'SEGMENTO_2':['max','nunique','size','std'], 'SEGMENTO_3':['max','nunique','size','std'], 'SEGMENTO_4':['max','nunique','size','std'], 'FLG_DCTO':['max','nunique','size','std']}).reset_index()
df_sa10.columns = [n1 +'_' + n2 for n1, n2 in df_sa10.columns]
df_sa10.rename(columns={'CLIENTE_':'CLIENTE', 'MARCA_':'MARCA', 'CUPO_':'CUPO'},inplace=True)
df_sa10.rename(columns=dict([ (k, k + '_12') for k in df_sa10.columns if k not in ['CLIENTE','MARCA','CUPO']]),inplace=True)

df_sa11 =  df_sa[(df_sa['CODMES'] >= 201801) & (df_sa['CODMES'] <= 201908)].groupby(['CLIENTE','MARCA','CUPO']).agg({'ENVASE_CLASE_1':['max','nunique','size','std'],'ENVASE_CLASE_2':['max','nunique','size','std'], 'SEGMENTO_1':['max','nunique','size','std'], 'SEGMENTO_2':['max','nunique','size','std'], 'SEGMENTO_3':['max','nunique','size','std'], 'SEGMENTO_4':['max','nunique','size','std'], 'FLG_DCTO':['max','nunique','size','std']}).reset_index()
df_sa11.columns = [n1 +'_' + n2 for n1, n2 in df_sa11.columns]
df_sa11.rename(columns={'CLIENTE_':'CLIENTE', 'MARCA_':'MARCA', 'CUPO_':'CUPO'},inplace=True)
df_sa11.rename(columns=dict([ (k, k + '_ALL') for k in df_sa11.columns if k not in ['CLIENTE','MARCA','CUPO']]),inplace=True)


dfp = df_sa1.merge(df_sa2, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa3, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa4, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa5, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa6, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa7, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa8, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa9, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa10, on=['CLIENTE','MARCA','CUPO'], how='left')
dfp = dfp.merge(df_sa11, on=['CLIENTE','MARCA','CUPO'], how='left')

# Creacion de variables historicas dentro de los meses actuales (6 meses atras)
df_sa.sort_values(['CLIENTE','CODMES','MARCA','CUPO'],ascending=True,inplace=True)
for x in ['NR', 'BR', 'HL', 'DCTO', 'ENVASE_CLASE_1', 'ENVASE_CLASE_2', 'SEGMENTO_1', 'SEGMENTO_2',
          'SEGMENTO_3', 'SEGMENTO_4','FLG_DCTO','CODMES','CLASEENVASE','SEGMENTOPRECIO','VAR_BR_HL']:
    df_sa[x+'_1']=df_sa.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(1)
    df_sa[x+'_2']=df_sa.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(2)
    df_sa[x+'_3']=df_sa.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(3)
    df_sa[x+'_4']=df_sa.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(4)
    df_sa[x+'_5']=df_sa.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(5)
    df_sa[x+'_6']=df_sa.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(6)

# Al tener datos historicos, se agrega un mes a todos los meses debido a que 
# ya se tienen todos los historicos necesarios
df_sa['CODMES'] = df_sa['CODMES'].apply(lambda k: agregarMeses(k,1))

# Creacion de veces que han accedido a al promocion
df_sa['NRO_RENOVACIONES'] = df_sa[['NR_1','NR_2','NR_3','NR_4',
                                   'NR_5','NR_6']].notnull().sum(axis=1) 

df_sa['NR_TOTAL'] = df_sa[['NR_1','NR_2','NR_3','NR_4',
                                   'NR_5','NR_6']].sum(axis=1) 
df_sa['NR_PROMEDIO'] = df_sa['NR_TOTAL']/df_sa['NRO_RENOVACIONES']

# creacion de donde fue que accedio a la promocion
df_sa['RECENCIA'] = 0
df_sa.loc[df_sa['FLG_DCTO_1']>=1,'RECENCIA'] = 1
df_sa.loc[(df_sa['FLG_DCTO_1']==0)&(df_sa['FLG_DCTO_2']>=1),'RECENCIA'] = 2
df_sa.loc[(df_sa['FLG_DCTO_1']==0)&(df_sa['FLG_DCTO_2']==0)&(df_sa['FLG_DCTO_3']>=1),'RECENCIA'] = 3
df_sa.loc[(df_sa['FLG_DCTO_1']==0)&(df_sa['FLG_DCTO_2']==0)&(df_sa['FLG_DCTO_3']==0)
      &(df_sa['FLG_DCTO_4']>=1),'RECENCIA'] = 4
df_sa.loc[(df_sa['FLG_DCTO_1']==0)&(df_sa['FLG_DCTO_2']==0)&(df_sa['FLG_DCTO_3']==0)
      &(df_sa['FLG_DCTO_4']==0)&(df_sa['FLG_DCTO_5']>=1),'RECENCIA'] = 5
df_sa.loc[(df_sa['FLG_DCTO_1']==0)&(df_sa['FLG_DCTO_2']==0)&(df_sa['FLG_DCTO_3']==0)
      &(df_sa['FLG_DCTO_4']==0)&(df_sa['FLG_DCTO_5']==0)&(df_sa['FLG_DCTO_6']>=1),'RECENCIA'] = 6

# Continuidad de descuentos
df_sa['FLG_CONT_1'] = (df_sa['CODMES'].apply(lambda k: convertirNumeroAMes(agregarMeses(k,-1))) - df_sa['CODMES_1'].apply(lambda k: convertirNumeroAMes(k))).fillna(0)
df_sa['FLG_CONT_2'] = (df_sa['CODMES'].apply(lambda k: convertirNumeroAMes(agregarMeses(k,-2))) - df_sa['CODMES_2'].apply(lambda k: convertirNumeroAMes(k))).fillna(0)
df_sa['FLG_CONT_3'] = (df_sa['CODMES'].apply(lambda k: convertirNumeroAMes(agregarMeses(k,-3))) - df_sa['CODMES_3'].apply(lambda k: convertirNumeroAMes(k))).fillna(0)
df_sa['FLG_CONT_4'] = (df_sa['CODMES'].apply(lambda k: convertirNumeroAMes(agregarMeses(k,-4))) - df_sa['CODMES_4'].apply(lambda k: convertirNumeroAMes(k))).fillna(0)
df_sa['FLG_CONT_5'] = (df_sa['CODMES'].apply(lambda k: convertirNumeroAMes(agregarMeses(k,-5))) - df_sa['CODMES_5'].apply(lambda k: convertirNumeroAMes(k))).fillna(0)
df_sa['FLG_CONT_6'] = (df_sa['CODMES'].apply(lambda k: convertirNumeroAMes(agregarMeses(k,-6))) - df_sa['CODMES_6'].apply(lambda k: convertirNumeroAMes(k))).fillna(0)

df_sa['CONTINUO'] = df_sa[['FLG_CONT_1','FLG_CONT_2','FLG_CONT_3','FLG_CONT_4','FLG_CONT_5','FLG_CONT_6']].sum(axis=1)

# creacion de variaciones de Sales
columns_sa = [ 'NR', 'BR', 'HL',
               'NR_1', 'NR_2', 'NR_3', 'NR_4', 'NR_5','NR_6',
               'BR_1', 'BR_2', 'BR_3', 'BR_4', 'BR_5','BR_6', 
               'HL_1', 'HL_2', 'HL_3', 'HL_4', 'HL_5', 'HL_6', 
               'DCTO_1','DCTO_2', 'DCTO_3', 'DCTO_4', 'DCTO_5', 'DCTO_6', 
               'CODMES_1','CODMES_2', 'CODMES_3', 'CODMES_4', 'CODMES_5', 'CODMES_6',
               'ENVASE_CLASE_1_1', 'ENVASE_CLASE_1_2', 'ENVASE_CLASE_1_3',
               'ENVASE_CLASE_1_4', 'ENVASE_CLASE_1_5', 'ENVASE_CLASE_1_6',
               'ENVASE_CLASE_2_1', 'ENVASE_CLASE_2_2', 'ENVASE_CLASE_2_3',
               'ENVASE_CLASE_2_4', 'ENVASE_CLASE_2_5', 'ENVASE_CLASE_2_6',
               'SEGMENTO_1_1', 'SEGMENTO_1_2', 'SEGMENTO_1_3', 'SEGMENTO_1_4',
               'SEGMENTO_1_5', 'SEGMENTO_1_6', 'SEGMENTO_2_1', 'SEGMENTO_2_2',
               'SEGMENTO_2_3', 'SEGMENTO_2_4', 'SEGMENTO_2_5', 'SEGMENTO_2_6',
               'SEGMENTO_3_1', 'SEGMENTO_3_2', 'SEGMENTO_3_3', 'SEGMENTO_3_4',
               'SEGMENTO_3_5', 'SEGMENTO_3_6', 'SEGMENTO_4_1', 'SEGMENTO_4_2',
               'SEGMENTO_4_3', 'SEGMENTO_4_4', 'SEGMENTO_4_5', 'SEGMENTO_4_6']

for x in columns_sa:
    df_sa[x] = df_sa[x].fillna(0)


df_sa["HL_prom_u3m"] = (df_sa[["HL_1","HL_2","HL_3"]].sum(axis=1))/3
df_sa["HL_prom_u6m"] = (df_sa[["HL_1","HL_2","HL_3","HL_4","HL_5","HL_6"]].sum(axis=1))/6

df_sa["DCTO_prom_u3m"] = (df_sa[["DCTO_1","DCTO_2","DCTO_3"]].sum(axis=1))/3
df_sa["DCTO_prom_u6m"] = (df_sa[["DCTO_1","DCTO_2","DCTO_3","DCTO_4","DCTO_5","DCTO_6"]].sum(axis=1))/6

df_sa["NR_prom_u3m"] = (df_sa[["NR_1","NR_2","NR_3"]].sum(axis=1))/3
df_sa["NR_prom_u6m"] = (df_sa[["NR_1","NR_2","NR_3","NR_4","NR_5","NR_6"]].sum(axis=1))/6

df_sa["BR_prom_u3m"] = (df_sa[["BR_1","BR_2","BR_3"]].sum(axis=1))/3
df_sa["BR_prom_u6m"] = (df_sa[["BR_1","BR_2","BR_3","BR_4","BR_5","BR_6"]].sum(axis=1))/6

df_sa["HL_var_u3m"] = df_sa["HL"]/df_sa["HL_prom_u3m"]
df_sa["HL_var_u6m"] = df_sa["HL"]/df_sa["HL_prom_u6m"]
df_sa.loc[df_sa['HL_var_u3m'] == -inf,'HL_var_u3m'] = np.nan
df_sa.loc[df_sa['HL_var_u3m'] == inf,'HL_var_u3m'] = np.nan
df_sa.loc[df_sa['HL_var_u6m'] == -inf,'HL_var_u6m'] = np.nan
df_sa.loc[df_sa['HL_var_u6m'] == inf,'HL_var_u6m'] = np.nan

df_sa["NR_var_u3m"] = df_sa["NR"]/df_sa["NR_prom_u3m"]
df_sa["NR_var_u6m"] = df_sa["NR"]/df_sa["NR_prom_u6m"]
df_sa.loc[df_sa['NR_var_u3m'] == -inf,'NR_var_u3m'] = np.nan
df_sa.loc[df_sa['NR_var_u3m'] == inf,'NR_var_u3m'] = np.nan
df_sa.loc[df_sa['NR_var_u6m'] == -inf,'NR_var_u6m'] = np.nan
df_sa.loc[df_sa['NR_var_u6m'] == inf,'NR_var_u6m'] = np.nan

df_sa["BR_var_u3m"] = df_sa["BR"]/df_sa["BR_prom_u3m"]
df_sa["BR_var_u6m"] = df_sa["BR"]/df_sa["BR_prom_u6m"]
df_sa.loc[df_sa['BR_var_u3m'] == -inf,'BR_var_u3m'] = np.nan
df_sa.loc[df_sa['BR_var_u3m'] == inf,'BR_var_u3m'] = np.nan
df_sa.loc[df_sa['BR_var_u6m'] == -inf,'BR_var_u6m'] = np.nan
df_sa.loc[df_sa['BR_var_u6m'] == inf,'BR_var_u6m'] = np.nan

df_sa["DCTO_var_u3m"] = df_sa["DCTO"]/df_sa["DCTO_prom_u3m"]
df_sa["DCTO_var_u6m"] = df_sa["DCTO"]/df_sa["DCTO_prom_u6m"]
df_sa.loc[df_sa['DCTO_var_u3m'] == -inf,'DCTO_var_u3m'] = np.nan
df_sa.loc[df_sa['DCTO_var_u3m'] == inf,'DCTO_var_u3m'] = np.nan
df_sa.loc[df_sa['DCTO_var_u6m'] == -inf,'DCTO_var_u6m'] = np.nan
df_sa.loc[df_sa['DCTO_var_u6m'] == inf,'DCTO_var_u6m'] = np.nan


df_sa['ENVASE_CLASE_1_COUNT'] = (df_sa[['ENVASE_CLASE_1_1','ENVASE_CLASE_1_2','ENVASE_CLASE_1_3','ENVASE_CLASE_1_4','ENVASE_CLASE_1_5','ENVASE_CLASE_1_6']].sum(axis=1))/6
df_sa['ENVASE_CLASE_2_COUNT'] = (df_sa[['ENVASE_CLASE_2_1','ENVASE_CLASE_2_2','ENVASE_CLASE_2_3','ENVASE_CLASE_2_4','ENVASE_CLASE_2_5','ENVASE_CLASE_2_6']].sum(axis=1))/6

df_sa['SEGMENTO_1_PORC'] = (df_sa[['SEGMENTO_1_1','SEGMENTO_1_2','SEGMENTO_1_3','SEGMENTO_1_4','SEGMENTO_1_5','SEGMENTO_1_6']].sum(axis=1))/6
df_sa['SEGMENTO_2_PORC'] = (df_sa[['SEGMENTO_2_1','SEGMENTO_2_2','SEGMENTO_2_3','SEGMENTO_2_4','SEGMENTO_2_5','SEGMENTO_2_6']].sum(axis=1))/6
df_sa['SEGMENTO_3_PORC'] = (df_sa[['SEGMENTO_3_1','SEGMENTO_3_2','SEGMENTO_3_3','SEGMENTO_3_4','SEGMENTO_3_5','SEGMENTO_3_6']].sum(axis=1))/6
df_sa['SEGMENTO_4_PORC'] = (df_sa[['SEGMENTO_4_1','SEGMENTO_4_2','SEGMENTO_4_3','SEGMENTO_4_4','SEGMENTO_4_5','SEGMENTO_4_6']].sum(axis=1))/6

df_sa['VAR_BR_HL_PROM_6']= (df_sa[['VAR_BR_HL_1','VAR_BR_HL_2','VAR_BR_HL_3','VAR_BR_HL_4','VAR_BR_HL_5','VAR_BR_HL_6']].sum(axis=1))/6
df_sa['VAR_BR_HL_PROM_3']= (df_sa[['VAR_BR_HL_1','VAR_BR_HL_2','VAR_BR_HL_3']].sum(axis=1))/3

df_sa['SEGMENTOPRECIO_MAX'] = (df_sa[['SEGMENTOPRECIO_1', 'SEGMENTOPRECIO_2','SEGMENTOPRECIO_3', 'SEGMENTOPRECIO_4', 'SEGMENTOPRECIO_5','SEGMENTOPRECIO_6']].max(axis=1))
df_sa['SEGMENTOPRECIO_MIN'] = (df_sa[['SEGMENTOPRECIO_1', 'SEGMENTOPRECIO_2','SEGMENTOPRECIO_3', 'SEGMENTOPRECIO_4', 'SEGMENTOPRECIO_5','SEGMENTOPRECIO_6']].min(axis=1))

# Uso de variables
df_sa_u = df_sa.drop(columns=['DCTO','CODMES_1','CODMES_2','CODMES_3','CODMES_4','CODMES_5','CODMES_6',
                             'FLG_CONT_1','FLG_CONT_2','FLG_CONT_3','FLG_CONT_4','FLG_CONT_5','FLG_CONT_6',
                             'DCTO_1','DCTO_2','DCTO_3','DCTO_4','DCTO_5','DCTO_6',
                             "HL_1","HL_2","HL_3","HL_4","HL_5","HL_6",
                             "NR_1","NR_2","NR_3","NR_4","NR_5","NR_6",
                             "BR_1","BR_2","BR_3","BR_4","BR_5","BR_6"
                             ])

# Merge con data no historica acumulada
df_sa_u = df_sa_u.merge(dfp,on=['CLIENTE','MARCA','CUPO'], how='left')

df_sa_final = df_sa_u.copy()

df_sa_final.shape

##### CONTRUCCION DEL TARGET 

# Juntando la base de promociones activas con promociones anteriormente ejecutadas
df = df_ap.merge(df_ep, on=['CODIGODC','CLIENTE','MARCA','CUPO'], how='left')
df.loc[df['TARGET'].isnull(),'TARGET'] = 0
print(f'Dimensión de nuestra base general: {df.shape}')

print(f"Cantidad de Promociones desde 201808 - 201909: {len(df['CODIGODC'].unique())}")

# Juntando la base final con la base de catributos de clientes
df2 = df.merge(df_ca_final, on=['CLIENTE'], how='left')
df2 = df2.sort_values(by=['CODIGODC','CLIENTE','MARCA','CUPO']).reset_index(drop=True)
df2['CODMES'] = df2['FECHA_DESDE'].str[0:4].map(int) * 100 + df2['FECHA_DESDE'].str[5:7].map(int)
print(f'Dimensión de nuestra base general: {df2.shape}')

# Antiguedad del cliente
df2['MESES_CONTINUOS'] = df2['CODMES'].apply(lambda k: convertirMesAnumero(k)) - df2['CODMES_ALTA'].apply(lambda k: convertirMesAnumero(k))

# Verificacion y creacion de data mensual
df3 = df2.groupby(['CLIENTE','CODMES','MARCA','CUPO']).\
          agg({'GERENCIA':'first','SUBCANAL':'first','TIPOPOBLACION':'first',
               'ESTRATO':'first','EF':'first',
               'REGION_1':'first','REGION_3':'first','REGION_4':'first','REGION_5':'first',
               'TARGET':['max','size','sum','nunique'],'MESES_CONTINUOS':'max'}).\
          reset_index()

# Conversiones de columnas
df3.columns = [n1 +'_' + n2 for n1, n2 in df3.columns]
df3 = df3.reset_index(drop=True)

df3.rename(columns={'TARGET_max':'TARGET',
                      'CLIENTE_':'CLIENTE',
                      'CODMES_':'CODMES',
                      'CUPO_':'CUPO',
                      'MARCA_':'MARCA',
                      'GERENCIA_first':'GERENCIA',
                      'TIPOPOBLACION_first':'TIPOPOBLACION',
                      'ESTRATO_first':'ESTRATO',
                      'EF_first':'EF',
                      'REGION_1_first':'REGION_1',
                      'REGION_3_first':'REGION_3',
                      'REGION_4_first':'REGION_4',
                      'REGION_5_first':'REGION_5',
                      'MESES_CONTINUOS_max':'MESES_CONTINUOS',
                     },
                   inplace=True)


# Binarización de continuidad
df3['MESES_CONTINUOS'] = pd.cut(df3['MESES_CONTINUOS'],bins=10,labels=[x for x in range(10)]).map(int)

# Distribución de Target a lo largo de los meses (6 meses atras)
df3.sort_values(['CLIENTE','CODMES','MARCA','CUPO'],ascending=True,inplace=True)
for x in ['TARGET']:
    df3[x+'_1']=df3.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(1)
    df3[x+'_2']=df3.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(2)
    df3[x+'_3']=df3.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(3)
    df3[x+'_4']=df3.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(4)
    df3[x+'_5']=df3.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(5)
    df3[x+'_6']=df3.groupby(['CLIENTE','MARCA','CUPO'])[x].shift(6)

df4 = df3.merge(df_sa_final, on=['CLIENTE','CODMES','MARCA','CUPO'], how='left')
df4 = df4.sort_values(by=['CLIENTE','CODMES','MARCA','CUPO']).reset_index(drop=True)
print(f'Dimensión de nuestra base general: {df4.shape}')

# Cantidad  de veces que accedio a la promocion
df5 = df4.copy()
df5['NCOMPRAS'] = df5[['TARGET_1','TARGET_2','TARGET_3','TARGET_4','TARGET_5','TARGET_6']].sum(axis=1)
df5['NCOMPRAS2'] = (df5['NCOMPRAS'] > 0) * 1

### Split de Entrenamiento y Testeo

# Se observo que la data de entrenamiento es toda la que tenemos en las promociones actuales hasta 201908 y la de testeo es la
# de 201909
train = df5[df5['CODMES']!=201909].reset_index(drop=True)
test  = df5[df5['CODMES']==201909].reset_index(drop=True)

# Caracteristicas que se usarán
features=[ x for x in df4.columns if x not in ['TARGET_sum','TARGET_size','TARGET_nunique','CLIENTE','CODMES','MARCA','CUPO','TARGET',
                                               'FECHA_DESDE','FECHA_HASTA','FECHAALTACLIENTE','CODMES_ALTA',
                                               'ENVASE_CLASE_1_1','ENVASE_CLASE_1_2','ENVASE_CLASE_1_3','ENVASE_CLASE_1_4','ENVASE_CLASE_1_5','ENVASE_CLASE_1_6',
                                               'ENVASE_CLASE_2_1','ENVASE_CLASE_2_2','ENVASE_CLASE_2_3','ENVASE_CLASE_2_4','ENVASE_CLASE_2_5','ENVASE_CLASE_2_6',
                                               'SEGMENTO_1_1','SEGMENTO_1_2','SEGMENTO_1_3','SEGMENTO_1_4','SEGMENTO_1_5','SEGMENTO_1_6',
                                               'SEGMENTO_2_1','SEGMENTO_2_2','SEGMENTO_2_3','SEGMENTO_2_4','SEGMENTO_2_5','SEGMENTO_2_6',
                                               'SEGMENTO_3_1','SEGMENTO_3_2','SEGMENTO_3_3','SEGMENTO_3_4','SEGMENTO_3_5','SEGMENTO_3_6',
                                               'SEGMENTO_4_1','SEGMENTO_4_2','SEGMENTO_4_3','SEGMENTO_4_4','SEGMENTO_4_5','SEGMENTO_4_6',
                                               'VAR_BR_HL_1','VAR_BR_HL_2','VAR_BR_HL_3','VAR_BR_HL_4','VAR_BR_HL_5','VAR_BR_HL_6',
                                               'SEGMENTOPRECIO_1', 'SEGMENTOPRECIO_2','SEGMENTOPRECIO_3', 'SEGMENTOPRECIO_4', 'SEGMENTOPRECIO_5','SEGMENTOPRECIO_6']]

categorical = ['GERENCIA','SUBCANAL','MESES_CONTINUOS','TIPOPOBLACION','ESTRATO','EF','REGION']
cat_ind=[features.index(x) for x in categorical if x in features]

target='TARGET'

### Entrenamiento

# parametros usados
params = {  'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {  'binary', 'auc'}, # 'auc' ,
            "max_depth":8,
            "num_leaves":64,
            'learning_rate': 0.001,
            "min_child_samples": 100,
            'feature_fraction': 0.5,
            "bagging_freq":1,
            'bagging_fraction': 0.9,
            # "scale_pos_weight":30,
            'is_unbalance':True,
            "lambda_l1":1,
            "lambda_l2":1,
            'verbose': 1    
         }

model, importancias, test = model_training(train, test, features, cat_ind, target, params, 5)

w=[x for x in importancias.columns if 'gain_' in x]
importancias['gain-avg']=importancias[w].mean(axis=1)
importancias=importancias.sort_values('gain-avg',ascending=False).reset_index(drop=True)
importancias

importancias

# Testeo
test['EJECUTO_PROMO'] = test[[k for k in test.columns if 'FOLD' in k]].mean(axis=1)

now = datetime.now()
timestamp = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) 

dfpred = test[['CLIENTE','MARCA','CUPO','EJECUTO_PROMO']].\
      rename(columns={'CLIENTE':'Cliente','MARCA':'Marca','CUPO':'Cupo','EJECUTO_PROMO':'Ejecuto_Promo'})
dfpred.to_csv(f'predictions/predictions_{timestamp}.csv',index=False)
print(f'predictions_{timestamp}.csv')

### Prueba usando CATBOOST

train = df4[df4['CODMES']!=201909].reset_index(drop=True)
test  = df4[df4['CODMES']==201909].reset_index(drop=True)

features=[ x for x in df4.columns if x not in ['TARGET_sum','TARGET_nunique','CLIENTE','CODMES','MARCA','CUPO','TARGET',
                                               'FECHA_DESDE','FECHA_HASTA','FECHAALTACLIENTE','CODMES_ALTA',
                                               'ENVASE_CLASE_1_1','ENVASE_CLASE_1_2','ENVASE_CLASE_1_3','ENVASE_CLASE_1_4','ENVASE_CLASE_1_5','ENVASE_CLASE_1_6',
                                               'ENVASE_CLASE_2_1','ENVASE_CLASE_2_2','ENVASE_CLASE_2_3','ENVASE_CLASE_2_4','ENVASE_CLASE_2_5','ENVASE_CLASE_2_6',
                                               'SEGMENTO_1_1','SEGMENTO_1_2','SEGMENTO_1_3','SEGMENTO_1_4','SEGMENTO_1_5','SEGMENTO_1_6',
                                               'SEGMENTO_2_1','SEGMENTO_2_2','SEGMENTO_2_3','SEGMENTO_2_4','SEGMENTO_2_5','SEGMENTO_2_6',
                                               'SEGMENTO_3_1','SEGMENTO_3_2','SEGMENTO_3_3','SEGMENTO_3_4','SEGMENTO_3_5','SEGMENTO_3_6',
                                               'SEGMENTO_4_1','SEGMENTO_4_2','SEGMENTO_4_3','SEGMENTO_4_4','SEGMENTO_4_5','SEGMENTO_4_6',
                                               'VAR_BR_HL_1','VAR_BR_HL_2','VAR_BR_HL_3','VAR_BR_HL_4','VAR_BR_HL_5','VAR_BR_HL_6',
                                               'SEGMENTOPRECIO_1', 'SEGMENTOPRECIO_2','SEGMENTOPRECIO_3', 'SEGMENTOPRECIO_4', 'SEGMENTOPRECIO_5','SEGMENTOPRECIO_6']]




features2 = ['CLIENTE','MARCA','CUPO'] + features

categorical = ['GERENCIA','SUBCANAL','MESES_CONTINUOS','TIPOPOBLACION','ESTRATO','EF','REGION']
cat_ind=[features.index(x) for x in categorical if x in features]

target='TARGET'

# Uso de la libreria
kf_previo=StratifiedKFold(n_splits=5,random_state=256,shuffle=True)

i=1
r=[]

importancias=[]

for train_index,test_index in kf_previo.split(train,train[target]):

    model=CatBoostClassifier(iterations=30000, depth=8, learning_rate=0.05,l2_leaf_reg=8, rsm=0.95,bootstrap_type="Bernoulli",subsample=0.9,eval_metric='AUC',use_best_model=True,early_stopping_rounds=50,verbose=25,cat_features=cat_ind)
    model.fit(train.loc[train_index,features], train.loc[train_index,target], eval_set=(train.loc[test_index,features] ,train.loc[test_index,target]),plot=False)
    print ("Fold_"+str(i))
    a= (roc_auc_score(train.loc[test_index,target],model.predict_proba(train.loc[test_index,features].values)[:,1]))
    train.loc[test_index,"probabilidad"]=model.predict_proba(train.loc[test_index,features].values)[:,1]
    test["TARGET_FOLD"+str(i)]=model.predict_proba(test[features].values)[:,1]
    
    r.append(a)
    print (a)
    print ("")
    
    i=i+1

print ("mean: "+str(np.mean(np.array(r))))
print ("std: "+str(np.std(np.array(r))))

test['EJECUTO_PROMO'] = test[[k for k in test.columns if 'FOLD' in k]].mean(axis=1)

now = datetime.now()
timestamp = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) 

dfpred = test[['CLIENTE','MARCA','CUPO','EJECUTO_PROMO']].\
      rename(columns={'CLIENTE':'Cliente','MARCA':'Marca','CUPO':'Cupo','EJECUTO_PROMO':'Ejecuto_Promo'})
dfpred.to_csv(f'predictions/predictions_{timestamp}.csv',index=False)
print(f'predictions_{timestamp}.csv')

### Prueba usando aprendizaje continuo

df5 = df5.sort_values('CODMES')
x = df5['CODMES'].unique()

z = {}
for i in range(0,len(x)):
    z[i] = df5.loc[df5['CODMES']==x[i]].reset_index(drop=True)
    print(i,x[i])

for i in range(1,len(x)-1):

    X = z[i].loc[z[i][target].notnull(), features]
    y = z[i].loc[z[i][target].notnull(), target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    train_set = lgb.Dataset(X_train, y_train)
    validation_sets = lgb.Dataset(X_test, y_test, reference=train_set)

    model = lgb.train(
            params,
            train_set,
            num_boost_round=20000,
            valid_sets=validation_sets,
            categorical_feature=index_categorical,
            early_stopping_rounds=50,
            verbose_eval=100,
            init_model = model
            )
    
    z[13]['predict_{}'.format(i)] = model.predict(z[13][features], num_iteration=model.best_iteration) 
    
    print('TERMINADO',i)

test['EJECUTO_PROMO'] = model.predict(z[13][features], num_iteration=model.best_iteration)

from datetime import datetime
now = datetime.now()
timestamp = str(now.year) + str(now.month) + str(now.day) + str(now.hour) + str(now.minute) 

dfpred = test[['CLIENTE','MARCA','CUPO','EJECUTO_PROMO']].\
      rename(columns={'CLIENTE':'Cliente','MARCA':'Marca','CUPO':'Cupo','EJECUTO_PROMO':'Ejecuto_Promo'})
dfpred.to_csv(f'predictions/predictions_{timestamp}.csv',index=False)
print(f'predictions_{timestamp}.csv')
