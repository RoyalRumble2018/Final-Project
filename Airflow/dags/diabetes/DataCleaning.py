
# coding: utf-8

# ## Data Cleaning

# Before starting with analyzing the data, it is important for us to clean our dataset.

# Importing necessary packages 

# In[1]:


import pandas as pd
import numpy as np


# Importing dataset.

# In[2]:


df=pd.read_csv('/home/ec2-user/airflow/dags/kidney_disease.csv')




# We can see that many colummns contains data which are not in either float or integer format. This can hinder our analysis. So, it is advisable for us to change these to numeric data.




# We found that there are special characters in our dataset. This can hinder our analysis. So, replacing it with `null`

# In[8]:


df.replace(to_replace="\t?",value=np.nan,inplace=True)
df.replace(to_replace=" ",value=np.nan,inplace=True)




# We have to handle missing values. We analyzed our dataste throughly and we can up with few reults.

# 1) In columns like: age, bgr, bu, sc, sod, pot, hemo, pcv and rc , missing values will be replaced with average/mean of that column's data.

# 2) In columns like: rbc, pc, pcc, ba, pcv, wc, rc, htn, dm , cad, appet, pe, ane and classification , missing values will be replaced with most frequently occuring data as these contain string data.

# In[10]:


bp = pd.DataFrame(df.groupby('bp').size()).idxmax()[0]
df['bp'] = df['bp'].fillna(bp)

sg = pd.DataFrame(df.groupby('sg').size()).idxmax()[0]
df['sg'] = df['sg'].fillna(sg)

al = pd.DataFrame(df.groupby('al').size()).idxmax()[0]
df['al'] = df['al'].fillna(al)

su = pd.DataFrame(df.groupby('su').size()).idxmax()[0]
df['su'] = df['su'].fillna(su)

rbc = pd.DataFrame(df.groupby('rbc').size()).idxmax()[0]
df['rbc'] = df['rbc'].fillna(rbc)

pc = pd.DataFrame(df.groupby('pc').size()).idxmax()[0]
df['pc'] = df['pc'].fillna(pc)

pcc = pd.DataFrame(df.groupby('pcc').size()).idxmax()[0]
df['pcc'] = df['pcc'].fillna(pcc)

ba = pd.DataFrame(df.groupby('ba').size()).idxmax()[0]
df['ba'] = df['ba'].fillna(ba)

bgr = pd.DataFrame(df.groupby('bgr').size()).idxmax()[0]
df['bgr'] = df['bgr'].fillna(bgr)

wc = pd.DataFrame(df.groupby('wc').size()).idxmax()[0]
df['wc'] = df['wc'].fillna(wc)

htn = pd.DataFrame(df.groupby('htn').size()).idxmax()[0]
df['htn'] = df['htn'].fillna(htn)

dm = pd.DataFrame(df.groupby('dm').size()).idxmax()[0]
df['dm'] = df['dm'].fillna(dm)

wc = pd.DataFrame(df.groupby('wc').size()).idxmax()[0]
df['wc'] = df['wc'].fillna(wc)

cad = pd.DataFrame(df.groupby('cad').size()).idxmax()[0]
df['cad'] = df['cad'].fillna(cad)

appet = pd.DataFrame(df.groupby('appet').size()).idxmax()[0]
df['appet'] = df['appet'].fillna(appet)

pe = pd.DataFrame(df.groupby('pe').size()).idxmax()[0]
df['pe'] = df['pe'].fillna(pe)

ane = pd.DataFrame(df.groupby('ane').size()).idxmax()[0]
df['ane'] = df['ane'].fillna(ane)

classification = pd.DataFrame(df.groupby('classification').size()).idxmax()[0]
df['classification'] = df['classification'].fillna(classification)


# In[11]:


df['rc'] = df['rc'].astype('float64') 
df['pcv'] = df['pcv'].astype('float64')
df['wc'] = df['wc'].astype('int64')


# In[12]:


df['age'] = df['age'].fillna(df['age'].mean(axis=0))
df['bgr'] = df['bgr'].fillna(df['bgr'].mean(axis=0))
df['sc'] = df['sc'].fillna(df['sc'].mean(axis=0))
df['bu'] = df['bu'].fillna(df['bu'].mean(axis=0))
df['sod'] = df['sod'].fillna(df['sod'].mean(axis=0))
df['pot'] = df['pot'].fillna(df['pot'].mean(axis=0))
df['pcv'] = df['pcv'].fillna(df['pcv'].mean(axis=0))
df['hemo'] = df['hemo'].fillna(df['hemo'].mean(axis=0))
df['rc'] = df['rc'].fillna(df['rc'].mean(axis=0))


# Rounding off to nearest decimal value.

# In[14]:


df.age = df.age.round()
df.bu = df.bu.round()
df.sod = df.sod.round()
df.pcv = df.pcv.round()



# Now, we can see that we have removed all null values. But we still have `string` data in our which has to be removed. For this, we are creating dummy variables. 

# In[17]:


df=pd.get_dummies(df,prefix=['RBC','PC','PCC' , 'BA' , 'HTN' , 'DM' , 'CAD' , 'Appet' , 'PE' , 'Ane' , 'Classification'],columns=['rbc','pc','pcc' , 'ba' , 'htn' , 'dm' , 'cad' , 'appet' , 'pe' , 'ane' , 'classification'])



# Removing all unwanted columns

# In[20]:


df.drop(['RBC_abnormal'],axis=1,inplace=True)
df.drop(['PC_abnormal'],axis=1,inplace=True)
df.drop(['PCC_notpresent'],axis=1,inplace=True)
df.drop(['BA_notpresent'],axis=1,inplace=True)
df.drop(['HTN_no'],axis=1,inplace=True)
df.drop(['DM_no'],axis=1,inplace=True)
df.drop(['CAD_no'],axis=1,inplace=True)
df.drop(['Appet_poor'],axis=1,inplace=True)
df.drop(['PE_no'],axis=1,inplace=True)
df.drop(['Ane_no'],axis=1,inplace=True)
df.drop(['Classification_notckd'],axis=1,inplace=True)
#df.drop(['id'],axis=1,inplace=True)


# Saving the cleaned data in a new `csv` file.

# In[21]:


df.to_csv('/home/ec2-user/airflow/dags/ckd.csv' , index = False)


# Conclusion: We have handled all the missing data from our dataset and we have also removed all the string values and converted it to numeric data which can make our data analysis easy.



