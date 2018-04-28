
# coding: utf-8

# ## Data Cleaning

# Before starting with analyzing the data, it is important for us to clean our dataset.

# Importing necessary packages 

# In[1]:


import pandas as pd
import numpy as np


# Importing dataset.

# In[2]:


df=pd.read_csv('/home/ec2-user/airflow/dags/liver/liver_disease.csv')

df['alkphos'] = df['alkphos'].fillna(df['alkphos'].mean(axis=0))
df = pd.get_dummies(df,prefix=['gender','patient'],columns=['gender','is_patient'])
df.rename(columns={'patient_2':'is_patient'}, inplace=True)
df.drop(['gender_Female','patient_1'],axis=1,inplace=True)
df.to_csv('/home/ec2-user/airflow/dags/liver/liver.csv' , index = False)


# Saving the cleaned data in a new `csv` file.



# Conclusion: We have handled all the missing data from our dataset and we have also removed all the string values and converted it to numeric data which can make our data analysis easy.



