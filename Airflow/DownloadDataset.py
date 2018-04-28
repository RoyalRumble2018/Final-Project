import pandas as pd

url1="https://s3-us-west-2.amazonaws.com/datasetads/kidney_disease.csv"
print("URL")
df =pd.read_csv(url1)
print("In")
df.to_csv('/home/ec2-user/airflow/dags/kidney_disease.csv',encoding='utf-8',index = False)
print("Out")

