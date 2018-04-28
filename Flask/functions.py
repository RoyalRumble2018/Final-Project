import pandas as pd
import numpy as np
import pickle
import urllib.request
import shutil
import logging 
import boto.s3
import sys
from boto.s3.key import Key
import imblearn
from imblearn.pipeline import make_pipeline as make_pipeline_imbfinal 
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler

def aws_connect(ak,sak):
    if not ak or not sak:
        logging.warning('Access Key and Secret Access Key not provided!!')
        print('Access Key and Secret Access Key not provided!!')
        exit()
    
    AWS_ACCESS_KEY_ID = ak
    AWS_SECRET_ACCESS_KEY = sak
    
    try:
        conn = boto.connect_s3(AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY)
        print(AWS_ACCESS_KEY_ID,',',AWS_SECRET_ACCESS_KEY)  
        print("Connected to S3")
        #bucket_get('modelsads','Models','Models.zip')
        return conn
    except:
        logging.info("Amazon keys are invalid!!")
        print("Amazon keys are invalid!!")
        exit()

def bucket_get(a,b,c,conn):
    try:
        bucket1 = conn.get_bucket(a)
        srcFileName = b
        k = Key(bucket1,srcFileName)
        #Get the contents of the key into a file 
        k.get_contents_to_filename(c)

    except:
        logging.info("getting AWS Buckect error!!",a)
        print("getting AWS bucket error!!",a)
        exit()

def percent_cb(complete, total):
    sys.stdout.write('.')
    sys.stdout.flush()

def bucket_set(a,b,c,conn):
    #try:
    bucket = conn.get_bucket(a)
    k1 = Key(bucket)
    print(a,' ',b,' ',c)
    k1.key=b
    #Get the contents of the key into a file
    k1.set_contents_from_filename(c,cb=percent_cb, num_cb=10)
    print("Zip File successfully uploaded to S3")
    #except:
    #    logging.info("setting AWS Buckect error!!",a)
    #    print("setting AWS bucket error!!",a)
    #    exit()

def credentials():
    credentials = {}
    with open('Usernames.txt', 'r') as f:
    	for line in f:
      	  user, pwd, url = line.strip().split(';')
      	  lst=[pwd,url]
      	  credentials[user] = lst
    	return credentials

def read_file(path):
    df=pd.read_csv(path)
    return df

def col_selected():
    column_list1=['pcv','hemo','sg','sc','rc','al','DM_yes','bgr','sod','HTN_yes','bu']
    return column_list1

def col_selected1():
    column_list1=['age','tot_bilirubin','direct_bilirubin','tot_proteins','albumin','ag_ratio','sgpt','sgot','alkphos','gender_Male']
    return column_list1

def col_selected2():
    column_list1=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    return column_list1
    
def zip_file(a):
    shutil.make_archive(a,'zip',a)

def unzip_file(a,b):
    shutil.unpack_archive(a,extract_dir=b)

def del_directory(a):
    shutil.rmtree(a)

def ckd(x):
    if x==1:
       return 'ckd'
    else:
       return 'nckd'

def liver(x):
    if x==1:
       return 'Yes'
    else:
       return 'No'

def form_output(dff,ff):
    rfc=0
    sdg=0
    log=0
    knn=0
    svc=0
    bnb=0
    gnb=0
    y=[]
    if (ff =='Kidney'):
       x=pd.DataFrame(data=[dff],columns=col_selected())
    else:
       if ff =='Liver':
          x=pd.DataFrame(data=[dff],columns=col_selected1())
       else:
          x=pd.DataFrame(data=[dff],columns=col_selected2())
    #x.columns=col_selected()
    scaler = StandardScaler()
    scaler.fit(x)
    x_test_sc=scaler.transform(x)
    print("fffff",x.shape)
    #filename = 'Models/RFC_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/RFC_model.sav'
    else:
       if ff =='Liver':
          filename = 'Models1/RFC_model.sav'
       else:
          filename = 'Models2/RFC_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    rfc=mod.predict(x_test_sc)
    #filename = 'Models/SGD_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/SGD_model.sav'
    else:
       if ff =='Liver':
          filename = 'Models1/SGD_model.sav'
       else:
          filename = 'Models2/SGD_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    sdg=mod.predict(x_test_sc)
    #filename = 'Models/logReg_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/logReg_model.sav'
    else:
       if ff =='Liver':
          filename = 'Models1/logReg_model.sav'
       else:
          filename = 'Models2/logReg_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    log=mod.predict(x_test_sc)
    #filename = 'Models/SVC_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/SVC_model.sav'
    else:
       if ff =='Liver':
          filename = 'Models1/SVC_model.sav'
       else:
          filename = 'Models2/SVC_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    svc=mod.predict(x_test_sc)
    #filename = 'Models/KNN_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/KNN_model.sav'
    else:
       if ff =='Liver':
          filename = 'Models1/KNN_model.sav'
       else:
          filename = 'Models2/KNN_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    knn=mod.predict(x_test_sc)
    #filename = 'Models/BernoulliNB_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/BernoulliNB_model.sav'
    else:
       if ff =='Liver':
          filename = 'Models1/BernoulliNB_model.sav'
       else:
          filename = 'Models2/BernoulliNB_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    bnb=mod.predict(x_test_sc)
    #filename = 'Models/GNB_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/GNB_model.sav'
    else:
       if ff =='Liver':
          filename = 'Models1/GNB_model.sav'
       else:
          filename = 'Models2/GNB_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    gnb=mod.predict(x_test_sc)
    y=[rfc,sdg,log,svc,knn,bnb,gnb]
    if (ff =='Kidney'):
       yy=map(lambda n: 'ckd' if n==1 else 'nckd',y)
    else:
       yy=map(lambda n: 'Yes' if n==1 else 'No',y)
    df=pd.DataFrame(data=[yy],columns=['Random Forest Classifier','Stochastic Gradient Decent','Logistic Regression','SVC  ','KNN   ','BNB  ','GNB   '])
    return x,df

def model_run(df,ff):
    #df.columns=header_col()
    print (df.shape)
    scaler = StandardScaler()
    #x=df[col_selected()]
    if (ff =='Kidney'):
       t='Classification_ckd'
       x=df[col_selected()]
    else:
       if ff=='Liver':
          t='is_patient'
          x=df[col_selected1()]
       else:
          t='Outcome'
          x=df[col_selected2()]
    print(x.head())
    scaler.fit(x)
    x_test_sc=scaler.transform(x)
    print(x.shape)
    d1=df.copy()
    #filename = 'Models/logReg_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/logReg_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/logReg_model.sav'
       else:
          filename = 'Models2/logReg_model.sav'
    
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    d1[t]=mod.predict(x_test_sc)
    #d1[t]=d1[t].map(ckd)
    if (ff =='Kidney'):
       d1[t]=d1[t].map(ckd)
       d1.to_csv('Output/Logistic_Reg.csv',sep=',',index=False)
    else:
       d1[t]=d1[t].map(liver)
       if ff=='Liver':
          d1.to_csv('Output1/Logistic_Reg.csv',sep=',',index=False)
       else:
          d1.to_csv('Output2/Logistic_Reg.csv',sep=',',index=False)
    d2=df.copy()
    #filename = 'Models/BernoulliNB_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/BernoulliNB_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/BernoulliNB_model.sav'
       else:
          filename = 'Models2/BernoulliNB_model.sav'

    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    d2[t]=mod.predict(x_test_sc)
    if (ff =='Kidney'):
       d2[t]=d2[t].map(ckd)
       d2.to_csv('Output/BernoulliNB.csv',sep=',',index=False)
    else:
       d2[t]=d2[t].map(liver)
       if ff=='Liver':
          d2.to_csv('Output1/BernoulliNB.csv',sep=',',index=False)
       else:
          d2.to_csv('Output2/BernoulliNB.csv',sep=',',index=False)
    d3=df.copy()
    #filename = 'Models/RFC_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/RFC_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/RFC_model.sav'
       else:
          filename = 'Models2/RFC_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    d3[t]=mod.predict(x_test_sc)
    if (ff =='Kidney'):
       d3[t]=d3[t].map(ckd)
       d3.to_csv('Output/RFC.csv',sep=',',index=False)
    else:
       d3[t]=d3[t].map(liver)
       if ff=='Liver':
          d3.to_csv('Output1/RFC.csv',sep=',',index=False)
       else:
          d3.to_csv('Output2/RFC.csv',sep=',',index=False)
    d4=df.copy()
    #filename = 'Models/KNN_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/KNN_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/KNN_model.sav'
       else:
          filename = 'Models2/KNN_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    d4[t]=mod.predict(x_test_sc)
    if (ff =='Kidney'):
       d4[t]=d4[t].map(ckd)
       d4.to_csv('Output/KNN.csv',sep=',',index=False)
    else:
       d4[t]=d4[t].map(liver)
       if ff=='Liver':
          d4.to_csv('Output1/KNN.csv',sep=',',index=False)
       else:
          d4.to_csv('Output2/KNN.csv',sep=',',index=False)
    d5=df.copy()
    #filename = 'Models/SVC_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/SVC_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/SVC_model.sav'
       else:
          filename = 'Models2/SVC_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    d5[t]=mod.predict(x_test_sc)
    if (ff =='Kidney'):
       d5[t]=d5[t].map(ckd)
       d5.to_csv('Output/SVC.csv',sep=',',index=False)
    else:
       d5[t]=d5[t].map(liver)
       if ff=='Liver':
          d5.to_csv('Output1/SVC.csv',sep=',',index=False)
       else:
          d5.to_csv('Output2/SVC.csv',sep=',',index=False)
    d7=df.copy()
    #filename = 'Models/SGD_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/SGD_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/SGD_model.sav'
       else:
          filename = 'Models2/SGD_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    d7[t]=mod.predict(x_test_sc)
    if (ff =='Kidney'):
       d7[t]=d7[t].map(ckd)
       d7.to_csv('Output/SGD.csv',sep=',',index=False)
    else:
       d7[t]=d7[t].map(liver)
       if ff=='Liver':
          d7.to_csv('Output1/SGD.csv',sep=',',index=False)
       else:
          d7.to_csv('Output2/SGD.csv',sep=',',index=False)
    d8=df.copy()
    #filename = 'Models/GNB_model.sav'
    if (ff =='Kidney'):
       filename = 'Models/GNB_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/GNB_model.sav'
       else:
          filename = 'Models2/GNB_model.sav'
    mod = pickle.load(open(filename, 'rb'),encoding='latin1')
    d8[t]=mod.predict(x_test_sc)
    if (ff =='Kidney'):
       d8[t]=d8[t].map(ckd)
       d8.to_csv('Output/GNB.csv',sep=',',index=False)
    else:
       d8[t]=d8[t].map(liver)
       if ff=='Liver':
          d8.to_csv('Output1/GNB.csv',sep=',',index=False)
       else:
          d8.to_csv('Output2/GNB.csv',sep=',',index=False)
    if (ff =='Kidney'):
       d6=pd.read_csv("Models/Summary.csv")
       zip_file('Output')
    else:
       if ff=='Liver':
          d6=pd.read_csv("Models1/Summary.csv")
          zip_file('Output1')
       else:
          d6=pd.read_csv("Models2/Summary.csv")
          zip_file('Output2')
    return d3,d6

def clean_data(df):
    df.replace(to_replace="\t?",value=np.nan,inplace=True)
    df.replace(to_replace=" ",value=np.nan,inplace=True)
    
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
    
    df['rc'] = df['rc'].astype('float64') 
    df['pcv'] = df['pcv'].astype('float64')
    df['wc'] = df['wc'].astype('int64')
    
    df['age'] = df['age'].fillna(df['age'].mean(axis=0))
    df['bgr'] = df['bgr'].fillna(df['bgr'].mean(axis=0))
    df['sc'] = df['sc'].fillna(df['sc'].mean(axis=0))
    df['bu'] = df['bu'].fillna(df['bu'].mean(axis=0))
    df['sod'] = df['sod'].fillna(df['sod'].mean(axis=0))
    df['pot'] = df['pot'].fillna(df['pot'].mean(axis=0))
    df['pcv'] = df['pcv'].fillna(df['pcv'].mean(axis=0))
    df['hemo'] = df['hemo'].fillna(df['hemo'].mean(axis=0))
    df['rc'] = df['rc'].fillna(df['rc'].mean(axis=0))
    
    df.age = df.age.round()
    df.bu = df.bu.round()
    df.sod = df.sod.round()
    df.pcv = df.pcv.round()
    
    df=pd.get_dummies(df,prefix=['RBC','PC','PCC' , 'BA' , 'HTN' , 'DM' , 'CAD' , 'Appet' , 'PE' , 'Ane' , 'Classification'],columns=['rbc','pc','pcc' , 'ba' , 'htn' , 'dm' , 'cad' , 'appet' , 'pe' , 'ane' , 'classification'])
    
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
    
    df.to_csv('ckd.csv' , index = False)
    return df

def clean_run_data(df):
    df.replace(to_replace="\t?",value=np.nan,inplace=True)
    df.replace(to_replace=" ",value=np.nan,inplace=True)
    
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
    
    #classification = pd.DataFrame(df.groupby('classification').size()).idxmax()[0]
    #df['classification'] = df['classification'].fillna(classification)
    
    df['rc'] = df['rc'].astype('float64') 
    df['pcv'] = df['pcv'].astype('float64')
    df['wc'] = df['wc'].astype('int64')
    
    df['age'] = df['age'].fillna(df['age'].mean(axis=0))
    df['bgr'] = df['bgr'].fillna(df['bgr'].mean(axis=0))
    df['sc'] = df['sc'].fillna(df['sc'].mean(axis=0))
    df['bu'] = df['bu'].fillna(df['bu'].mean(axis=0))
    df['sod'] = df['sod'].fillna(df['sod'].mean(axis=0))
    df['pot'] = df['pot'].fillna(df['pot'].mean(axis=0))
    df['pcv'] = df['pcv'].fillna(df['pcv'].mean(axis=0))
    df['hemo'] = df['hemo'].fillna(df['hemo'].mean(axis=0))
    df['rc'] = df['rc'].fillna(df['rc'].mean(axis=0))
    
    df.age = df.age.round()
    df.bu = df.bu.round()
    df.sod = df.sod.round()
    df.pcv = df.pcv.round()
    
    df=pd.get_dummies(df,prefix=['RBC','PC','PCC' , 'BA' , 'HTN' , 'DM' , 'CAD' , 'Appet' , 'PE' , 'Ane' ],columns=['rbc','pc','pcc' , 'ba' , 'htn' , 'dm' , 'cad' , 'appet' , 'pe' , 'ane'])
    
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
    #df.drop(['Classification_notckd'],axis=1,inplace=True)
    #df.drop(['id'],axis=1,inplace=True)
    
    df.to_csv('ckd.csv' , index = False)
    return df


def model_build(df,ff):
    scaler = StandardScaler()
    df_train,df_test = train_test_split(df,train_size=0.7,random_state=42)
    if (ff =='Kidney'):
       t="Classification_ckd"
       x_train=df_train[col_selected()]
       x_test=df_test[col_selected()]
    else:
       if ff=='Liver':
          t="is_patient"
          x_train=df_train[col_selected1()]
          x_test=df_test[col_selected1()]
       else:
          t="Outcome"
          x_train=df_train[col_selected2()]
          x_test=df_test[col_selected2()]
    print(x_train.shape)
    y_train=df_train[t]
    scaler.fit(x_train)
    x_train_sc=scaler.transform(x_train)
    #x_test=df_test[col_selected()]
    print(x_test.shape)
    y_test=df_test[t]
    scaler.fit(x_test)
    x_test_sc=scaler.transform(x_test)
    accuracy =[]
    model_name =[]
    dataset=[]
    f1score = []
    precision = []
    recall = []
    true_positive =[]
    false_positive =[]
    true_negative =[]
    false_negative =[]
    
    rfc = RandomForestClassifier(n_estimators=50,random_state=0)
    rfc.fit(x_train_sc, y_train)
    if (ff =='Kidney'):
       filename = 'Models/RFC_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/RFC_model.sav'
       else:
          filename = 'Models2/RFC_model.sav'

    pickle.dump(rfc,open(filename,'wb'))
    model_name.append('Random Forest Classifier')
    dataset.append('Training')
    error_metrix(rfc,x_train_sc,y_train,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    model_name.append('Random Forest Classifier')
    dataset.append('Testing')
    error_metrix(rfc,x_test_sc,y_test,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    
    sgd = SGDClassifier(loss = 'modified_huber' , shuffle = True , random_state = 42)
    sgd.fit(x_train_sc, y_train)
    if (ff =='Kidney'):
       filename = 'Models/SGD_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/SGD_model.sav'
       else:
          filename = 'Models2/SGD_model.sav'

    pickle.dump(sgd,open(filename,'wb'))
    model_name.append('Stochastic Gradient Decent')
    dataset.append('Training')
    error_metrix(sgd,x_train_sc,y_train,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    model_name.append('Stochastic Gradient Decent')
    dataset.append('Testing')
    error_metrix(sgd,x_test_sc,y_test,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)

    knn = KNeighborsClassifier(n_neighbors = 3)
    knn.fit(x_train_sc, y_train)
    if (ff =='Kidney'):
       filename = 'Models/KNN_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/KNN_model.sav'
       else:
          filename = 'Models2/KNN_model.sav'
    pickle.dump(knn,open(filename,'wb'))
    model_name.append('k nearest Neighbour')
    dataset.append('Training')
    error_metrix(knn,x_train_sc,y_train,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    model_name.append('k nearest Neighbour')
    dataset.append('Testing')
    error_metrix(knn,x_test_sc,y_test,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)

    logreg = LogisticRegression()
    logreg.fit(x_train_sc, y_train)
    if (ff =='Kidney'):
       filename = 'Models/logReg_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/logReg_model.sav'
       else:
          filename = 'Models2/logReg_model.sav'
    
    pickle.dump(logreg,open(filename,'wb'))
    model_name.append('Logistic Regression')
    dataset.append('Training')
    error_metrix(logreg,x_train_sc,y_train,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    model_name.append('Logistic Regression')
    dataset.append('Testing')
    error_metrix(logreg,x_test_sc,y_test,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    
    svc = SVC(kernel = 'linear' , C = 0.025 , random_state = 42)
    svc.fit(x_train_sc, y_train)
    if (ff =='Kidney'):
       filename = 'Models/SVC_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/SVC_model.sav'
       else:
          filename = 'Models2/SVC_model.sav'

    pickle.dump(svc,open(filename,'wb'))
    model_name.append('Support Vector Classifier')
    dataset.append('Training')
    error_metrix(svc,x_train_sc,y_train,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    model_name.append('Support Vector Classifier')
    dataset.append('Testing')
    error_metrix(svc,x_test_sc,y_test,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    
    
    NB = BernoulliNB()
    NB.fit(x_train_sc, y_train)
    if (ff =='Kidney'):
       filename = 'Models/BernoulliNB_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/BernoulliNB_model.sav'
       else:
          filename = 'Models2/BernoulliNB_model.sav'

    pickle.dump(NB,open(filename,'wb'))
    model_name.append('Bernoulli Naive Bayes')
    dataset.append('Training')
    error_metrix(NB,x_train_sc,y_train,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    model_name.append('Bernoulli Naive Bayes')
    dataset.append('Testing')
    error_metrix(NB,x_test_sc,y_test,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    
    gnb = GaussianNB()
    gnb.fit(x_train_sc, y_train)
    if (ff =='Kidney'):
       filename = 'Models/GNB_model.sav'
    else:
       if ff=='Liver':
          filename = 'Models1/GNB_model.sav'
       else:
          filename = 'Models2/GNB_model.sav'

    pickle.dump(gnb,open(filename,'wb'))
    model_name.append('Gradient Naive Bayes')
    dataset.append('Training')
    error_metrix(gnb,x_train_sc,y_train,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    model_name.append('Gradient Naive Bayes')
    dataset.append('Testing')
    error_metrix(gnb,x_test_sc,y_test,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative)
    
    df_summ=summary(model_name,dataset,accuracy,f1score,precision,recall ,true_positive,false_positive,true_negative,false_negative,ff)

    if (ff =='Kidney'):
       zip_file('Models')
    else:
       if ff=='Liver':
          zip_file('Models1')
       else:
          zip_file('Models2')
    return df_summ

def error_metrix(a,x,y,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative):
    prediction = a.predict(x)
    f1 = f1_score(y, prediction)
    p = precision_score(y, prediction)
    r = recall_score(y, prediction)
    a = accuracy_score(y, prediction)
    cm = confusion_matrix(y, prediction)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[1][1]
    f1score.append(f1)
    precision.append(p)
    recall.append(r)
    accuracy.append(a)
    true_positive.append(tp) 
    false_positive.append(fp)
    true_negative.append(tn) 
    false_negative.append(fn)

def summary(model_name,dataset,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative,ff):
    Summary = model_name,dataset,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative
    describe1 = pd.DataFrame(Summary[0],columns = {"Model_Name             "})
    describe2 = pd.DataFrame(Summary[1],columns = {"Dataset"})
    describe3 = pd.DataFrame(Summary[2],columns = {"F1_score"})
    describe4 = pd.DataFrame(Summary[3],columns = {"Precision_score"})
    describe5 = pd.DataFrame(Summary[4],columns = {"Recall_score"})
    describe6 = pd.DataFrame(Summary[5], columns ={"Accuracy_score"})
    describe7 = pd.DataFrame(Summary[6], columns ={"True_Positive"})
    describe8 = pd.DataFrame(Summary[7], columns ={"False_Positive"})
    describe9 = pd.DataFrame(Summary[8], columns ={"True_Negative"})
    describe10 = pd.DataFrame(Summary[9], columns ={"False_Negative"})
    des = describe1.merge(describe2, left_index=True, right_index=True, how='inner')
    des = des.merge(describe3,left_index=True, right_index=True, how='inner')
    des = des.merge(describe4,left_index=True, right_index=True, how='inner')
    des = des.merge(describe5,left_index=True, right_index=True, how='inner')
    des = des.merge(describe6,left_index=True, right_index=True, how='inner')
    des = des.merge(describe7,left_index=True, right_index=True, how='inner')
    des = des.merge(describe8,left_index=True, right_index=True, how='inner')
    des = des.merge(describe9,left_index=True, right_index=True, how='inner')
    des = des.merge(describe10,left_index=True, right_index=True, how='inner')
    Summary_csv = des
    if ff=='Kidney':
       Summary_csv.to_csv('Models/Summary.csv')
    else:
       if ff=='Liver':
          Summary_csv.to_csv('Models1/Summary.csv')
       else:
          Summary_csv.to_csv('Models2/Summary.csv')
    return Summary_csv

def clean_data_liver(df):
    df['alkphos'] = round(df['alkphos'].fillna(df['alkphos'].mean(axis=0)),2)
    df=pd.get_dummies(df,prefix=['gender','patient'],columns=['gender','is_patient'])
    df.rename(columns={'patient_2':'is_patient'}, inplace=True)
    df.drop(['gender_Female','patient_1'],axis=1,inplace=True)
    df.to_csv('liver.csv' , index = False)
    return df 

def clean_data_liver_run(df):
    df['alkphos'] = round(df['alkphos'].fillna(df['alkphos'].mean(axis=0)),2)
    df=pd.get_dummies(df,prefix=['gender'],columns=['gender'])
    df.drop(['gender_Female'],axis=1,inplace=True)
    df.to_csv('liver.csv' , index = False)
    return df 
#def clean_data_diab(df):
    #df.replace(to_replace="\t?",value=np.nan,inplace=True)





#def clean_data_diab_run(df):
