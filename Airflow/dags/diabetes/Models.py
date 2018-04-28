
# coding: utf-8

# # Algorithms before feature selection

# In[41]:


## Import the required libraries
import pandas as pd
import numpy as np
import imblearn
from imblearn.pipeline import make_pipeline as make_pipeline_imbfinal 
from imblearn.over_sampling import SMOTE
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB 
#from sklearn import svm
from sklearn.metrics import *
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[42]:


df = pd.read_csv('/home/ec2-user/airflow/dags/ckd.csv')



# In[43]:
column_list1=['pcv','hemo','sg','sc','rc','al','DM_yes','bgr','sod','HTN_yes','bu']
df_train,df_test = train_test_split(df,train_size=0.7,random_state=42)
x_train=df_train[column_list1]
y_train=df_train['Classification_ckd']
scaler.fit(x_train)
x_train_sc=scaler.transform(x_train)
x_test=df_test[column_list1]
y_test=df_test['Classification_ckd']
scaler.fit(x_test)
x_test_sc=scaler.transform(x_test)
y_train.head()


# In[44]:


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
## fitiing the model
rfc.fit(x_train_sc, y_train)
filename = '/home/ec2-user/airflow/dags/Models/RFC_model.sav'
pickle.dump(rfc,open(filename,'wb'))
rfc

prediction = rfc.predict(x_train_sc)
f1 = f1_score(y_train, prediction)
p = precision_score(y_train, prediction)
r = recall_score(y_train, prediction)
a = accuracy_score(y_train, prediction)
cm = confusion_matrix(y_train, prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Random Forest Classifier')
dataset.append('Training')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm

prediction = rfc.predict(x_test_sc)
f1 = f1_score(y_test,  prediction)
p = precision_score(y_test,  prediction)
r = recall_score(y_test,  prediction)
a = accuracy_score(y_test,  prediction)
cm = confusion_matrix(y_test,  prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Random Forest Classifier')
dataset.append('Testing')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm

from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier (loss = 'modified_huber' , shuffle = True , random_state = 42)
sgd.fit(x_train_sc, y_train)
filename = '/home/ec2-user/airflow/dags/Models/SGD_model.sav'
pickle.dump(sgd,open(filename,'wb'))
sgd

prediction = sgd.predict(x_train_sc)
f1 = f1_score(y_train, prediction)
p = precision_score(y_train, prediction)
r = recall_score(y_train, prediction)
a = accuracy_score(y_train, prediction)
cm = confusion_matrix(y_train, prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Stochastic Gradient Decent')
dataset.append('Training')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm

prediction = sgd.predict(x_test_sc)
f1 = f1_score(y_test,  prediction)
p = precision_score(y_test,  prediction)
r = recall_score(y_test,  prediction)
a = accuracy_score(y_test,  prediction)
cm = confusion_matrix(y_test,  prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Stochastic Gradient Decent')
dataset.append('Testing')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(x_train_sc, y_train)
filename = '/home/ec2-user/airflow/dags/Models/KNN_model.sav'
pickle.dump(knn,open(filename,'wb'))
knn

prediction = knn.predict(x_train_sc)
f1 = f1_score(y_train, prediction)
p = precision_score(y_train, prediction)
r = recall_score(y_train, prediction)
a = accuracy_score(y_train, prediction)
cm = confusion_matrix(y_train, prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('k nearest Neighbour')
dataset.append('Training')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm

prediction = knn.predict(x_test_sc)
f1 = f1_score(y_test,  prediction)
p = precision_score(y_test,  prediction)
r = recall_score(y_test,  prediction)
a = accuracy_score(y_test,  prediction)
cm = confusion_matrix(y_test,  prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('k nearest Neighbour')
dataset.append('Testing')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm

# # Logistic Regression

# In[45]:


logreg = LogisticRegression()
## fitiing the model
logreg.fit(x_train_sc, y_train)
filename = '/home/ec2-user/airflow/dags/Models/logReg_model.sav'
pickle.dump(logreg,open(filename,'wb'))
logreg


# Training

# In[46]:


prediction = logreg.predict(x_train_sc)
f1 = f1_score(y_train, prediction)
p = precision_score(y_train, prediction)
r = recall_score(y_train, prediction)
a = accuracy_score(y_train, prediction)
cm = confusion_matrix(y_train, prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
dataset.append('Training')
model_name.append('Logistic Regression')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm


# Testing

# In[47]:


prediction = logreg.predict(x_test_sc)
f1 = f1_score(y_test,  prediction)
p = precision_score(y_test,  prediction)
r = recall_score(y_test,  prediction)
a = accuracy_score(y_test,  prediction)
cm = confusion_matrix(y_test,  prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Logistc Regression')
dataset.append('Testing')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm



from sklearn.svm import SVC
svc = SVC (kernel = 'linear' , C = 0.025 , random_state = 42)
svc.fit(x_train_sc, y_train)
filename = '/home/ec2-user/airflow/dags/Models/SVC_model.sav'
pickle.dump(svc,open(filename,'wb'))
svc

prediction = svc.predict(x_train_sc)
f1 = f1_score(y_train, prediction)
p = precision_score(y_train, prediction)
r = recall_score(y_train, prediction)
a = accuracy_score(y_train, prediction)
cm = confusion_matrix(y_train, prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Support Vector Classifier')
dataset.append('Training')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm

prediction = svc.predict(x_test_sc)
f1 = f1_score(y_test,  prediction)
p = precision_score(y_test,  prediction)
r = recall_score(y_test,  prediction)
a = accuracy_score(y_test,  prediction)
cm = confusion_matrix(y_test,  prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Support Vector Classifier')
dataset.append('Testing')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm

# #  Bernoulli Naive Bayes

# In[48]:


NB = BernoulliNB()
## fitiing the model
NB.fit(x_train_sc, y_train)
filename = '/home/ec2-user/airflow/dags/Models/BernoulliNB_model.sav'
pickle.dump(NB,open(filename,'wb'))
NB


# Training

# In[49]:


prediction = NB.predict(x_train_sc)
f1 = f1_score(y_train, prediction)
p = precision_score(y_train, prediction)
r = recall_score(y_train, prediction)
a = accuracy_score(y_train, prediction)
cm = confusion_matrix(y_train, prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Bernoulli Naive Bayes')
dataset.append('Training')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm


# Testing

# In[50]:


prediction = NB.predict(x_test_sc)
f1 = f1_score(y_test,  prediction)
p = precision_score(y_test,  prediction)
r = recall_score(y_test,  prediction)
a = accuracy_score(y_test,  prediction)
cm = confusion_matrix(y_test,  prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Bernoulli Naive Bayes')
dataset.append('Testing')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm



# # Gradient Naive Bayes

# In[63]:


from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train_sc, y_train)
filename = '/home/ec2-user/airflow/dags/Models/GNB_model.sav'
pickle.dump(gnb,open(filename,'wb'))
gnb


# Training

# In[64]:


prediction = gnb.predict(x_train_sc)
f1 = f1_score(y_train, prediction)
p = precision_score(y_train, prediction)
r = recall_score(y_train, prediction)
a = accuracy_score(y_train, prediction)
cm = confusion_matrix(y_train, prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Gradient Naive Bayes')
dataset.append('Training')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm


# Testing

# In[65]:


prediction = gnb.predict(x_test_sc)
f1 = f1_score(y_test,  prediction)
p = precision_score(y_test,  prediction)
r = recall_score(y_test,  prediction)
a = accuracy_score(y_test,  prediction)
cm = confusion_matrix(y_test,  prediction)
tp = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tn = cm[1][1]
model_name.append('Gradient Naive Bayes')
dataset.append('Testing')
f1score.append(f1)
precision.append(p)
recall.append(r)
accuracy.append(a)
true_positive.append(tp) 
false_positive.append(fp)
true_negative.append(tn) 
false_negative.append(fn)
cm


# # Writing summary metrics

# In[66]:


Summary = model_name,dataset,f1score,precision,recall,accuracy,true_positive,false_positive,true_negative,false_negative
#Summary


# In[67]:


## Making a dataframe of the accuracy and error metrics
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
#des = des.merge(describe9,left_index=True, right_index=True, how='inner')
#Summary_csv = des.sort_values(ascending=True,by="False_Negative").reset_index(drop = True)
Summary_csv=des


# In[68]:


Summary_csv.to_csv('/home/ec2-user/airflow/dags/Models/Summary.csv')


# Conclusion: Except for `Bernoulli Naive Bayes` , we are getting 100% accuracy for every other model.

import shutil
shutil.make_archive("/home/ec2-user/airflow/dags/Models",'zip',"/home/ec2-user/airflow/dags/Models")

