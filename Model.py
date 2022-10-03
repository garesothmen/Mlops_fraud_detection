#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import  statsmodels.api as sm
import matplotlib.pyplot as plt
import os


# In[2]:


print(os.listdir("data/"))


# In[3]:


X_train=pd.read_csv("data/X_train.csv",index_col=0)
y_train=pd.read_csv("data/y_train.csv",index_col=0)["is_fraud"]
X_test=pd.read_csv("data/X_test.csv",index_col=0)
y_test=pd.read_csv("data/y_test.csv",index_col=0)["is_fraud"]


# In[4]:


regressor=LogisticRegression()
regressor.fit(X_train,y_train)
regressor.score(X_train,y_train)


# In[5]:


X_train.head()


# In[6]:


y_pred =regressor.predict(X_test)


# In[7]:


conf=confusion_matrix(y_test,y_pred)


# In[8]:


acc=(conf[0][0]+conf[1][1])/sum(conf)
acc


# In[9]:


spec=conf[1][1]/(conf[1][1]+conf[1][0])
spec


# In[10]:


y_pred_prob=regressor.predict_proba(X_test)


# In[11]:


plt.hist(y_pred_prob)


# In[12]:


plt.hist(y_pred_prob[:,0],color="#ab56b6")


# In[13]:


plt.hist(y_pred_prob[:,1],color="red")


# In[14]:


y_pred2=regressor.predict_proba(X_test)[:,1]
fpr,tpr,thr=roc_curve(y_test,y_pred2)
plt.plot(fpr,tpr)


# In[15]:


regressor.score(X_test,y_test)


# In[ ]:




