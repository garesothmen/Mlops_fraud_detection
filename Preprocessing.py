#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn 
import datetime
import seaborn as sn 
import os


# In[2]:


train=pd.read_csv("data/fraudTrain.csv",index_col=0)


# In[3]:


test=pd.read_csv("data//fraudTest.csv",index_col=0)


# In[4]:


train.describe()


# In[5]:


train.shape


# In[6]:


train.head()


# In[7]:


corr=train.corr()


# In[8]:


corr.style.background_gradient(cmap='coolwarm')


# In[9]:


sn.heatmap(corr)


# In[10]:


nu = train.nunique().reset_index()
nu.columns = ['feature','nunique']
ax = sn.barplot(x='feature', y='nunique', data=nu)


# In[11]:


nu


# In[12]:


train.columns


# In[13]:


dict= {}
for c in train.columns:
    dict[c]=train[c].value_counts()


# In[14]:


dict


# In[15]:


def showhist(df):
    for x in df.columns:
        plt.hist(dict[x])
        plt.title(x)
        plt.show()
                   


# In[16]:


showhist(train)


# In[17]:


gtr=train['is_fraud'].groupby([train['category'],train['is_fraud']]).mean()


# In[18]:


colors = sn.color_palette('pastel')
labels=list(gtr.keys())


# In[19]:


values=[gtr[x] for x in labels]


# In[20]:


plt.hist(gtr)
plt.show()


# In[21]:


plt.pie(values, labels =labels, colors = colors, autopct='%.0f%%')
plt.show()


# In[23]:


def trainprerocessing(ctrain):
    df=ctrain
    df['trans_date_trans_time']=df['trans_date_trans_time'].apply(lambda x :datetime.datetime.timestamp(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")) )
    Y=df['is_fraud']
    X=df.drop(['is_fraud','trans_num'],axis=1)
    from sklearn import preprocessing
    X=X.apply(preprocessing.LabelEncoder().fit_transform)
    from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    X, y = oversample.fit_resample(X,Y)
    Xfilepath= 'data/X_train.csv'
    Yfilepath='data/y_train.csv'
    X.to_csv(Xfilepath)
    y.to_csv(Yfilepath)
    


# In[24]:


def testpreprocessing(ctest):
    df=ctest
    df['trans_date_trans_time']=df['trans_date_trans_time'].apply(lambda x :datetime.datetime.timestamp(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")) )
    Y=df['is_fraud']
    X=df.drop(['is_fraud','trans_num'],axis=1)
    from sklearn import preprocessing
    X=X.apply(preprocessing.LabelEncoder().fit_transform)
    Xfilepath= 'data/X_test.csv'
    Yfilepath='data/y_test.csv'
    X.to_csv(Xfilepath)
    Y.to_csv(Yfilepath)


# In[25]:


trainprerocessing(train)


# In[26]:


testpreprocessing(test)


# In[27]:


y_train=pd.read_csv('data/y_train.csv',index_col=0)


# In[22]:


plt.hist(train['is_fraud'])


# In[28]:


plt.hist(y_train['is_fraud'])


# In[29]:


plt.hist(test['is_fraud'])


# In[30]:


y_test=pd.read_csv('data/y_test.csv',index_col=0)


# In[31]:


plt.hist(y_test)


# In[ ]:




