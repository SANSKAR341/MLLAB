#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


df=pd.read_csv('train.csv')


# In[6]:


df.head()


# In[7]:


df.head(7)


# In[8]:


df.tail()


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.iloc[3]


# In[12]:


df.loc[0:4,'Ticket']


# In[13]:


df['Ticket'].head()


# In[14]:


df[df.Age>65]


# In[15]:


df[(df.Age==11)&(df.SibSp==5)]


# In[16]:


df[(df.Age==11)|(df.SibSp==5)]


# In[17]:


df['Embarked'].unique()


# In[18]:


df['Age'].unique()


# In[19]:


print(df['Age'].mean())


# In[20]:


print(df['Fare'].median())


# In[21]:


print((df['Sex']=='female').sum())


# In[22]:


df.info()


# In[23]:


df['Age'].head(6)


# In[24]:


newdf=df['Age'].fillna(30)


# In[25]:


newdf.head(6)


# In[26]:


df.isnull().sum()


# In[27]:


df.groupby('Survived')['Age'].mean()


# In[28]:


df.pivot_table(index='Sex',columns='Parch',values='Survived',aggfunc='sum')


# In[29]:


df.pivot_table(index='Sex',columns='SibSp',values='Survived',aggfunc='sum')


# In[30]:


df[df.Survived==0]


# In[31]:


df[(df.Fare<40.000)&(df.Pclass==3)]


# In[32]:


df['Survived'].count()


# In[33]:


print((df['Survived']== 0).sum())


# In[34]:


print((df['Survived']== 1).sum())


# In[36]:


df.groupby(['Sex', 'Survived']).count()


# In[45]:


df[df.PassengerId==674]


# In[ ]:




