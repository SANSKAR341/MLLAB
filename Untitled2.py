#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split


# In[2]:


df=pd.read_csv("employee_salary.csv")


# In[3]:


df.head()


# In[4]:


X=df.iloc[:, :-1].values 
Y=df.iloc[:, -1].values


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[6]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[7]:


model=linear_model.LinearRegression()
model.fit(X,Y)
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, model.predict(X_train))


# In[ ]:




