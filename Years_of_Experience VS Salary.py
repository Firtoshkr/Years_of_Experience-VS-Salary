#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


# In[33]:


data = pd.read_csv("C:\\Users\\firto\\jupyter\\salary.csv")


# In[34]:


X = np.array(data['YearsExperience'])
y = np.array(data['Salary'])


# In[35]:


plt.scatter(X,y)


# In[36]:


u = np.mean(X)
std = np.std(X)
X = (X-u)/std


# In[37]:


print(X.shape,y.shape)
x = X.reshape((30,1))
y = y.reshape((30,1))
print(x.shape,y.shape)


# In[79]:


def hypothesis(x,theta):
    y_ = theta[1]*x + theta[0]
    return y_
def error(x,y,theta):
    m,n = x.shape
    y_ = hypothesis(x,theta)
    err = np.sum((y_-y)**2)
    return err/m
def gradient(x,y,theta):
    m,n = x.shape
    y_ = hypothesis(x,theta)
    grad = np.zeros((2,))
    grad[0] = np.sum(y_-y)
    grad[1] = np.dot(x.T,y_-y)
    return grad/m
def gradientDescent(x,y,learning_rate = 0.1,epoch = 300):
    m,n = x.shape
    grad = np.zeros((2,))
    theta = np.zeros((2,))
    err = []
    for i in range(epoch):
        er = error(x,y,theta)
        err.append(er)
        grad = gradient(x,y,theta)
        theta = theta - learning_rate * grad
    return err, theta
    


# In[80]:


err , theta = gradientDescent(x,y)
err


# In[81]:


ypred = theta[1]*x + theta[0]
plt.scatter(x,y,c='green')
plt.plot(x,ypred,c='red')


# In[82]:


def r2score():
    Ypred = hypothesis(x,theta)
    num = np.sum((y-Ypred)**2)
    denom = np.sum((y-y.mean())**2)
    score = (1-num/denom)
    return score*100


# In[83]:


r2score()


# In[ ]:




