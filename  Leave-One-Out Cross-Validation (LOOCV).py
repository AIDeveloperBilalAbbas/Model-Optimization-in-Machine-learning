#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import numpy as np


# In[9]:


# Load Iris data
iris = load_iris()


# In[10]:


class_0_idx = np.where(iris.target == 0)[0][:5]
class_1_idx = np.where(iris.target == 1)[0][:5]
selected_idx = np.concatenate([class_0_idx, class_1_idx])


# In[11]:


X = iris.data[selected_idx]
y = iris.target[selected_idx]


# In[12]:


# LOOCV setup
loo = LeaveOneOut()
model = LogisticRegression(max_iter=200)
accuracies = []


# In[6]:


# Track accuracy
accuracies = []


# In[13]:


for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("📊 Final LOOCV Accuracy:", np.mean(accuracies))
    


# In[ ]:




