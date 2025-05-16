#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# In[2]:


# Load sample dataset
X, y = load_iris(return_X_y=True)


# In[3]:


# Define the model
model = LogisticRegression(max_iter=200)


# In[4]:


# Define repeated K-Fold cross-validator
rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)


# In[5]:


# Store accuracies
accuracies = []


# In[6]:


# Loop through each fold
for fold, (train_index, test_index) in enumerate(rkf.split(X), 1):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {fold}: Accuracy = {acc:.2f}")

# Average accuracy
print("\n✅ Average Accuracy across all folds:", np.mean(accuracies))


# In[ ]:




