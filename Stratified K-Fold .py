#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.model_selection import StratifiedKFold
import numpy as np


# In[3]:


X = np.zeros((10, 1))  # dummy features
y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])  # labels


# In[4]:


skf = StratifiedKFold(n_splits=2)


# In[5]:


for fold, (train_index, test_index) in enumerate(skf.split(X, y), 1):
    print(f"Fold {fold}")
    print("Train:", y[train_index])
    print("Test: ", y[test_index])
    print()


# In[ ]:




