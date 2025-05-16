#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import TimeSeriesSplit
import numpy as np


# In[2]:


# Sample time series data
data = np.arange(10)  # imagine it's day 0 to 9


# In[3]:


# Define the time series splitter
tscv = TimeSeriesSplit(n_splits=3)


# In[6]:


# Visualize the splits
for fold, (train_index, test_index) in enumerate(tscv.split(data), 1):
    print(f"Fold {fold}")
    print("Train indices:", train_index, "→", data[train_index])
    print("Test indices: ", test_index, "→", data[test_index])
    print("-" * 40)


# In[ ]:




