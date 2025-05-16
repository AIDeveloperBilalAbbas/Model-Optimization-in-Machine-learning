#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# In[2]:


# 1. Load some data
data = load_iris()
X = data.data
y = data.target


# In[3]:


# 2. Choose the number of folds
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)


# In[4]:


# 3. Prepare a model (you can use any classifier)
model = LogisticRegression(max_iter=200)


# In[5]:


# 4. Store scores
fold_accuracies = []


# In[6]:


# 5. Run K-Fold Cross-Validation
fold = 1
for train_index, test_index in kf.split(X):
    print(f"\n🔄 Fold {fold}")
    print(f"Train indices: {train_index}")
    print(f"Test indices: {test_index}")

    # Split data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train model
    model.fit(X_train, y_train)
  # Predict
    y_pred = model.predict(X_test)
     # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy for Fold {fold}: {acc:.2f}")

    fold_accuracies.append(acc)
    fold += 1

# 6. Final average accuracy
print("\n📊 Cross-Validation Results:")
print(f"Accuracies for each fold: {np.round(fold_accuracies, 2)}")
print(f"Average Accuracy: {np.mean(fold_accuracies):.2f}")


# In[ ]:




