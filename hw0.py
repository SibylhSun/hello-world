#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# # Problem 1

# In[59]:


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):
    X_n,X_q = data_NF, query_QF
    n = X_n.shape[0]
    q = X_q.shape[0]
    f = X_n.shape[1]
    k= K
    #dis = np.zeros((q,n))
    dis = np.zeros(n)
    kth_dis = np.zeros(q)
    neighb_QKF = np.zeros((q,k,f))
    for i in range(q):
        X_qtemp = np.array([X_q[i]]*n)
        delta2 = np.sum((X_n - X_qtemp)**2,axis=1)
        #print(np.sqrt(delta2).shape)
        
        dis = np.sqrt(delta2)
        dis = dis.reshape(n,1)
        arr = np.hstack((dis,X_n))
        #print(dis)
        #print(X_n.shape, dis.shape)
        #print(arr)
        
        
        
        arr = arr[arr[:,0].argsort()]
        #print(arr)
        X_k = arr[:k,]
        X_k = np.delete(X_k,0,axis=1)
        #print(X_k)
        neighb_QKF[i] = X_k
    #print(neighb_QKF)       
    return (neighb_QKF)


# # Problem 2

# In[7]:


def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
        
    frac = frac_test
    L,F = x_all_LF.shape
    X = random_state.permutation(x_all_LF)
    N = math.ceil(frac * L)
    M = L-N
    #print(N,M)
    x_train_MF = np.zeros((M,F))
    x_test_NF = np.zeros((N,F))
    x_train_MF = X[:M,]
    x_test_NF = X[M:,]
    
    return (x_train_MF,x_test_NF )


# In[8]:


#x_LF = np.eye(10)
#xcopy_LF = x_LF.copy() # preserve what input was before the call


# In[9]:


#train_MF, test_NF = split_into_train_and_test(x_LF, frac_test=0.201, random_state=np.random.RandomState(0))


# In[10]:


#train_MF.shape


# In[11]:


#test_NF.shape


# In[12]:


#print(train_MF)


# In[13]:


#print(test_NF)


# In[14]:


#np.allclose(x_LF, xcopy_LF)


# In[ ]:




