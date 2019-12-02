#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import numba


# In[2]:


from numba import vectorize, jit


# In[3]:


import time, random


# In[5]:


def serial_kde(eval_points, samples, band):
    
    re_x = (eval_points[:, np.newaxis] - samples[np.newaxis, :]) / band[np.newaxis, :]
    gaussian = np.exp(-0.5*(re_x**2))/np.sqrt(2*np.pi)/band[np.newaxis, :]
    
    return gaussian.sum(axis=1)/len(samples)


# In[7]:


@jit(nopython=True)
def gaussian(x):
    
    return np.exp(-0.5*(x**2))/np.sqrt(2*np.pi)


# In[92]:


@jit(nopython=True, parallel=True)
def parallel_kde(eval_points, samples, band):
    
    res = np.zeros_like(eval_points)
    for i in numba.prange(len(eval_points)):
        eval_x = eval_points[i]
        for s, b in zip(samples, band):
            res[i] += gaussian((eval_x-s)/b)/b
        res[i] /= len(samples)
        
    return res


# In[146]:


def generate_input_samples():
    
    for dtype in [np.float64]:
        for n in [1000,5000]:
            sigma=0.5
            samples = np.random.normal(loc=0.0, scale=sigma, size=n).astype(dtype)
            band = np.full_like(samples, 1.06*n**0.2*sigma)
            for n_eval in [10,1000, 5000]:
                cat = ('samples%d' %n,np.dtype(dtype).name)
                ep = np.random.normal(loc=0.0, scale=5.0, size=n_eval).astype(dtype)
                yield dict(category=cat, x=n_eval, input_args=(ep, samples, band), input_kwargs={})


# In[147]:


val = generate_input_samples()


# In[79]:


for item in val:
    print (item.keys())
    print (item['x'])


# In[148]:


ip_args, ip_kwargs, size = tuple(),dict(),0
for item in val:
    ip_args = item['input_args']
    ip_kwargs = item['input_kwargs']
    size = item['x']


# In[149]:


e_p, s, b = ip_args[0], ip_args[1], ip_args[2]


# In[150]:


start_s = time.time()
serial_kde(e_p,s,b)
end_s = time.time()


# In[151]:


print("Time taken for Serial implementation : {} seconds".format(end_s-start_s))
print("Dataset Size : {} tuples".format(size))


# In[152]:


start_p = time.time()
parallel_kde(e_p,s,b)
end_p = time.time()


# In[153]:


print("Time taken for Parallel implementation : {} seconds".format(end_p-start_p))
print("Dataset Size : {} tuples".format(size))


# In[154]:


print("Improvement with parallel implementation : {:.2f} times".format(1/((end_p-start_p)/(end_s-start_s))))


# In[ ]:




