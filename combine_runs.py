#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


methods = ["uniform", "entropy", "bald", "bald_generative", "entropy_generative", "bald_duplicate", "entropy_duplicate", "bald_gan"]
runs = {}
for method in methods:
    runs[method] = []
    for location in ["perf_lists0/", "perf_lists1/", "perf_lists2/"]:
            with open (location + method + "_perf_hist") as f:
                lines = [float(line.rstrip()) for line in f]
            runs[method].append(np.array(lines))
    runs[method] = np.array(runs[method])
for key in runs:
    m_arr = np.mean(runs[key], axis=0)
    std_arr = np.std(runs[key], axis=0)
    with open("perf_lists_final/mean/" + key, 'w') as f:
        input_list = m_arr.tolist()
        for val in input_list:
            f.write("%s\n" % val)
    with open("perf_lists_final/std/" + key, 'w') as f:
        input_list = std_arr.tolist()
        for val in input_list:
            f.write("%s\n" % val)


# In[ ]:




