#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[8]:


# methods = ["uniform", "entropy", "bald", "bald_generative", "entropy_generative", "bald_duplicate", "entropy_duplicate"]
methods = ["bald_gan", "bald_gan_std0", "entropy_gan", "entropy_gan_std0"]
location = "perf_lists/"
plt.figure(figsize=(15, 10))
plt.xlabel("Acquisition Iterations")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Performance")
for method in methods:
    with open (location + method + "_perf_hist") as f:
        lines = [float(line.rstrip()) for line in f]
    plt.plot(lines, label=method)
    # plt.ylim([0.7,1])
plt.legend()
plt.savefig("plots/test_acc_perf.png")

