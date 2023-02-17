#!/usr/bin/env python
# coding: utf-8

# In[4]:


import ares
import matplotlib.pyplot as plt


# In[6]:


z60 = ares.simulations.Global21cm(
    radiative_transfer=False, include_cgm=False, initial_redshift=60)
z1010 = ares.simulations.Global21cm(
    radiative_transfer=False, include_cgm=False, initial_redshift=1010)#, clumping_factor=2

z60.run()
z1010.run()


# In[22]:


z60.history['igm_Tk']


# In[28]:


plt.plot(z60.history['z'], 2.73*(1+z60.history['z']), label='Tcmb')
plt.plot(z60.history['z'], z60.history['igm_Tk'], label='T_z60')
plt.plot(z1010.history['z'], z1010.history['igm_Tk'], label='T_z1010')
# plt.xscale("log")
# plt.yscale("log")
plt.xlim((10,1000))
plt.ylim((1,2000))
plt.legend()
plt.show()

# In[ ]:




