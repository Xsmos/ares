#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ares


# In[2]:


noDM = ares.simulations.Global21cm(radiative_transfer=False, include_cgm=False, dark_matter_heating=False, initial_redshift=1010)


# In[3]:


noDM.run()


# In[4]:


DMv0 = ares.simulations.Global21cm(radiative_transfer=False, include_cgm=False, dark_matter_heating=True, initial_redshift=1010, initial_v_stream = 0)


# In[7]:


DMv0.run()


# In[8]:


DMv29 = ares.simulations.Global21cm(radiative_transfer=False, include_cgm=False, dark_matter_heating=True, initial_redshift=1010, initial_v_stream = 29000)


# In[9]:


DMv29.run()


# In[10]:


import matplotlib.pyplot as plt


# In[20]:


plt.plot(1+noDM.history['z'], (1+noDM.history['z'])*2.73, label='Tcmb', linestyle=':', c='green')

plt.plot(1+noDM.history['z'], noDM.history['igm_Tk'], linestyle='--', c='k')
plt.plot(1+noDM.history['z'], noDM.history['Ts'], label='noDM', linestyle='-', c='k')

plt.plot(1+DMv0.history['z'], DMv0.history['igm_Tk'], linestyle='--', c='b')
plt.plot(1+DMv0.history['z'], DMv0.history['Ts'], label='DM,v=0', linestyle='-', c='b')

plt.plot(1+DMv29.history['z'], DMv29.history['igm_Tk'], linestyle='--', c='r')
plt.plot(1+DMv29.history['z'], DMv29.history['Ts'], label='DM,v=29000m/s', linestyle='-', c='r')

plt.title("Ts (solid) and Tgas (dashed) vs. redshift")
plt.xlabel("1+z")
plt.ylabel("T [K]")
plt.xlim(20,500)
plt.ylim(20,1000)
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()


# In[ ]:




