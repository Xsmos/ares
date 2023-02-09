import ares
import numpy as np
import matplotlib.pyplot as plt

sim = ares.simulations.Global21cm(
    radiative_transfer=False, verbose=False, dark_matter_heating=False, include_cgm=False, initial_redshift=1010)
sim.run()
plt.plot(1+sim.history['z'], sim.history['igm_Tk'],
         label='igm_Tk_default', linestyle='-', c='k')
history_z = sim.history['z']
# history_cgm_e = sim.history['cgm_e']
history_igm_e = sim.history['igm_e']


sim = ares.simulations.Global21cm(
    radiative_transfer=False, verbose=False, dark_matter_heating=True, include_cgm=False, initial_v_stream=0, initial_redshift=1010)  # 300)#
sim.run()
plt.plot(1+sim.history['z'], 2.73*(1+sim.history['z']),
         label='Tgamma', c='green', linestyle=':')
plt.plot(1+sim.history['z'], sim.history['igm_Tk'],
         label=r'igm_Tk, $V{\chi b,0}=$'+'{}m/s'.format(sim.pf['initial_v_stream']), linestyle='--', c='b')
if sim.pf["dark_matter_heating"]:
    plt.plot(1+sim.history['z'], sim.history['igm_Tchi'],
             label=r'igm_Tchi, $V{\chi b,0}=$'+'{}m/s'.format(sim.pf['initial_v_stream']), c='b', linestyle='-.')


sim = ares.simulations.Global21cm(
    radiative_transfer=False, verbose=False, dark_matter_heating=True, include_cgm=False, initial_v_stream=29000, initial_redshift=1010)  # 300)#
sim.run()
# plt.plot(1+sim.history['z'], 2.73*(1+sim.history['z']),
#          label='Tgamma', c='green', linestyle=':')
plt.plot(1+sim.history['z'], sim.history['igm_Tk'],
         label='igm_Tk, $V{\chi b,0}=$'+'{}m/s'.format(sim.pf['initial_v_stream']), linestyle='--', c='r')
if sim.pf["dark_matter_heating"]:
    plt.plot(1+sim.history['z'], sim.history['igm_Tchi'],
             label='igm_Tchi, $V{\chi b,0}=$'+'{}m/s'.format(sim.pf['initial_v_stream']), c='r', linestyle='-.')
plt.title("T_k, T_chi, T_gamma vs. redshift")
plt.xlabel("1+z")
plt.ylabel(r"$T$ (K)")
plt.xlim(10, 1000)
plt.ylim(1, 2000)
# plt.xscale("log")
# plt.yscale("log")
plt.legend(loc='upper left')
plt.show()


""" 
plt.plot(1+sim.history['z'], sim.history['igm_e'], label='igm_e', linestyle = '-')
# plt.plot(1+sim.history['z'], sim.history['cgm_e'], label='cgm_e', linestyle = '-')
plt.plot(history_z, history_igm_e, label='igm_e_default', linestyle = '--')
# plt.plot(history_z, history_cgm_e, label='cgm_e_default', linestyle = '--')
plt.title("x_e vs. z")
plt.xlabel("z")
plt.ylabel("x_e")
plt.xscale("log")
plt.legend()
plt.show()
 """
