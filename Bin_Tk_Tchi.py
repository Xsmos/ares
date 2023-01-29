import ares
import numpy as np
import matplotlib.pyplot as plt

# sim = ares.simulations.Global21cm(verbose=False, dark_matter_heating=False)
# sim.run()
# plt.plot(sim.history['z'], sim.history['igm_Tk'],
#          label='igm_Tk_default', linestyle='-', c='k')
# history_z = sim.history['z']
# # history_cgm_e = sim.history['cgm_e']
# history_igm_e = sim.history['igm_e']


sim = ares.simulations.Global21cm(
    verbose=False, dark_matter_heating=True, include_cgm=False, initial_v_stream = 0, initial_redshift=1010)#300)#
sim.run()

plt.plot(sim.history['z'], 2.73*(1+sim.history['z']),
         label='Tgamma', c='green', linestyle=':')

plt.plot(sim.history['z'], sim.history['igm_Tk'],
         label='igm_Tk', linestyle='--', c='b')

if sim.pf["dark_matter_heating"]:
    plt.plot(sim.history['z'], sim.history['igm_Tchi'],
             label='igm_Tchi', c='b', linestyle='-.')

plt.title("T_k, T_chi, T_gamma vs. redshift")
plt.xlabel("z")
plt.ylabel(r"$T$ (mK)")
plt.xlim(10, 1000)
plt.ylim(1, 2000)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()

plt.plot(sim.history['z'], sim.history['igm_e'], label='igm_e', linestyle = '-')
# plt.plot(sim.history['z'], sim.history['cgm_e'], label='cgm_e', linestyle = '-')

plt.plot(history_z, history_igm_e, label='igm_e_default', linestyle = '--')
# plt.plot(history_z, history_cgm_e, label='cgm_e_default', linestyle = '--')

plt.title("x_e vs. z")
plt.xlabel("z")
plt.ylabel("x_e")
plt.xscale("log")
plt.legend()
plt.show()
