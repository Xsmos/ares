import ares
import numpy as np
import matplotlib.pyplot as plt

sim = ares.simulations.Global21cm(dark_matter_heating=False)
sim.run()
plt.plot(sim.history['z'], 2.73*(1+sim.history['z']), label='Tgamma', c = 'green', linestyle = ':')
plt.plot(sim.history['z'], sim.history['Ts'], label='Ts', c = 'black', linestyle = '-')
plt.plot(sim.history['z'], sim.history['igm_Tk'], label='igm_Tk', c = 'black', linestyle = '--')
# plt.plot(sim.history['z'], sim.history['cgm_Tk'], label='cgm_Tk', c = 'black', linestyle = '-.')

# if sim.pf["dark_matter_heating"]:
#     plt.plot(sim.history['z'], sim.history['Tchi'], label='Tchi')

sim = ares.simulations.Global21cm(dark_matter_heating=True)
sim.run()
plt.plot(sim.history['z'], sim.history['Ts'], label='Ts', c = 'blue', linestyle = '-')
plt.plot(sim.history['z'], sim.history['igm_Tk'], label='igm_Tk', c = 'blue', linestyle = '--')

plt.title("T_k, T_s, T_gamma vs. redshift")
plt.xlabel("z")
plt.ylabel(r"$T$ (mK)")
plt.xlim(20, 500)
plt.ylim(20, 1000)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.show()
