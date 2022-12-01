import ares
import numpy as np
import matplotlib.pyplot as plt

sim = ares.simulations.Global21cm(dark_matter_heating=False)
sim.run()
plt.plot(sim.history['z'], sim.history['dTb'])

if sim.pf["dark_matter_heating"]:
    plt.plot(sim.history['z'], sim.history['Tchi'])

plt.title("T_b and T_chi vs. redshift")
plt.xlabel("z")
plt.ylabel(r"$T$ (mK)")
plt.xlim(10, 1000)
plt.ylim(1, 2000)
plt.xscale("log")
plt.yscale("log")
plt.show()
