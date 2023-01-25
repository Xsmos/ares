import ares
import numpy as np
import matplotlib.pyplot as plt

sim = ares.simulations.Global21cm(dark_matter_heating=False)
sim.run()
plt.plot(sim.history['z'], sim.history['dTb'], label='default')

sim = ares.simulations.Global21cm(dark_matter_heating=True)
sim.run()
plt.plot(sim.history['z'], sim.history['dTb'], label='DM, z_initial = 60', linestyle='--')

sim = ares.simulations.Global21cm(dark_matter_heating=True, initial_redshift=300)
sim.run()
plt.plot(sim.history['z'], sim.history['dTb'], label='DM, z_initial = 300')

plt.title("global 21 cm signal vs. redshift")
plt.xlabel("z")
plt.ylabel(r"$T_{21}$ (mK)")
plt.xlim(0, 300)
plt.ylim(-60, 0)
plt.legend()
plt.show()