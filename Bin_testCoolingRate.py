import ares
import numpy as np
import matplotlib.pyplot as pl
import os

pf = \
{
    'grid_cells': 1,
    'stop_time': 1e2,
    'radiative_transfer': False,
    'density_units': 1.0,
    'initial_timestep': 1,
    'max_timestep': 1e2,
    'restricted_timestep': None,
    'initial_temperature': 20000,#np.logspace(3, 5, 64),
    'initial_ionization': [1.-1e-8, 1e-8],        # neutral
    'isothermal': False, # Bin Xia wants False
    'expansion': True, # added by Bin Xia
    'dark_matter_heating': False, # added by Bin Xia
}

if os.path.exists("Bin_Tk.txt"):
    os.remove("Bin_Tk.txt")
if os.path.exists("Bin_coolingRate.txt"):
    os.remove("Bin_coolingRate.txt")
if os.path.exists("Bin_z.txt"):
    os.remove("Bin_z.txt")
sim = ares.simulations.GasParcel(**pf)
sim.run()


import numpy as np
import matplotlib.pyplot as pl
Tk = np.loadtxt("Bin_Tk.txt")
CoolingRate = np.loadtxt("Bin_coolingRate.txt")
z = np.loadtxt("Bin_z.txt")
#grid_data_Tk = np.loadtxt("grid_data_Tk.txt")

fig, ax = pl.subplots(1, 2, figsize = (10,6))

ax[0].scatter(Tk, CoolingRate, s = 1)
#pl.plot(Tk, grid_data_Tk)
ax[0].set_xlabel('T [K]')
ax[0].set_ylabel('$\mathrm{Cooling\ Rate\ [erg\ s^{-1}\ cm^3]}$')
ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlim([1e2, 1e8])
ax[0].set_ylim([1e-30, 1e-20])


ax[1].scatter(z+1, Tk, s = 1)
ax[1].set_xlabel('$z$ + 1')
ax[1].set_ylabel('$T_k$ [K]')
ax[1].set_yscale("log")

pl.show()