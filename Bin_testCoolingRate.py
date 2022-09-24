import ares
import numpy as np
import matplotlib.pyplot as pl
import os

pf = \
{
    'grid_cells': 64,
    'isothermal': False, # Bin Xia wants False
    'stop_time': 1e2,
    'radiative_transfer': False,
    'density_units': 1.0,
    'initial_timestep': 1,
    'max_timestep': 1e2,
    'restricted_timestep': None,
    'initial_temperature': np.logspace(3, 5, 64),
    'initial_ionization': [1.-1e-8, 1e-8],        # neutral
    #'expansion' : True, # added by Bin Xia
}

if os.path.exists("Bin_Tk.txt"):
    os.remove("Bin_Tk.txt")
if os.path.exists("Bin_coolingRate.txt"):
    os.remove("Bin_coolingRate.txt")

sim = ares.simulations.GasParcel(**pf)
sim.run()


import numpy as np
import matplotlib.pyplot as pl
Tk = np.loadtxt("Bin_Tk.txt")
CoolingRate = np.loadtxt("Bin_coolingRate.txt")
grid_data_Tk = np.loadtxt("grid_data_Tk.txt")

pl.scatter(Tk, CoolingRate, s = 1)
pl.plot(Tk, grid_data_Tk)

pl.xlabel('T [K]')
pl.ylabel('$\mathrm{Cooling\ Rate\ [erg\ s^{-1}\ cm^3]}$')
pl.xscale("log")
pl.yscale("log")
'''
pl.xlim([1e2, 1e8])
pl.ylim([1e-30, 1e-20])
'''
pl.show()