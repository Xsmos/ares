import ares
import numpy as np

pf = \
{
    # 'grid_cells': 64,
    # 'stop_time': 1e2,
    # 'radiative_transfer': False,
    # 'density_units': 1.0,
    # 'initial_timestep': 1,
    # 'max_timestep': 1e2,
    # 'restricted_timestep': None,
    # 'initial_temperature': np.logspace(3, 5, 64),#20000,#
    # 'initial_ionization': [1.-1e-8, 1e-8],        # neutral
    'dark_matter_heating': True, # added by Bin Xia
    # 'isothermal': True, # Bin Xia wants False
    # 'expansion': False, # added by Bin Xia
    # 'Bin': True
}

sim = ares.simulations.Global21cm(**pf)
sim.run()