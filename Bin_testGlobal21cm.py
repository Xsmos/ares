import ares
import numpy as np

pf = \
{
    # 'include_cgm': False,
    'dark_matter_heating': True, # added by Bin Xia
    # 'isothermal': True, # Bin Xia wants False
    # 'expansion': False, # added by Bin Xia
}

sim = ares.simulations.Global21cm(**pf)
sim.run()