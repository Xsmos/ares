import numpy as np
import ares

sim = ares.simulations.Global21cm(dark_matter_heating=True, initial_redshift= 400.) #)#1100)#
sim.run()