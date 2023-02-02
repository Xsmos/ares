import ares
import matplotlib.pyplot as plt
import numpy as np

sim = ares.simulations.Global21cm(
    radiative_transfer=False, verbose=False, dark_matter_heating=True, include_cgm=False, initial_v_stream=29000/2, initial_redshift=1010, secondary_ionization=0)  # 300)#
sim.run()
