import ares
import matplotlib.pyplot as plt
import numpy as np

pf = \
    {
    'radiative_transfer':False,
    'verbose':False,
    'dark_matter_heating':True, 
    'include_cgm':False, 
    # 'initial_v_stream':0, 
    'initial_redshift':1010, 
    'include_He':True
    }

V_rms = 29000 # m/s


initial_v_stream = np.random.normal(0, V_rms)
sim = ares.simulations.Global21cm(initial_v_stream = initial_v_stream, **pf)
