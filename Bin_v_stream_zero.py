import ares

sim = ares.simulations.Global21cm(
    radiative_transfer=False, verbose=False, dark_matter_heating=True, include_cgm=False, initial_v_stream=0, initial_redshift=1010)  # 300)#
sim.run()
