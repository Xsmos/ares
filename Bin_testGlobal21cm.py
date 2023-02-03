import ares
import numpy as np
import matplotlib.pyplot as plt

sim = ares.simulations.Global21cm(radiative_transfer=False, verbose=False, dark_matter_heating=False, include_cgm=False)
sim.run()
plt.plot(1+sim.history['z'], sim.history['dTb'],
             label=r'DM, $V{\chi b,0}=$'+'{}m/s'.format(sim.pf['initial_v_stream']), c='k', linestyle='-')


sim = ares.simulations.Global21cm(radiative_transfer=False, verbose=False, dark_matter_heating=True, include_cgm=False, initial_v_stream=0, initial_redshift=1010)
sim.run()
plt.plot(1+sim.history['z'], sim.history['dTb'],
             label=r'DM, $V{\chi b,0}=$'+'{}m/s'.format(sim.pf['initial_v_stream']), c='b', linestyle='--')


sim = ares.simulations.Global21cm(radiative_transfer=False, verbose=False, dark_matter_heating=True, include_cgm=False, initial_redshift=1010)
sim.run()
plt.plot(1+sim.history['z'], sim.history['dTb'],
             label=r'DM, $V{\chi b,0}=$'+'{}m/s'.format(sim.pf['initial_v_stream']), c='r', linestyle=':')


plt.title("global 21 cm signal vs. redshift")
plt.xlabel("z")
plt.ylabel(r"$T_{21}$ (mK)")
plt.xlim(0, 300)
plt.ylim(-60, 0)
plt.legend()
plt.show()