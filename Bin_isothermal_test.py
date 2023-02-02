# test the effect of isothermal for igm and cgm. test batcat
import ares
import matplotlib.pyplot as pl
import numpy as np

ax = None
for i, igm_isothermal in enumerate([False]):
    for j, cgm_isothermal in enumerate([True, False]):
        sim = ares.simulations.Global21cm(
            igm_isothermal=igm_isothermal, cgm_isothermal=cgm_isothermal, verbose=False, progress_bar=False
        )
        sim.run()

        # Plot the global signal
        ax, zax = sim.GlobalSignature(
            ax=ax,
            fig=3,
            z_ax=i == j == 0,
            label="isothermal: igm={}, cgm={}".format(igm_isothermal, cgm_isothermal),
        )

ax.legend()
pl.show()
