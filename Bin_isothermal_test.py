# test the effect of isothermal for igm and cgm.
import ares
import matplotlib.pyplot as pl

ax = None
for i, igm in enumerate([True, False]):
    for j, cgm in enumerate([True, False]):
        sim = ares.simulations.Global21cm(
            igm_isothermal=igm, cgm_isothermal=cgm, verbose=False, progress_bar=False
        )
        sim.run()

        # Plot the global signal
        ax, zax = sim.GlobalSignature(
            ax=ax,
            fig=3,
            z_ax=i == j == 0,
            label="isothermal: igm={}, cgm={}".format(igm, cgm),
        )

ax.legend()

pl.show()
