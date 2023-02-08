import ares
import matplotlib.pyplot as plt

z60 = ares.simulations.Global21cm(radiative_transfer=False, include_cgm=False, initial_redshift=60)
z1010 = ares.simulations.Global21cm(radiative_transfer=False, include_cgm=False, initial_redshift=150)

z60.run()
z1010.run()

plt.plot(z60.history['z'], z60.history['dTb'])
plt.plot(z1010.history['z'], z1010.history['dTb'])

plt.xlim(0,300)
plt.ylim(-60,0)

plt.xlabel('z')
plt.ylabel('dTb [mK]')

plt.show()
