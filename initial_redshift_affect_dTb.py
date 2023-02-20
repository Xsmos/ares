import ares
import matplotlib.pyplot as plt

z60 = ares.simulations.Global21cm(
    radiative_transfer=False, include_cgm=False, initial_redshift=60, include_He=True)
z1010 = ares.simulations.Global21cm(
    radiative_transfer=False, include_cgm=False, initial_redshift=1010, include_He=True)#, clumping_factor=2

z60.run()
z1010.run()

plt.plot(z60.history['z'], z60.history['dTb'], label='60')
plt.plot(z1010.history['z'], z1010.history['dTb'], label='1010')

plt.xlim(0, 1100)#300)
# plt.ylim(-60, 0)
plt.ylim(-45, 10)

plt.xlabel('z')
plt.ylabel('dTb [mK]')
plt.legend()

# plt.xscale('log')
# plt.yscale('log')
plt.show()
