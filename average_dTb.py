import ares
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from scipy import interpolate
import multiprocessing
from multiprocessing import Pool
import warnings
import shutil
# from test_ares import test_ares

# V_rms = 29000  # m/s
# N = 5  # number of initial_v_stream


def dTb_random_v_stream(m_chi=0.1, N=10, cores=1, V_rms=29000, average_dir='average_dTb', **kwargs):
    """
    randomly generate N initial_v_streams and calculate their 21cm temperatures with dark_matter_heating.
    """
    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    pf = \
        {
            'radiative_transfer': False,
            'verbose': False,
            'dark_matter_heating': True,
            'include_cgm': False,
            # 'initial_v_stream':0,
            'initial_redshift': 1010,
            'include_He': True  # ,
            # 'dark_matter_mass': 1
        }

    # initial_v_stream_list = np.random.normal(0, V_rms, N)
    mean = np.zeros(3)
    cov = np.eye(3, 3) * V_rms**2 / 3
    # if not kwargs['method'] == 'enumerate':
    if False:
        initial_v_stream_list = np.random.multivariate_normal(mean, cov, N)
        initial_v_stream_list = np.sqrt(np.sum(initial_v_stream_list**2, axis=1))
    else:
        initial_v_stream_list = np.linspace(1, 3*V_rms, kwargs['adequate_random_v_streams'])
    # print("dark_matter_mass = {} GeV".format(m_chi), end='')

    # path = "{}/average_dTb/V_rms{:.0f}/m_chi{:.2f}".format(average_dir, round(V_rms, -1), m_chi)
    path = "{}/V_rms{}/m_chi{}".format(average_dir, V_rms, m_chi)
    # print(__name__, ': average_path =', path)
    if not os.path.exists(path):
        os.makedirs(path)

    start_time = time.time()
    if cores == 1:
        print("1 CPU working...", end='')
        for i, initial_v_stream in enumerate(initial_v_stream_list):
            if verbose:
                print("\ninitial_v_stream =", initial_v_stream, 'm/s', end='')

            sim = ares.simulations.Global21cm(initial_v_stream=initial_v_stream, dark_matter_mass=m_chi, **pf)
            # sim = ares.simulations.Global21cm(initial_v_stream=V_rms, dark_matter_mass=m_chi, **pf)
            # sim = test_ares(initial_v_stream=V_rms, dark_matter_mass=m_chi)
            # sim = test_ares(initial_v_stream=initial_v_stream, dark_matter_mass=m_chi)
            # sim = sim_dict[initial_v_stream]
            sim.run()

            # path = "./average_dTb/V_rms{:.0f}/m_chi{:.2f}".format(
            #     V_rms, sim.pf['dark_matter_mass'])
            # if not os.path.exists(path):
            #     os.makedirs(path)
            # print(__name__, "dTb =", sim.history['dTb'])
            # print(__name__, path+"/{}".format(initial_v_stream))
            np.save(path+"/{}".format(initial_v_stream), np.vstack((sim.history["z"], sim.history["dTb"])))

            number_of_CPUs = 1
    else:
        if cores == -1:
            cpu_count = multiprocessing.cpu_count()
        else:
            cpu_count = cores
        print("{} CPUs working in parallel...".format(cpu_count), end='')
        # print("\n{} CPUs working...".format(multiprocessing.cpu_count()), end='')
        global f_mpi

        def f_mpi(initial_v_stream):
            if verbose:
                print("\npid = {}, initial_v_stream = {} m/s".format(os.getpid(),
                      initial_v_stream), end='')
            
            # sim = test_ares(initial_v_stream=initial_v_stream, dark_matter_mass=m_chi)
            sim = ares.simulations.Global21cm(initial_v_stream=initial_v_stream, dark_matter_mass=m_chi, **pf)
            sim.run()

            np.save(path+"/{}".format(initial_v_stream), np.vstack((sim.history["z"], sim.history["dTb"])))

            return os.getpid()

        with Pool(cpu_count) as p:
            pids = p.map(f_mpi, initial_v_stream_list)
        number_of_CPUs = np.unique(pids).size

    end_time = time.time()
    time_elapse = end_time - start_time
    print("\nIt costs {:.2f} seconds to calculate dTb of {} different initial_v_streams by {} CPU(s).".format(
        time_elapse, N, number_of_CPUs))

def Gaussian_3D(v, V_rms):
    P = np.exp(-3*v**2/(2*V_rms**2)) / (2*np.pi/3*V_rms**2)**(3/2)
    P_v_square = v**2 * P
    return P_v_square

def integrate_dTb_with_Probability(dTbs, file_names, V_rms):
    velocities = np.array([float(name[:-4]) for name in file_names])
    probabilities = Gaussian_3D(velocities, V_rms)
    dTb_averaged = dTbs.T@probabilities / probabilities.sum()
    # print(__name__, "velocities", velocities)
    # print(__name__, "probabilities", probabilities)
    # print(__name__, "dTb_averaged", dTb_averaged)
    return dTb_averaged

def average_dTb(m_chi=0.1, N_z=1000, plot=False, more_random_v_streams=10, cores=1, V_rms=29000, average_dir="average_dTb", **kwargs):
    warnings.simplefilter("ignore", UserWarning)
    # path = "{}/average_dTb/V_rms{:.0f}/m_chi{:.2f}".format(average_dir, round(V_rms, -1), m_chi)
    if 'adequate_random_v_streams' not in kwargs:
        kwargs['adequate_random_v_streams'] = 10
    if 'verbose' not in kwargs:
        verbose = kwargs['verbose'] = False

    path = "{}/V_rms{}/m_chi{}".format(average_dir, V_rms, m_chi)
    if not os.path.exists(path+'.npy') or more_random_v_streams:
        dTb_random_v_stream(m_chi, N=more_random_v_streams, cores=cores, V_rms=V_rms, average_dir=average_dir, **kwargs)

    file_names = os.listdir(path)
    # print("Preprocessing {} files of dTb for m_chi = {} GeV...".format(len(file_names), m_chi))

    z_array = np.linspace(5, 1010, N_z)

    for file_name in file_names:
        data = np.load(path+"/{}".format(file_name))
        dTb_interp = np.interp(z_array, data[0][::-1], data[1][::-1])
        # print(__name__, 'dTb_interp =', dTb_interp)
        if "all_dTb_interp" not in vars():
            all_dTb_interp = dTb_interp.copy()
        else:
            all_dTb_interp = np.vstack((all_dTb_interp, dTb_interp))

    # print("{} files have been interpolated.".format(all_dTb_interp.shape[0]))
    print("---"*15)
    # dTb_averaged = np.average(all_dTb_interp, axis=0)

    # update and save data
    if not os.path.exists(path+'.npy'):
        np.save(path, np.vstack((z_array, all_dTb_interp)))
    else:
        old_data = np.load(path+'.npy')
        new_data = np.vstack((old_data, all_dTb_interp))
        np.save(path, new_data)

    # calculate the averaged value
    data = np.load(path+'.npy')
    # if kwargs['method'] != 'enumerate':
    if False: # for test only!
        dTb_averaged = np.average(data[1:,:], axis=0)
    else:
        dTb_averaged = integrate_dTb_with_Probability(data[1:], file_names, V_rms)
    # print(__name__, 'dTb_averaged =', dTb_averaged)
    np.save(path+"_averaged".format(m_chi), np.vstack((z_array, dTb_averaged)))
    shutil.rmtree(path, ignore_errors=True)
    # print(__name__ + ": Files in " + path + " have been removed.")

    if plot:
        z, T = np.load(path+"_averaged.npy")
        # print(z.shape)
        # print(T.shape)
        # print("plotting...")
        plt.title("averaged dTb")
        plt.xlabel("z")
        plt.ylabel("dTb [mK]")
        plt.ylim(-60, 0)
        plt.plot(z, T, label="m_chi = {} GeV".format(m_chi))
        plt.legend()
        plt.show()
    else:
        return (z_array, dTb_averaged, m_chi, V_rms)

def compare(mylist):
    plt.figure(dpi=150)
    for m_chi, V_rms in mylist:
        print(f"m_chi = {m_chi}, V_rms = {V_rms}")
        z, dTb, m, V = average_dTb(m_chi=m_chi, V_rms=V_rms, cores=1, method='least_squares', adequate_random_v_streams=24)
        # print('dTb =', dTb)
        plt.plot(z, dTb, label=f"{m} GeV, {V} m/s", linestyle=":")
    plt.xlim(0,100)
    # plt.ylim()
    plt.legend()
    plt.savefig("test_compare.png")

if __name__ == '__main__':
    points = np.array([(10.0, 0.27884247752898056), (20000, 79973.86)]).T
    print(f"points = {points}")
    compare(points)

# if __name__ == "__main__":
#     # dTb_random_v_stream()
#     # average_dTb(plot=True)
#     m_chi_list = [0.1, 1, 10]
#     color_dict = {0.1: "blue", 0.5: 'green', 1: "purple", 10: "red"}
#     style_dict = {0.1: '--', 0.5: ':', 1: "-.", 10: ":"}

#     fig, ax = plt.subplots()
#     # fig.figure(dpi = 150)

#     for m_chi in m_chi_list:
#         z, T, m_chi = average_dTb(m_chi, more_random_v_streams=2, cores=4)
#         ax.plot(z, T, label='$m_{\chi}$'+' = {} GeV'.format(m_chi),
#                 color=color_dict[m_chi], linewidth=3, linestyle=style_dict[m_chi])
#         print("---"*30)

#     # sim = ares.simulations.Global21cm(radiative_transfer=False, verbose =False)
#     # sim.run()
#     # plt.plot(sim.history['z'], sim.history['dTb'], label="no DMheat, z_initial = {}".format(sim.pf['initial_redshift']), color='k', linestyle=':', linewidth=0.5)

#     sim0 = ares.simulations.Global21cm(
#         radiative_transfer=False, verbose=False, initial_redshift=1010, include_cgm=False, include_He=True)
#     sim0.run()
#     ax.plot(sim0.history['z'], sim0.history['dTb'], label="no DM heating".format(
#         sim0.pf['initial_redshift']), color='k', linewidth=3, linestyle='-')

#     z0 = sim0.history['z']
#     t0 = sim0.history['t']/31556952000000
#     z2t = interpolate.interp1d(z0, t0, fill_value='extrapolate')
#     t2z = interpolate.interp1d(t0, z0, fill_value='extrapolate')

#     secax = ax.secondary_xaxis('top', functions=(z2t, t2z))
#     secax.set_xlabel('Age of Universe [Myr]', fontsize=11)
#     secax.set_xticks([1, 3, 5, 10, 20, 40, 160])

#     ax.set_xlabel("Redshift", fontsize=11)
#     ax.set_ylabel("Brightness Temperature [mK]", fontsize=11)
#     ax.set_ylim(-60, 0)
#     ax.set_xlim(10, 300)
#     # plt.title("global dTb vs z")
#     plt.legend(handlelength=3)
#     plt.savefig("average_dTb", dpi=720)
#     plt.show()
