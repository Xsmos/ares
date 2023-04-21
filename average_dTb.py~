import ares
import matplotlib.pyplot as plt
import numpy as np
import time
import os

V_rms = 29000  # m/s
# N = 5  # number of initial_v_stream


def dTb_random_v_stream(m_chi=0.1, N=5):
    """
    randomly generate N initial_v_streams and calculate their 21cm temperatures with dark_matter_heating.
    """

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
    initial_v_stream_list = np.random.multivariate_normal(mean, cov, N)
    initial_v_stream_list = np.sqrt(np.sum(initial_v_stream_list**2, axis=1))
    # print(mean, cov, initial_v_stream_list.shape)
    # print(initial_v_stream_list)
    # initial_v_stream_list = abs(initial_v_stream_list)
    # sim_dict = {initial_v_stream:0 for initial_v_stream in initial_v_stream_list}

    # z_array = np.linspace(6, 300, N_z)
    # dTb_dict = {initial_v_stream:0 for initial_v_stream in initial_v_stream_list}

    # default = ares.simulations.Global21cm(verbose=False, radiative_transfer=False)
    # default.run()

    print("dark_matter_mass = {} GeV".format(m_chi), end='')
    start_time = time.time()
    for i, initial_v_stream in enumerate(initial_v_stream_list):
        print("\ninitial_v_stream =", initial_v_stream, 'm/s', end='')
        if os.path.exists("./average_dTb/m_chi{}/{}.npy".format(m_chi, int(initial_v_stream))):
            print(" is skipped because file exists", end='')
            continue

        sim = ares.simulations.Global21cm(
            initial_v_stream=initial_v_stream, dark_matter_mass=m_chi, **pf)
        # sim = sim_dict[initial_v_stream]
        sim.run()

        if not os.path.exists("./average_dTb/m_chi{}".format(sim.pf['dark_matter_mass'])):
            os.makedirs(
                "./average_dTb/m_chi{}".format(sim.pf['dark_matter_mass']))

        np.save("./average_dTb/m_chi{}/{}".format(sim.pf['dark_matter_mass'], (int(
            initial_v_stream))), np.vstack((sim.history["z"], sim.history["dTb"])))
        # dTb_dict[initial_v_stream] = np.interp(z_array, sim.history['z'][::-1], sim.history['dTb'][::-1])
        # sim_dict[initial_v_stream].save()

    end_time = time.time()
    time_elapse = end_time - start_time
    print("\nIt costs {:.2f} seconds to calculate dTb of {} different initial_v_streams.".format(
        time_elapse, N))


def average_dTb(m_chi=0.1, N_z=100, plot=False, save_time=True):
    if not os.path.exists("./average_dTb/m_chi{}".format(m_chi)) or not save_time:
        dTb_random_v_stream(m_chi)

    file_names = os.listdir("./average_dTb/m_chi{}".format(m_chi))
    print("Preprocessing {} files of dTb for m_chi = {} GeV...".format(len(file_names), m_chi))

    z_array = np.linspace(10, 300, N_z)

    for file_name in file_names:
        data = np.load("./average_dTb/m_chi{}/{}".format(m_chi, file_name))
        dTb_interp = np.interp(z_array, data[0][::-1], data[1][::-1])
        if "all_dTb_interp" not in vars():
            all_dTb_interp = dTb_interp.copy()
        else:
            all_dTb_interp = np.vstack((all_dTb_interp, dTb_interp))

    print("{} files have been interpolated.".format(all_dTb_interp.shape[0]))
    dTb_averaged = np.average(all_dTb_interp, axis=0)
    np.save("./average_dTb/m_chi{}_averaged".format(m_chi),
            np.vstack((z_array, dTb_averaged)))

    if plot:
        z, T = np.load("./average_dTb/m_chi{}_averaged.npy".format(m_chi))
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
        return (z_array, dTb_averaged, m_chi)


if __name__ == "__main__":
    # dTb_random_v_stream()
    # average_dTb(plot=True)
    m_chi_list = [0.1, 1, 10]
    color_dict = {0.1: "blue", 1: "purple", 10: "red"}

    for m_chi in m_chi_list:
        z, T, m_chi = average_dTb(m_chi, save_time=False)
        plt.plot(z, T, label='m_chi = {} GeV'.format(m_chi), color=color_dict[m_chi])
        print("---"*30)

    sim = ares.simulations.Global21cm(radiative_transfer=False, verbose =False)
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label="no DM heating", color='k')

    plt.xlabel("z")
    plt.ylabel("dTb [mK]")
    plt.ylim(-60,0)
    plt.xlim(10,300)
    plt.title("global dTb vs z")
    plt.legend()
    plt.show()
