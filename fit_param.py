#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import ares
from average_dTb import average_dTb
import os
import random
import warnings
import time


# In[2]:


def interp_dTb(param, z, cores=True, adequate_random_v_streams=100):  # 200 by default
    """
    functions:
    1. generate adequate random stream velocities subject to 3D Gaussian distribution;
    2. average dTb over these velocities;
    3. interpolate dTb by input 'z'.
    """
    m_chi, V_rms = param

    directory = "average_dTb/V_rms{:.0f}/m_chi{:.2f}".format(V_rms, m_chi)
    if os.path.exists(directory):
        if np.size(os.listdir(directory)) < adequate_random_v_streams:
            more_random_v_streams = adequate_random_v_streams - \
                np.size(os.listdir(directory))
            print("{} more random v_streams will be generated for m_chi = {} GeV and V_rms = {} m/s...".format(
                more_random_v_streams, m_chi, V_rms))
        else:
            more_random_v_streams = 0
            try:
                z_array, dTb_averaged = np.load(directory+'_averaged.npy')
                print(
                    "Existing averaged dTb and z are loaded for m_chi = {} GeV and V_rms = {} m/s.".format(m_chi, V_rms))
                print("---"*15)
            except FileNotFoundError:
                more_random_v_streams = 1
                print("{} more random v_stream to be generated for m_chi = {} GeV and V_rms = {} m/s...".format(
                    more_random_v_streams, m_chi, V_rms))
    else:
        more_random_v_streams = adequate_random_v_streams
        print("{} random v_streams will be generated for m_chi = {} GeV and V_rms = {} m/s...".format(
            more_random_v_streams, m_chi, V_rms))

    if more_random_v_streams:
        z_array, dTb_averaged, m_chi = average_dTb(
            m_chi=m_chi, more_random_v_streams=more_random_v_streams, cores=cores, verbose=True, V_rms=V_rms)

    dTb = np.interp(z, z_array, dTb_averaged)
    return dTb


def residual(param, z_sample, dTb_sample, cores=True):
    residual = interp_dTb(param, z_sample, cores) - dTb_sample
    return residual


def fit_param(z_sample, dTb_sample, param_guess=[0.1, 29000], bounds=([0, 29000*(1-1/np.sqrt(3))], [10, 29000*(1+1/np.sqrt(3))]), cores=True):
    '''
    fit the parameter(s) by z_sample and dTb_sample via scipy.optimize.least_squares.
    '''
    warnings.simplefilter("ignore", UserWarning)

    if z_sample.ndim == 1 and dTb_sample.ndim != 1:
        z_sample = np.tile(z_sample, (dTb_sample.shape[0], 1))
    elif z_sample.ndim != 1:
        if z_sample.shape != dTb_sample.shape:
            print("z_sample and dTb_sample should have same shape.")
            return

    # fitting_results_txt = open("average_dTb/fitting_results.txt", 'x')
    if dTb_sample.ndim == 1:
        repeat = 1
    else:
        repeat = dTb_sample.shape[0]

    for i in range(0, repeat):
        if dTb_sample.ndim == 1:
            args_z = z_sample
            args_dTb = dTb_sample
        else:
            args_z = z_sample[i]
            args_dTb = dTb_sample[i]

        start_time = time.time()
        res = least_squares(residual, param_guess, diff_step=0.1,
                            bounds=bounds, xtol=1e-3, args=(args_z, args_dTb, cores))
        end_time = time.time()

        print('#{}'.format(i+1), ', fit:', res.x, ', success:', res.success,
              ', status:', res.status, ', cost {} seconds'.format(end_time-start_time))
        print('---'*30)

        if res.success:
            # fitting_results_txt.write("{} ".format(res.x[0]))
            if "fitting_results" not in vars():
                fitting_results = res.x
            else:
                fitting_results = np.vstack((fitting_results, res.x))
    # fitting_results_txt.close()
    # fitting_results = np.loadtxt("fitting_results.txt")
    # fitting_result = np.average(fitting_results)
#     print('return:', fitting_results)
    return fitting_results


# In[3]:


def test(param_true=[0.15, 29000], noise=3, cores=True, z_sample=np.arange(10, 300, 5), stop_plot=5, repeat=20, plot=True):
    """
    functions:
    1. test the fit_param();
    2. showed that fit_param() works well for m_chi < 1 GeV. 
    """
    # sampling
    dTb_accurate = interp_dTb(param_true, z_sample, cores)
    dTb_sample = dTb_accurate + noise * \
        np.random.normal(size=(repeat, z_sample.shape[0]))

    # fitting
    # param_fit, success, status = fit_param(z_sample, dTb_sample, cores=cores)
    start_time = time.time()
    param_fits = fit_param(z_sample, dTb_sample, cores=cores)
    end_time = time.time()
    # print("param_fits =", param_fits)

    # take the average
    if param_fits.ndim <= 1:
        param_fit = np.array([np.average(param_fits, axis=0)])
    else:
        param_fit = np.average(param_fits, axis=0)

    print("It costs {:.0f} seconds to achieve param_fit = {}".format(
        end_time-start_time, param_fit))
    # print('success =', success)
    # print('status =', status)

    if plot:
        plt.figure(dpi=120)
        sim = ares.simulations.Global21cm(
            radiative_transfer=False, verbose=False)
        sim.run()
        plt.plot(sim.history['z'], sim.history['dTb'],
                 label='no DM heating', color='k', linestyle='--')

        plt.plot(z_sample, dTb_accurate,
                 label=r'$m_{\chi, \rm real}$'+' = {} GeV'.format(param_true[0]))
        for i in range(dTb_sample.shape[0]):
            plt.scatter(z_sample, dTb_sample[i], label=r'fit{} = {:.4f} GeV'.format(
                i, param_fits[i][0]), s=2)
            if i >= stop_plot:
                break
        plt.plot(z_sample, interp_dTb(param_fit, z_sample),
                 label=r'$m_{\chi, \rm fit}$'+' = {:.3f} GeV'.format(param_fit[0]), linestyle=':', c='r')
        plt.xlim(0, 300)
        # plt.ylim(-60,0)
        plt.xlabel(r"$z$")
        plt.ylabel(r"$\overline{\delta T_b} \rm\ [mK]$")
        plt.legend()
        plt.title(
            r"fit $m_\chi$ from observed global $\delta T_b$ with $\sigma_{\rm noise}$"+" = {} mK".format(noise))
        plt.show()

        plt.hist(param_fits, density=True, bins=20)
        mean = np.average(param_fits)
        median = np.median(param_fits)
        std = np.std(param_fits)

        plt.title(
            "distribution of {} fitting values for dark matter mass".format(repeat))
        plt.axvline(param_true, c='r', linestyle='-',
                    label='real = '+'{} GeV'.format(param_true[0]))
        plt.axvline(mean, c='k', linestyle='--',
                    label='mean = '+'{:.3f} GeV'.format(mean))
        plt.axvline(median, c='gold', linestyle='-.',
                    label='median = '+'{:.3f} GeV'.format(median))
        plt.axvline(mean+std, c='k', linestyle=':', label=r'mean $\pm$ std')
        plt.axvline(mean-std, c='k', linestyle=':')
        plt.legend()
        plt.xlabel(r"$m_{\chi}$ [GeV]")
        plt.ylabel("pdf")
        plt.savefig("{}.png".format(param_true[0]*10))
        plt.show()
    else:
        return param_fits


# In[4]:


def demonstrate():
    """
    functions:
    1. show how many random velocities are required to achieve stable and accurate dTb_averaged;
    2. demonstrate that it is good enough to average dTb over 200 stream velocities.
    """
    z_array = np.linspace(10, 1010, 1000)

    try:
        file_names = os.listdir("average_dTb/m_chi0.62")
    except FileNotFoundError:
        print(
            "Interrupt demonstrate() if you don't want following steps to cost long time.")
        average_dTb(m_chi=0.62, more_random_v_streams=400,
                    cores=1, verbose=False)
        file_names = os.listdir("average_dTb/m_chi0.62")

    print("file_names[:5] =", file_names[:5])
    random.shuffle(file_names)
    print("After shuffling, file_names[:5] =", file_names[:5])
    for file_name in file_names:
        data = np.load("./average_dTb/m_chi0.62/{}".format(file_name))
        dTb_interp = np.interp(z_array, data[0][::-1], data[1][::-1])
        if "all_dTb_interp" not in vars():
            all_dTb_interp = dTb_interp.copy()
        else:
            all_dTb_interp = np.vstack((all_dTb_interp, dTb_interp))
    # print(all_dTb_interp.shape)

    plt.figure(figsize=(15, 4), dpi=150)

    N = [100, 200, 300]

    plt.subplot(131)
    v_streams = np.array([float(name[:-4]) for name in file_names])
    plt.title(
        r"velocity distribution (3D gaussian with $V_{\rm rms} \equiv$ 29 km/s)")
    for i in range(len(N)):
        V_sample_rms = np.sqrt(np.average(v_streams[:N[i]]**2))/1000
        plt.hist(v_streams[:N[i]]/1000, label="N={}, ".format(N[i])+r"$\sqrt{\overline{V^2}}$" +
                 "={:.1f}km/s".format(V_sample_rms), density=True, histtype='step', bins=10)
    plt.legend()
    plt.xlabel("stream velocity [km/s]")
    plt.ylabel('probability density')

    plt.subplot(132)
    dTb_averaged = [np.average(all_dTb_interp[:N[i]], axis=0)
                    for i in range(len(N))]
    # print(np.size(dTb_averaged))
    plt.title("z vs. dTb_averaged")
    for i in range(len(N)):
        plt.plot(z_array, dTb_averaged[i], label="N = {}".format(N[i]))
    plt.xlabel("z")
    plt.ylabel(r'$\overline{dTb}$ [mK]')
    plt.legend()
    plt.xlim(0, 300)

    plt.subplot(133)
    N = np.arange(10, 500, 5)
    dTb_averaged = [np.average(all_dTb_interp[:N[i]], axis=0)
                    for i in range(len(N))]
    # print(np.shape(dTb_averaged))
    dTb_averaged_diff = np.array(
        [dTb_averaged[i] - dTb_averaged[-1] for i in range(len(N)-1)])
    # dTb_averaged_diff = dTb_averaged_diff[]
    # print(np.shape(dTb_averaged_diff))
    # plt.figure(dpi=120)
    plt.plot(N[:-1], np.max(abs(dTb_averaged_diff), axis=1))
    plt.hlines(0, xmin=0, xmax=N[-1], linestyles=':')
    plt.xlabel("N")
    plt.ylabel("maximum difference [mK]")
    plt.title(r"N vs. max($\Delta$T)")
    plt.show()


if __name__ == '__main__':
    test([0.5, 29000], cores=False, plot=False)
