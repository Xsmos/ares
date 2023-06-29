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


# In[2]:


def interp_dTb(param, z, mpi=True, adequate_random_v_streams=200):
    """
    functions:
    1. generate adequate random stream velocities subject to 3D Gaussian distribution;
    2. average dTb over these velocities;
    3. interpolate dTb by input 'z'.
    """
    m_chi = param[0]
    
    directory = "average_dTb/m_chi"+"{:.2f}".format(m_chi)
    if os.path.exists(directory):
        if np.size(os.listdir(directory)) < adequate_random_v_streams:
            more_random_v_streams = adequate_random_v_streams - np.size(os.listdir(directory))
            print("{} more random v_streams will be generated for m_chi = {} GeV...".format(more_random_v_streams, m_chi))
        else:
            more_random_v_streams = 0
            z_array, dTb_averaged = np.load(directory+'_averaged.npy')
            print("Adequate random v_streams already exist for m_chi = {} GeV. Existing averaged dTb and z loaded.".format(m_chi))
            print("---"*15)
    else:
        more_random_v_streams = adequate_random_v_streams
        print("{} random v_streams will be generated for m_chi = {} GeV...".format(more_random_v_streams, m_chi))
    
    if more_random_v_streams:
        z_array, dTb_averaged, m_chi = average_dTb(m_chi=m_chi, more_random_v_streams=more_random_v_streams, mpi=mpi, verbose=False)
    
    dTb = np.interp(z, z_array, dTb_averaged)
    return dTb

def residual(param, z_sample, dTb_sample, mpi=True):
    residual = interp_dTb(param, z_sample, mpi) - dTb_sample
    return residual

def fit_param(z_sample, dTb_sample, param_guess=[0.1], bounds=([0,10]), mpi=True, repeat=2):
    '''
    fit the parameter(s) by z_sample and dTb_sample via scipy.optimize.least_squares.
    '''
    warnings.simplefilter("ignore", UserWarning)
    if z_sample.shape != dTb_sample.shape:
        print("z_sample and dTb_sample should have same shape.")
        return
    
    # fitting_results_txt = open("average_dTb/fitting_results.txt", 'x')
    for i in range(0, repeat):
        res = least_squares(residual, param_guess, diff_step=0.1, bounds=bounds, xtol=1e-3, args=(z_sample, dTb_sample, mpi))
        print('fit:', res.x, 'success:', res.success, 'status:', res.status)
        if res.success:
            # fitting_results_txt.write("{} ".format(res.x[0]))
            if "fitting_results" not in vars():
                fitting_results = res.x
            else:
                fitting_results = np.vstack((fitting_results, res.x))
    # fitting_results_txt.close()
    # fitting_results = np.loadtxt("fitting_results.txt")
    # fitting_result = np.average(fitting_results)
    print('---'*15)
    return fitting_results



# In[3]:


def test(param_true=[0.15], noise=3, mpi=True, z_sample = np.arange(10, 300, 5)):
    """
    functions:
    1. test the fit_param();
    2. showed that fit_param() works well for m_chi < 1 GeV. 
    """
    # sampling
    dTb_accurate = interp_dTb(param_true, z_sample)
    dTb_sample = dTb_accurate + noise * np.random.normal(size = z_sample.shape[0])
    
    # fitting
    # param_fit, success, status = fit_param(z_sample, dTb_sample, mpi=mpi)
    param_fit = fit_param(z_sample, dTb_sample, mpi=mpi)
    print("fitting_results =", param_fit)
    param_fit = np.average(param_fit, axis=0)
    print("param_fit =", param_fit)
    # print('success =', success)
    # print('status =', status)
    
    plt.figure(dpi=120)
    sim = ares.simulations.Global21cm(radiative_transfer=False, verbose=False)
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = 'no DM heating', color='k', linestyle='--')
    
    plt.plot(z_sample, dTb_accurate, label = r'$m_{\chi, \rm real}$'+' = {} GeV'.format(param_true[0]))
    plt.scatter(z_sample, dTb_sample, label=r'sample, $\sigma_{\rm noise}$'+' = {} mK'.format(noise), s=8)
    plt.plot(z_sample, interp_dTb(param_fit, z_sample), label = r'$m_{\chi, \rm fit}$'+' = {:.2f} GeV'.format(param_fit[0]), linestyle=':', c='r')
    plt.xlim(0,300)
    # plt.ylim(-60,0)
    plt.xlabel(r"$z$")
    plt.ylabel(r"$\overline{\delta T_b} \rm\ [mK]$")
    plt.legend()
    plt.title(r"fit $m_\chi$ from observed global $\delta T_b$")
    plt.show()


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
        print("Interrupt demonstrate() if you don't want following steps to cost long time.")
        average_dTb(m_chi=0.62, more_random_v_streams=400, mpi=1, verbose=False)
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

    plt.figure(figsize=(15,4), dpi=150)
    
    N = [100, 200, 300]
    
    plt.subplot(131)
    v_streams = np.array([float(name[:-4]) for name in file_names])
    plt.title(r"velocity distribution (3D gaussian with $V_{\rm rms} \equiv$ 29 km/s)")
    for i in range(len(N)):
        V_sample_rms = np.sqrt(np.average(v_streams[:N[i]]**2))/1000
        plt.hist(v_streams[:N[i]]/1000, label="N={}, ".format(N[i])+r"$\sqrt{\overline{V^2}}$"+"={:.1f}km/s".format(V_sample_rms), density=True, histtype='step', bins=10)
    plt.legend()
    plt.xlabel("stream velocity [km/s]")
    plt.ylabel('probability density')
    
    plt.subplot(132)
    dTb_averaged = [np.average(all_dTb_interp[:N[i]], axis=0) for i in range(len(N))]
    # print(np.size(dTb_averaged))
    plt.title("z vs. dTb_averaged")
    for i in range(len(N)):
        plt.plot(z_array, dTb_averaged[i], label="N = {}".format(N[i]))
    plt.xlabel("z")
    plt.ylabel(r'$\overline{dTb}$ [mK]')
    plt.legend()
    plt.xlim(0,300)

    plt.subplot(133)
    N = np.arange(10, 500, 5)
    dTb_averaged = [np.average(all_dTb_interp[:N[i]], axis=0) for i in range(len(N))]
    # print(np.shape(dTb_averaged))
    dTb_averaged_diff = np.array([dTb_averaged[i] - dTb_averaged[-1] for i in range(len(N)-1)])
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
    test([0.5], mpi=False)
