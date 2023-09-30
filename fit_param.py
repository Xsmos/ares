#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.optimize import least_squares, curve_fit
import numpy as np
import matplotlib.pyplot as plt
import ares
from average_dTb import average_dTb
import os
# import random
import warnings
import time
from npy_append_array import NpyAppendArray
from itertools import product

# In[2]:

# average_path = 'average_dTb_'


def interp_dTb(param, z, cores=1, average_dir="average_dTb", **kwargs):  # 200 by default
    """
    functions:
    1. generate adequate random stream velocities subject to 3D Gaussian distribution;
    2. average dTb over these velocities;
    3. interpolate dTb by input 'z'.
    """
    m_chi, V_rms = param
    # V_rms = int(round(V_rms,-1)) # accuracy: 10 m/s

    if "adequate_random_v_streams" not in kwargs:
        adequate_random_v_streams = kwargs['adequate_random_v_streams'] = 200
    else:
        adequate_random_v_streams = kwargs['adequate_random_v_streams']

    directory = "{}/V_rms{}/m_chi{}".format(average_dir, V_rms, m_chi)
    # print("__name__: directory =", directory)
    # directory = "{}/average_dTb/V_rms{:.0f}/m_chi{:.2f}".format(average_path, round(V_rms, -1), m_chi)
    if os.path.exists(directory+'.npy'):
        data = np.load(directory+'.npy')
        if data.shape[0]-1 < adequate_random_v_streams:
            more_random_v_streams = adequate_random_v_streams - (data.shape[0]-1)
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
        z_array, dTb_averaged, m_chi, V_rms = average_dTb(
            m_chi=m_chi, more_random_v_streams=more_random_v_streams, cores=cores, verbose=False, V_rms=V_rms, average_dir=average_dir, **kwargs)

    dTb = np.interp(z, z_array, dTb_averaged)
    return dTb

def interp_dTb_for_curve_fit(z, m_chi, V_rms):
    param = [m_chi, V_rms]
    dTb_interp = interp_dTb(param, z)
    return dTb_interp
    

def residual(param, z_sample, dTb_sample, kwargs, cores=1, average_dir="average_dTb"):
    residual = interp_dTb(param, z_sample, cores, average_dir=average_dir, **kwargs) - dTb_sample
    print(f"residual = {np.average(residual**2)}, shape = {residual.shape}")
    return residual

class Grid():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        print("Grid's parameters:", self.kwargs)
        print(__name__, f"Initiated grid {self.grid}")

    @property
    def bounds(self):
        if not hasattr(self, '_bounds'):
            self._bounds = self.kwargs['bounds']
            # self._bounds = np.array([[0.01,0],[1,60000]])
        return self._bounds
    
    @bounds.setter
    def bounds(self, value):
        if np.shape(value) != (2,2):
            raise TypeError("Shape of bounds must be (2,2).")
        self._bounds = np.array(value)

    @property
    def N_grid(self):
        if not hasattr(self, '_N_grid'):
            self._N_grid = self.kwargs['N_grid']
        return self._N_grid
    
    @N_grid.setter
    def N_grid(self, value):
        if np.shape(value) != (2,):
            raise TypeError("Shape of N_grid must be (2,).")
        self._N_grid = np.array(value)

    @property
    def grid(self):
        if not hasattr(self, '_grid'):
            grid_x = np.logspace(np.log10(self.bounds[0,0]), np.log10(self.bounds[1,0]), self.N_grid[0])
            grid_y = np.linspace(self.bounds[0,1], self.bounds[1,1], self.N_grid[1])
            # print(grid_x)
            self._grid = list(product(grid_x, grid_y))
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = value

    def generate_dTb_map(self):
        self.dTb = {}
        for grid in self.grid:
            self.dTb[grid] = interp_dTb(grid, self.kwargs['z_sample'], **self.kwargs)

    def calculate_difference(self):
        self.diff = {}
        for grid in self.grid:
            self.diff[grid] = np.average((self.dTb[grid]-self.kwargs['dTb_sample'])**2)
            # print("dTb", self.dTb[grid])
            # print("dTb_sample", self.kwargs['dTb_sample'])
            # print(f"difference for {grid} is {self.diff[grid]}")

    def find_least_squares(self):
        idx = np.argmin(list(self.diff.values()))
        grid = list(self.diff.keys())[idx]
        print("closest fit is:", grid)
        return grid


def fit_param(z_sample, dTb_sample, param_guess=[0.1, 29000], cores=1, average_dir='average_dTb', delete_if_exists=False, save_name="fitted_m_chi_V_rms.npy", **kwargs):
    '''
    fit the parameter(s) by z_sample and dTb_sample via scipy.optimize.least_squares.
    '''
    if 'method' in kwargs:
        method = kwargs['method']
    else:
        method = kwargs['method'] = 'enumerate'
        
    if "bounds" in kwargs:
        bounds = kwargs['bounds']
    else:
        bounds = kwargs['bounds'] = np.array([[0.001,10000],[100,100000]])

    if "N_grid" in kwargs:
        N_grid = kwargs['N_grid']
    else:
        N_grid = kwargs['N_grid'] = (5,5)


    warnings.simplefilter("ignore", UserWarning)
    # global average_path
    # average_path = average_dir

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
        # print(__name__, "method =", method)
        if method == "least_squares":
            res = least_squares(residual, param_guess, bounds=bounds, args=(args_z, args_dTb, kwargs, cores, average_dir), diff_step=0.1)
            theta_fit = res.x
            if res.success == False:
                continue
        elif method == "curve_fit":
            theta_fit, err_fit = curve_fit(interp_dTb_for_curve_fit, args_z, args_dTb, p0=param_guess, bounds=bounds)
        elif method == 'enumerate':
            grid = Grid(z_sample=z_sample, dTb_sample=dTb_sample, param_guess=param_guess, cores=cores, 
                        average_dir=average_dir, delete_if_exists=delete_if_exists, save_name=save_name, **kwargs)
            grid.generate_dTb_map()
            grid.calculate_difference()
            theta_fit = grid.find_least_squares()
            

        end_time = time.time()
        
        if method == "least_squares":
            print('#{}'.format(i+1), ', fit:', theta_fit, ', success:', res.success, ', status:', res.status, f', cost {(end_time-start_time)/60:.2f} min')
        elif method == "enumerate":
            print('#{}'.format(i+1), ', fit:', theta_fit, f', cost {(end_time-start_time)/60:.2f} min')
            
        print('---'*30)

        with NpyAppendArray(save_name, delete_if_exists=delete_if_exists) as npaa:
                npaa.append(np.array([theta_fit]))
    return

# In[3]:


def test(param_true=[0.15, 29000], noise=0.01, cores=-1, z_sample=np.arange(10, 800, 1), stop_plot=5, repeat=20, plot=True, average_dir="average_dTb", delete_if_exists=False, param_guess=[0.1, 29000], **kwargs):
    """
    functions:
    1. test the fit_param();
    2. showed that fit_param() works well for m_chi < 1 GeV. 
    """
    print("param_true =", param_true)
    if 'method' not in kwargs:
        method = kwargs['method'] = 'enumerate'
    else:
        method = kwargs['method']
        
    # sampling
    dTb_accurate = interp_dTb(param_true, z_sample, cores, average_dir=average_dir, **kwargs)
    dTb_sample = dTb_accurate + noise * np.random.normal(size=(repeat, z_sample.shape[0]))

    # fitting
    # param_fit, success, status = fit_param(z_sample, dTb_sample, cores=cores)
    start_time = time.time()
    # fit_param(z_sample, dTb_sample, cores=cores, average_dir=average_dir, delete_if_exists=delete_if_exists, save_name="m_chi{:.2f}_V_rms{:.0f}.npy".format(param_true[0], param_true[1]))
    fit_param(z_sample, dTb_sample, cores=cores, average_dir=average_dir, delete_if_exists=delete_if_exists, save_name="m_chi{}-V_rms{}.npy".format(param_true[0], param_true[1]), param_guess=param_guess, **kwargs)
    end_time = time.time()

    # np.savetxt("m_chi{:.2f}_V_rms{:.0f}.txt".format(param_true[0], param_true[1]), param_fits)

    # take the average
    # if param_fits.ndim <= 1:
    #     param_fit = np.array([np.average(param_fits, axis=0)])
    # else:
    #     param_fit = np.average(param_fits, axis=0)

    print(f"It costs {(end_time-start_time)/3600:.3f} hours to complete the calculation.")
    # print('success =', success)
    # print('status =', status)

    if False:
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
        plt.plot(z_sample, interp_dTb(param_fit, z_sample), label=r'$m_{\chi, \rm fit}$'+' = {:.3f} GeV'.format(param_fit[0]), linestyle=':', c='r')
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


# In[4]:

def estimate_N_random_v_stream(N = [10,500,1000,2000,3000,4000,5000]):
    data = np.load("average_dTb/V_rms19000.0/m_chi0.01.npy")
    # data = np.load("average_dTb/V_rms19000.0/m_chi0.1.npy")
    z_array = data[0]
    all_dTb_interp = data[1:]
    plt.subplot(121)
    dTb_averaged = [np.average(all_dTb_interp[:N[i]], axis=0)
                    for i in range(len(N))]
    # print(np.size(dTb_averaged))
    plt.title("z vs. dTb_averaged")
    for i in range(len(N)):
        plt.plot(z_array, dTb_averaged[i], label="N = {}".format(N[i]))
    plt.xlabel("z")
    plt.ylabel(r'$\overline{dTb}$ [mK]')
    plt.legend()
    # plt.xlim(0, 300)

    plt.subplot(122)
    N = np.arange(10, 5000, 100)
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


def demonstrate_depreciate(file_dir="average_dTb/V_rms29000.0/m_chi0.10", N=[100, 200, 300, 400, 500, 600, 700, 800,900,999]):
    """
    functions:
    1. show how many random velocities are required to achieve stable and accurate dTb_averaged;
    2. demonstrate that it is good enough to average dTb over 200 stream velocities.
    """
    z_array = np.linspace(10, 1010, 1000)

    try:
        file_names = os.listdir(file_dir)
    except FileNotFoundError:
        print(
            "Interrupt demonstrate() if you don't want following steps to cost long time.")
        average_dTb(m_chi=0.10, more_random_v_streams=400,
                    cores=1, verbose=False)
        file_names = os.listdir(file_dir)

    print("file_names[:5] =", file_names[:5])
    np.random.shuffle(file_names)
    print("After shuffling, file_names[:5] =", file_names[:5])
    for file_name in file_names:
        data = np.load(file_dir+"/{}".format(file_name))
        dTb_interp = np.interp(z_array, data[0][::-1], data[1][::-1])
        if "all_dTb_interp" not in vars():
            all_dTb_interp = dTb_interp.copy()
        else:
            all_dTb_interp = np.vstack((all_dTb_interp, dTb_interp))
    # print(all_dTb_interp.shape)

    plt.figure(figsize=(15, 4), dpi=150)

    plt.subplot(131)
    v_streams = np.array([float(name[:-4]) for name in file_names])
    plt.title(
        r"velocity distribution (3D gaussian with $V_{\rm rms} \equiv$ 29 km/s)")
    for i in range(len(N)):
        V_sample_rms = np.sqrt(np.average(v_streams[:N[i]]**2))/1000
        plt.hist(v_streams[:N[i]]/1000, label="N={}, ".format(N[i])+r"$\sqrt{\overline{V^2}}$" +
                 "={:.1f}km/s".format(V_sample_rms), density=True, histtype='step', bins=20)
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
    N = np.arange(10, 2000, 20)
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

    # for m_chi in np.logspace(-2, 0, 3):
    #     for V_rms in np.linspace(19000, 39000, 3):
    #         param_fits = test([m_chi, V_rms], cores=1, method="least_squares", repeat=20, noise=1, average_dir="average_dTb", adequate_random_v_streams=10, N_grid=[10,10])

    ######################################################################
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print("SLURM_ARRAY_TASK_ID", os.environ["SLURM_ARRAY_TASK_ID"])

    m_chi_array = np.logspace(-2, 1, 13)
    V_rms_array = np.linspace(20000, 40000, 9)
    parameters = list(product(m_chi_array, V_rms_array))

    myparam = parameters[idx]
    print("myparam =", myparam)
    param_fits = test(myparam, cores=-1, repeat=100, average_dir=f'average_dTb-{idx}-{myparam}', method="least_squares", noise=1, adequate_random_v_streams=48, N_grid=[100,100])
