#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import ares
import os
import warnings
from datetime import datetime
from itertools import product
from scipy import interpolate
import multiprocessing
from multiprocessing import Pool
import shutil
from scipy.optimize import least_squares, curve_fit

def dTb_v_stream_list(m_chi=0.1, cores=1, V_rms=29000, average_dir='average_dTb', **kwargs):
    """
    generate initial_v_streams and calculate their 21cm temperatures with dark_matter_heating.
    """
    if 'verbose' not in kwargs:
        verbose = False
    else:
        verbose = kwargs['verbose']

    if "N_v_stream" not in kwargs:
        kwargs['N_v_stream'] = 3

    pf = \
        {
            'radiative_transfer': False,
            'verbose': False,
            'dark_matter_heating': True,
            'include_cgm': False,
            'initial_redshift': 1010,
            'include_He': True
        }

    initial_v_stream_list = np.linspace(1, 3*V_rms, kwargs['N_v_stream'])
    path = "{}/V_rms{}/m_chi{}".format(average_dir, V_rms, m_chi)
    if not os.path.exists(path):
        os.makedirs(path)

    start_time = datetime.now()
    if cores == 1:
        print(f"Sampling {kwargs['N_v_stream']} dTb's by 1 CPU...", end='')
        for i, initial_v_stream in enumerate(initial_v_stream_list):
            if verbose:
                print("\ninitial_v_stream =", initial_v_stream, 'm/s', end='')

            sim = ares.simulations.Global21cm(initial_v_stream=initial_v_stream, dark_matter_mass=m_chi, **pf)
            sim.run()

            np.save(path+"/{}".format(initial_v_stream), np.vstack((sim.history["z"], sim.history["dTb"])))

            number_of_CPUs = 1
    else:
        if cores == -1:
            cpu_count = multiprocessing.cpu_count()
        else:
            cpu_count = cores
        print(f"Sampling {kwargs['N_v_stream']} dTb's by {cpu_count} CPUs parallelly...", end='')
        global f_mpi

        def f_mpi(initial_v_stream):
            if verbose:
                print("\npid = {}, initial_v_stream = {} m/s".format(os.getpid(),
                      initial_v_stream), end='')
            
            sim = ares.simulations.Global21cm(initial_v_stream=initial_v_stream, dark_matter_mass=m_chi, **pf)
            sim.run()

            np.save(path+"/{}".format(initial_v_stream), np.vstack((sim.history["z"], sim.history["dTb"])))

            return os.getpid()

        with Pool(cpu_count) as p:
            pids = p.map(f_mpi, initial_v_stream_list)
        number_of_CPUs = np.unique(pids).size

    end_time = datetime.now()
    time_elapse = end_time - start_time
    print("\nIt costs {} to calculate dTb of {} different initial_v_streams by {} CPU(s).".format(
        time_elapse, kwargs['N_v_stream'], number_of_CPUs))

def Gaussian_3D(v, V_rms):
    P = np.exp(-3*v**2/(2*V_rms**2)) / (2*np.pi/3*V_rms**2)**(3/2)
    P_v_square = v**2 * P
    return P_v_square

def integrate_dTb_with_Probability(dTbs, file_names, V_rms):
    velocities = np.array([float(name[:-4]) for name in file_names])
    probabilities = Gaussian_3D(velocities, V_rms)
    dTb_averaged = dTbs.T@probabilities / probabilities.sum()
    return dTb_averaged

def average_dTb(m_chi=0.1, N_z=1000, cores=1, V_rms=29000, average_dir="average_dTb", **kwargs):
    warnings.simplefilter("ignore", UserWarning)
    if 'N_v_stream' not in kwargs:
        kwargs['N_v_stream'] = 10
    if 'verbose' not in kwargs:
        verbose = kwargs['verbose'] = False

    path = "{}/V_rms{}/m_chi{}".format(average_dir, V_rms, m_chi)
    if not os.path.exists(path+'.npy') or kwargs["N_v_stream"]:
        dTb_v_stream_list(m_chi, cores=cores, V_rms=V_rms, average_dir=average_dir, **kwargs)

    file_names = os.listdir(path)

    z_array = np.linspace(5, 1010, N_z)

    for file_name in file_names:
        data = np.load(path+"/{}".format(file_name))
        dTb_interp = np.interp(z_array, data[0][::-1], data[1][::-1])
        if "all_dTb_interp" not in vars():
            all_dTb_interp = dTb_interp.copy()
        else:
            all_dTb_interp = np.vstack((all_dTb_interp, dTb_interp))

    if not os.path.exists(path+'.npy'):
        np.save(path, np.vstack((z_array, all_dTb_interp)))
    else:
        old_data = np.load(path+'.npy')
        new_data = np.vstack((old_data, all_dTb_interp))
        np.save(path, new_data)

    data = np.load(path+'.npy')
    dTb_averaged = integrate_dTb_with_Probability(data[1:], file_names, V_rms)
    np.save(path+"_averaged".format(m_chi), np.vstack((z_array, dTb_averaged)))
    shutil.rmtree(path, ignore_errors=True)
    return (z_array, dTb_averaged, m_chi, V_rms)

def interp_dTb(param, z, cores=1, average_dir="average_dTb", **kwargs):
    """
    functions:
    1. generate adequate random stream velocities subject to 3D Gaussian distribution;
    2. average dTb over these velocities;
    3. interpolate dTb by input 'z'.
    """
    m_chi, V_rms = param

    if "N_v_stream" not in kwargs:
        kwargs['N_v_stream'] = 24

    N_v_stream = kwargs['N_v_stream']
    directory = "{}/V_rms{}/m_chi{}".format(average_dir, V_rms, m_chi)
    if os.path.exists(directory+'.npy'):
        data = np.load(directory+'.npy')
        if data.shape[0]-1 == kwargs['N_v_stream'] and os.path.exists(directory+'_averaged.npy'):
            z_array, dTb_averaged = np.load(directory+'_averaged.npy')
            N_v_stream = 0
            print("Existing averaged dTb and z are loaded for m_chi = {} GeV and V_rms = {} m/s.".format(m_chi, V_rms))

    if N_v_stream:
        print("{} v_streams will be generated for m_chi = {} GeV and V_rms = {} m/s...".format(N_v_stream, m_chi, V_rms))
        z_array, dTb_averaged, m_chi, V_rms = average_dTb(m_chi=m_chi, cores=cores, V_rms=V_rms, average_dir=average_dir, **kwargs)
    
    print("---"*15)
    dTb = np.interp(z, z_array, dTb_averaged)
    return dTb

def residual(param, z_sample, dTb_sample, kwargs, cores=1, average_dir="average_dTb"):
    residual = interp_dTb(param, z_sample, cores, average_dir=average_dir, **kwargs) - dTb_sample
    return residual

def fit_param(z_sample, dTb_sample, param_guess=[0.1, 29000], cores=1, average_dir='average_dTb', save_name="fitted_m_chi_V_rms.npy", **kwargs):
    '''
    fit the parameter(s) by z_sample and dTb_sample via scipy.optimize.least_squares.
    '''        
    fit_start = datetime.now()
    if "bounds" in kwargs:
        bounds = kwargs['bounds']
    else:
        bounds = kwargs['bounds'] = np.array([[0.001,10000],[100,100000]])

    if "N_grid" in kwargs:
        N_grid = kwargs['N_grid']
    else:
        N_grid = kwargs['N_grid'] = (5,5)

    warnings.simplefilter("ignore", UserWarning)

    if z_sample.ndim == 1 and dTb_sample.ndim != 1:
        z_sample = np.tile(z_sample, (dTb_sample.shape[0], 1))
    elif z_sample.ndim != 1:
        if z_sample.shape != dTb_sample.shape:
            print("z_sample and dTb_sample should have same shape.")
            return

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

        start_time = datetime.now()
        res = least_squares(residual, param_guess, bounds=bounds, args=(args_z, args_dTb, kwargs, cores, average_dir), diff_step=0.1)
        theta_fit = res.x
        if res.success == False:
            continue
        end_time = datetime.now()

        print('#{}'.format(i+1), ', fit:', theta_fit, ', success:', res.success, ', status:', res.status, f', cost {end_time-start_time}')

        try:
            pre_data = np.load(save_name)
            data_updated = np.vstack((pre_data, theta_fit))
            np.save(save_name, data_updated)
        except FileNotFoundError:
            np.save(save_name, theta_fit)
        print('---'*30)
    fit_end = datetime.now()
    print(f"The fitting costs {fit_end-fit_start} for {dTb_sample.shape[1]} observations.")
    return

def test(param_true=[0.15, 29000], noise=0.01, cores=-1, z_sample=np.arange(10, 800, 1), stop_plot=5, repeat=20, plot=True, average_dir="average_dTb", delete_if_exists=False, param_guess=[0.1, 29000], **kwargs):
    """
    functions:
    1. test the fit_param();
    2. showed that fit_param() works well for m_chi < 1 GeV. 
    """
    print("param_true =", param_true)
        
    # sampling
    dTb_accurate = interp_dTb(param_true, z_sample, cores, average_dir=average_dir, **kwargs)
    dTb_sample = dTb_accurate + noise * np.random.normal(size=(repeat, z_sample.shape[0]))

    start_time = datetime.now()
    fit_param(z_sample, dTb_sample, cores=cores, average_dir=average_dir, delete_if_exists=delete_if_exists, save_name="m_chi{}-V_rms{}.npy".format(param_true[0], param_true[1]), param_guess=param_guess, **kwargs)
    end_time = datetime.now()

    print(f"It costs {end_time-start_time} to complete the calculation.")

if __name__ == '__main__':
    # for m_chi in np.logspace(-2, 0, 3):
    #     for V_rms in np.linspace(19000, 39000, 3):
    #         param_fits = test([m_chi, V_rms], cores=1, repeat=20, noise=1, average_dir="average_dTb", N_v_stream=10, N_grid=[10,10])

    ######################################################################
    idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
    print("SLURM_ARRAY_TASK_ID", os.environ["SLURM_ARRAY_TASK_ID"])

    m_chi_array = np.logspace(-2, 1, 13)
    V_rms_array = np.linspace(20000, 40000, 9)
    parameters = list(product(m_chi_array, V_rms_array))

    myparam = parameters[idx]
    print("myparam =", myparam)
    param_fits = test(myparam, cores=-1, repeat=100, average_dir=f'average_dTb-{idx}-{myparam}', noise=1, N_v_stream=48, N_grid=[100,100])
