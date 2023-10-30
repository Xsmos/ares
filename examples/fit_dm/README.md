# ARES
The Accelerated Reionization Era Simulations ([`ARES`](https://ares.readthedocs.io/en/latest/index.html)) code was designed to rapidly generate models for the global 21-cm signal.

Based on [Muñoz et al., 2015](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.92.083528), we added `ares/physics/DarkMatterHeating.py` to the original ARES to discuss the heating of baryons due to scattering with dark matter during the dark ages.

Following is a documentation about using `examples/fit_dm/test_fitting_dm.py` to fit dark matter mass, `m_chi`, and the R.M.S. of streaming velocity, `V_rms`, between dark matter and gas, given 21-cm temperature signals, `dTb`, and the correponding redshifts, `z`.

Be warned: this code is still under active development – use at your own risk! Correctness of results is not guaranteed.

## Citation
If you use this code in paper please reference [TBD]() if it’s an application of fitting dark matter mass and R.M.S. of streaming velocity.

## Dependencies
You will need:
- numpy
- scipy
- matplotlib
- h5py
## Getting started
To clone a copy and install:
```
git clone https://github.com/Xsmos/ares.git
cd ares
python setup.py install
```
`ares` will look in `ares/input` for lookup tables of various kinds. To download said lookup tables, run
```
python remote.py
```
Check out the original [**ares**](https://ares.readthedocs.io/en/latest/install.html) for more details about installing the lookup tables.
## Parameters:
- `z_sample`: a 1-D array of redshifts.
- `dTb_sample`: a 2-D array of brightness temperatures in unit of `mK`. `dTb_sample.shape[1]` must be equal to `z_sample.shape[0]`.
- `param_guess`: the initial guess for the parameters `[m_chi, V_rms]` in units of `GeV` and `m/s`, respectively. Default is `[0.1, 29000]`.
- `cores`: number of CPUs to calculate the `dTb`s for different initial streaming velocites. Default is 1.
- `average_dir`: name of the directory to save the averaged dTb's. Default is `'dTb_averaged'`.
- `save_name`: name of the file to save the fitting results. Default is `'m_chi-V_rms.npy'`.
- `N_v_streams`: number of initial streaming velocities to be generated equally-spaced between `[0, 3*V_rms]` in order to calculate the averaged global `dTb`. Default is 12.
- `verbose`: 0 for showing the final fitting results only, 1 for including intermediate results of the fitting process. Default is 1.
- `bounds`: Lower and upper bounds on independent variables. Default is `[[0.001,10000],[100,100000]]`.

## Example:
Please check out this [notebook](example.ipynb).