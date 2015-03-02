"""

Global21cm.py

Author: Jordan Mirocha
Affiliation: University of Colorado at Boulder
Created on: Wed Sep 24 14:55:35 MDT 2014

Description: 

"""

import numpy as np
from ..static import Grid
import copy, os, re, pickle
from ..sources import DiffuseSource
from ..util.ReadData import load_inits
from ..util.WriteData import CheckPoints
from ..util.ManageHistory import WriteData
from ..util.PrintInfo import print_21cm_sim
from ..populations import CompositePopulation
from ..solvers.RadiationField import RadiationField
from ..solvers.UniformBackground import UniformBackground
from ..util.SetDefaultParameterValues import SetAllDefaults
from ..util import ProgressBar, RestrictTimestep, ParameterFile
from ..physics.Constants import k_B, m_p, G, g_per_msun, c, sigma_T, \
    erg_per_ev, nu_0_mhz

try:
    import h5py
    have_h5py = True
except ImportError:
    have_h5py = False

try:
    from scipy.interpolate import interp1d
except ImportError:
    pass

try:
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
except ImportError:
    rank = 0
    size = 1   
    
HOME = os.environ.get('HOME')
ARES = os.environ.get('ARES')

class Global21cm:
    def __init__(self, **kwargs):
        """
        Set up a two-zone model for the global 21-cm signal.

        See Also
        --------
        Set of all acceptable kwargs in:
            ares/util/SetDefaultParameterValues.py
            
        """
        
        if kwargs:
            
            if 'tanh_model' not in kwargs:
                self.pf = ParameterFile(**kwargs)
            else:
                if kwargs['tanh_model']:
                    from ..util.TanhModel import TanhModel

                    tanh_model = TanhModel(**kwargs)
                    self.pf = tanh_model.pf
                    
                    if self.pf['tanh_nu'] is not None:
                        nu = self.pf['tanh_nu']
                        z = nu_0_mhz / nu - 1.
                    else:
                        z = np.arange(self.pf['final_redshift'] + self.pf['tanh_dz'],
                            self.pf['initial_redshift'], self.pf['tanh_dz'])[-1::-1]
                    
                    self.history = tanh_model(z, **self.pf).data

                    self.grid = Grid(dims=1)
                    self.grid.set_cosmology(
                        initial_redshift=self.pf['initial_redshift'], 
                        omega_m_0=self.pf["omega_m_0"], 
                        omega_l_0=self.pf["omega_l_0"], 
                        omega_b_0=self.pf["omega_b_0"], 
                        hubble_0=self.pf["hubble_0"], 
                        helium_by_number=self.pf['helium_by_number'], 
                        cmb_temp_0=self.pf["cmb_temp_0"],
                        approx_highz=self.pf["approx_highz"])

                    return
                    
                else:
                    self.pf = ParameterFile(**kwargs)

        else:
            self.pf = ParameterFile()
                    
        # Check for identical realization
        self.found_sim = False
        if self.pf['load_sim'] and os.path.exists('%s/.ares' % HOME):
            for fn in os.listdir('%s/.ares' % HOME):
                if not re.search('.pkl', fn):
                    continue
                
                f = open('%s/.ares/%s' % (HOME, fn), 'rb')
                pf = pickle.load(f)
                f.close()
                
                if pf == self.pf:
                    break
            
            self.found_sim = True
            
            prefix = fn.partition('.')[0]
            self.history = dict(np.load('%s/.ares/%s.npz' % (HOME, prefix)))
            
            if rank == 0:
                print "\nFound identical realization! Loaded %s/.ares/%s.npz" \
                    % (HOME, prefix)
            return 
                        
        # Initialize two grid patches   
        self.grid_igm = Grid(dims=1, approx_Salpha=self.pf['approx_Salpha'])
        self.grid_cgm = Grid(dims=1)
        self.grids = [self.grid_igm, self.grid_cgm]
            
        # If we just need access to cosmology, use igm grid (arbitrarily)
        self.grid = self.grid_igm    
            
        # Set physics
        self.grid_igm.set_physics(isothermal=0, compton_scattering=1, 
            secondary_ionization=0, expansion=1, recombination='B')
        self.grid_cgm.set_physics(isothermal=1, compton_scattering=0, 
            secondary_ionization=0, expansion=1, recombination='A',
            clumping_factor=self.pf['clumping_factor'])        
            
        # Artificially turn off recombination (perhaps unnecessary)
        if self.pf['recombination'] == 0:
            self.grid_igm._recombination = 0
            self.grid_cgm._recombination = 0
            
        x0 = self.pf['initial_ionization']
        if ARES and self.pf['load_ics']:
            if have_h5py:
                inits_path = os.path.join(ARES,'input','inits','initial_conditions.hdf5')
            else:
                inits_path = os.path.join(ARES,'input','inits','initial_conditions.npz')
                
            if os.path.exists(inits_path):
                inits = self.inits = load_inits(inits_path)
                
                T0 = np.interp(self.pf['initial_redshift'], inits['z'],
                    inits['Tk'])
                xe = np.interp(self.pf['initial_redshift'], inits['z'],
                    inits['xe'])
                   
                if self.pf['include_He']:
                    x0 = [xe, 1e-8]
                else:
                    x0 = [min(xe, 1.0)]

                self.inits_path = inits_path
            else:
                raise NotImplementedError('Something going on.')
        else:
            if self.pf['include_He']:
                x0 = [self.pf['initial_ionization'][0]] * 2 
            else:
                x0 = [self.pf['initial_ionization'][0]]
                
            # Set to adiabatic temperature if no inits supplied
            T0 = self.grid.cosm.Tgas(self.pf['initial_redshift'])
                                                 
        # Set cosmological initial conditions  
        for grid in self.grids:  
            grid.set_cosmology(initial_redshift=self.pf['initial_redshift'], 
                omega_m_0=self.pf["omega_m_0"], 
                omega_l_0=self.pf["omega_l_0"], 
                omega_b_0=self.pf["omega_b_0"], 
                hubble_0=self.pf["hubble_0"], 
                helium_by_number=self.pf['helium_by_number'], 
                cmb_temp_0=self.pf["cmb_temp_0"], 
                approx_highz=self.pf["approx_highz"])
                
            grid.set_chemistry(include_He=self.pf['include_He'])
            grid.set_density(grid.cosm.nH(self.pf['initial_redshift']))

        self.helium = self.pf['include_He']

        self.grid_igm.set_temperature(T0)
        self.grid_cgm.set_temperature(1.e4)
        self.grid_cgm.set_recombination_rate(in_bubbles=True)

        for i, Z in enumerate(self.grid.Z):
            self.grid_igm.set_ionization(Z=Z, x=x0[i])
            self.grid_cgm.set_ionization(Z=Z, x=1e-8)
                                                
        self.grid_igm.data['n'] = \
            self.grid_igm.particle_density(self.grid_igm.data, 
            z=self.pf['initial_redshift'])
        self.grid_cgm.data['n'] = \
            self.grid_cgm.particle_density(self.grid_cgm.data, 
            z=self.pf['initial_redshift'])

        # To compute timestep
        self.timestep_igm = RestrictTimestep(self.grid_igm, 
            self.pf['epsilon_dt'], self.pf['verbose'])
        self.timestep_cgm = RestrictTimestep(self.grid_cgm, 
            self.pf['epsilon_dt'], self.pf['verbose'])
            
        # For regulating time/redshift steps
        self.checkpoints = CheckPoints(pf=self.pf, 
            grid=self.grid,
            time_units=self.pf['time_units'],
            initial_redshift=self.pf['initial_redshift'],
            final_redshift=self.pf['final_redshift'],
            dzDataDump=self.pf['dzDataDump'],
            dtDataDump=self.pf['dtDataDump'],
            )
            
        ##
        # PRINT STUFF!
        ##
        print_21cm_sim(self)    

        # Initialize radiation sources / populations
        if self.pf["radiative_transfer"]:
            self._init_RT(self.pf)
        else:
            self.approx_all_lw = 1
            self.approx_all_xray = 1  # will be updated in _init_RT
            self.srcs = None
            
        if type(self.pf['feedback']) in [bool, int]:
            try:
                self.feedback = [self.pf['feedback']] * len(self.pops.pops)    
            except AttributeError:
                self.feedback = [self.pf['feedback']]
        else:
            self.feedback = self.pf['feedback']
            
        self.feedback_ON = sum(self.feedback) > 0    
        
        # Initialize radiative transfer solver
        self.rt_igm = RadiationField(self.grid_igm, self.srcs, **self.pf)
        self.rt_cgm = RadiationField(self.grid_cgm, self.srcs, **self.pf)
            
        self.rt_cgm.chem.chemnet.SourceIndependentCoefficients(T=1.e4)    
            
        # Set up X-ray flux generator
        if not self.approx_all_xray:
            if self.pf['EoR_xavg'] == 0:
                raise NotImplemented('This needs work (EoR_xavg == 0)')
                self._init_XRB(pre_EoR=True)
                self._init_XRB(pre_EoR=False)
            else:
                self._init_XRB()
            
        if not self.approx_all_lw:
            self._init_LWB()    
            
        if self.pf['track_extrema']:
            from ..analysis.TurningPoints import TurningPoints
            self.track = TurningPoints(inline=True, **self.pf)    
        
        # should raise error if different tau tables passed to each source.    
    
        self.write = WriteData(self)
        
        # Create dictionary for output - store initial conditions
        self.history = self.write._initialize_history()
    
    def _init_RT(self, pf, use_tab=True):
        """
        Initialize astrophysical populations & radiation backgrounds.
        """
        
        self.pops = CompositePopulation(**pf)

        if len(self.pops.pops) == 1:
            self.pop = self.pops.pops[0]
                        
        # Loop over populations, make separate RB and RS instances for each
        self.rbs = [UniformBackground(pop) for pop in self.pops.pops]
        
        self.Nrbs = len(self.rbs)
        
        self.approx_all_xray = 1
        self.approx_all_lw  = 1
        
        for rb in self.rbs:
            self.approx_all_xray *= rb.pf['approx_xrb']
            self.approx_all_lw *= rb.pf['approx_lwb']
        
        # Don't have to do this...could just stick necessary attributes in RB
        self.srcs = [DiffuseSource(rb) for rb in self.rbs]

    def _init_LWB(self):
        """ Setup cosmic Lyman-Werner background calculation. """

        self._Jrb = []
        for rb in self.rbs:
            if (not rb.pf['is_lya_src']) or (not rb.pf['discrete_lwb'])\
                or rb.pf['approx_lwb']:
                self._Jrb.append(None)
                continue

            lwb_z, lwb_En, lwb_J = rb.LWBackground()
            self._Jrb.append((lwb_z, lwb_En, lwb_J))

        # Sum up Lyman-alpha flux
        self.lwb_z, self.lwb_En = lwb_z, lwb_En

        self.lwb_Ja = []
        for i, rb in enumerate(self.rbs):
            if self._Jrb[i] is None:
                self.lwb_Ja.append(np.zeros_like(self.lwb_z))
                continue

            self.lwb_Ja.append(rb.LymanAlphaFlux(fluxes=self._Jrb[i][-1]))

    def _init_XRB(self, pre_EoR=True, **kwargs):
        """ Setup cosmic X-ray background calculation. """
                
        if pre_EoR:
            self.pre_EoR = True
            
            # Store XrayFluxGenerators
            self.cxrb_gen = [None for i in range(self.Nrbs)]
            for i, rb in enumerate(self.rbs):
                if rb.pf['approx_xrb']:
                    continue

                self.cxrb_gen[i] = rb.XrayFluxGenerator(rb.igm.tau)

                # All UniformBackgrounds must share these properties
                if not hasattr(self, 'cxrb_shape'):
                    self.cxrb_shape = (rb.igm.L, rb.igm.N)
                if not hasattr(self, 'zmin_igm'):
                    self.zmin_igm = rb.igm.z[0]
                    self.cxrb_zall = rb.igm.z.copy()

            # Save X-ray background incrementally
            self.xray_flux = [[] for i in range(self.Nrbs)]
            self.xray_heat = [[] for i in range(self.Nrbs)]

            # Generate fluxes at first two redshifts
            fluxes_lo = []; fluxes_hi = []
            for cxrb in self.cxrb_gen:

                if cxrb is None:
                    fluxes_hi.append(0.0)
                    continue

                fluxes_hi.append(cxrb.next())

            for cxrb in self.cxrb_gen:
                if cxrb is None:
                    fluxes_lo.append(0.0)
                    continue
                    
                fluxes_lo.append(cxrb.next())
                    
            self.cxrb_flo = fluxes_lo
            self.cxrb_lhi = self.cxrb_shape[0] - 1
            self.cxrb_llo = self.cxrb_shape[0] - 2
            
            self.cxrb_fhi = fluxes_hi
            
            # Figure out first two redshifts
            for i, rb in enumerate(self.rbs):
                if rb.pop.pf['approx_xrb']:
                    continue

                self.cxrb_zhi = rb.igm.z[-1]
                self.cxrb_zlo = rb.igm.z[-2]
                
                break
            
            # Store first two fluxes, separate by rb instance    
            for i, rb in enumerate(self.rbs):
                self.xray_flux[i].extend([fluxes_hi[i], fluxes_lo[i]])
        else:
            for i, rb in enumerate(self.rbs):
                if rb.pf['approx_xrb']:
                    continue
                    
                self.cxrb_shape = (rb.igm.L, rb.igm.N)     
                
                self.zmin_igm = rb.igm.z[0]
                self.cxrb_zall = rb.igm.z.copy()
                
                self.cxrb_zhi = rb.igm.z[-1]
                self.cxrb_zlo = rb.igm.z[-2]
                self.cxrb_lhi = self.cxrb_shape[0] - 1
                self.cxrb_llo = self.cxrb_shape[0] - 2
                
                # Done in ComputeXRB upon switch to EoR
                #self.cxrb_flo = fluxes_lo
                #self.cxrb_fhi = fluxes_hi
                
                break
                
            # Save X-ray background incrementally
            if not hasattr(self, 'xray_flux'):
                self.xray_flux = [[] for i in range(self.Nrbs)]
            if not hasattr(self, 'xray_heat'):
                self.xray_heat = [[] for i in range(self.Nrbs)]            

        # Store rate coefficients by (source, absorber)
        self.cxrb_hlo = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_G1lo = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_G2lo = np.zeros([self.Nrbs, self.grid.N_absorbers,
            self.grid.N_absorbers])
        self.cxrb_hhi = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_G1hi = np.zeros([self.Nrbs, self.grid.N_absorbers])
        self.cxrb_G2hi = np.zeros([self.Nrbs, self.grid.N_absorbers,
            self.grid.N_absorbers])
        
        if self.pf['secondary_lya']:
            self.cxrb_JXlo = np.zeros([self.Nrbs, self.grid.N_absorbers])
            self.cxrb_JXhi = np.zeros([self.Nrbs, self.grid.N_absorbers])
        
        for i, rb in enumerate(self.rbs):
            
            self.xray_heat[i].extend([self.cxrb_hhi[i], self.cxrb_hlo[i]])
            
            for j, absorber in enumerate(self.grid.absorbers):
            
                if rb.pop.pf['approx_xrb']:
                    self.cxrb_hlo[i,j] = \
                        rb.igm.HeatingRate(self.cxrb_zlo, return_rc=True, **kwargs)
                    self.cxrb_hhi[i,j] = \
                        rb.igm.HeatingRate(self.cxrb_zhi, return_rc=True, **kwargs)
                    self.cxrb_G1lo[i,j] = \
                        rb.igm.IonizationRateIGM(self.cxrb_zlo, return_rc=True, **kwargs)
                    self.cxrb_G1hi[i,j] = \
                        rb.igm.IonizationRateIGM(self.cxrb_zhi, return_rc=True, **kwargs)
                    
                    for k, donor in enumerate(self.grid.absorbers):
                        self.cxrb_G2lo[i,j,k] = \
                            rb.igm.SecondaryIonizationRateIGM(self.cxrb_zlo, 
                                species=j, donor=k, return_rc=True, **kwargs)
                        self.cxrb_G2hi[i,j,k] = \
                            rb.igm.SecondaryIonizationRateIGM(self.cxrb_zhi, 
                                species=j, donor=k, return_rc=True, **kwargs)
                    
                    continue
                    
                if self.pf['secondary_lya']:
                    self.cxrb_JXlo[i,j] = \
                        rb.igm.DiffuseLymanAlphaFlux(self.cxrb_zlo, species=j,
                        xray_flux=self.cxrb_flo[i], return_rc=True, **kwargs)
                    self.cxrb_JXhi[i,j] = \
                        rb.igm.DiffuseLymanAlphaFlux(self.cxrb_zhi, species=j,
                        xray_flux=self.cxrb_fhi[i], return_rc=True, **kwargs)
                    
                # Otherwise, compute heating etc. from background intensity    
                self.cxrb_hlo[i,j] = \
                    rb.igm.HeatingRate(self.cxrb_zlo, species=j,
                    xray_flux=self.cxrb_flo[i], return_rc=True, **kwargs)
                self.cxrb_hhi[i,j] = \
                    rb.igm.HeatingRate(self.cxrb_zhi, species=j,
                    xray_flux=self.cxrb_fhi[i], return_rc=True, **kwargs)
                
                # IGM ionization
                self.cxrb_G1lo[i,j] = \
                    rb.igm.IonizationRateIGM(self.cxrb_zlo, species=j,
                    xray_flux=self.cxrb_flo[i], return_rc=True, **kwargs)
                self.cxrb_G1hi[i,j] = \
                    rb.igm.IonizationRateIGM(self.cxrb_zhi, species=j,
                    xray_flux=self.cxrb_fhi[i], return_rc=True, **kwargs)
                                
                if self.pf['secondary_ionization'] > 0:
                    for k, donor in enumerate(self.grid.absorbers):
                        self.cxrb_G2lo[i,j,k] = \
                            rb.igm.SecondaryIonizationRateIGM(self.cxrb_zlo,
                            species=j, donor=k,
                            xray_flux=self.cxrb_flo[i], return_rc=True, **kwargs)
                        self.cxrb_G2hi[i,j,k] = \
                            rb.igm.SecondaryIonizationRateIGM(self.cxrb_zhi,
                            species=j, donor=k,
                            xray_flux=self.cxrb_fhi[i], return_rc=True, **kwargs)
                
    def run(self):
        """Just another pathway to our __call__ method. Runs simulation."""
        self.__call__()
     
    def rerun(self):
        del self.history
        self.run() 
     
    def __call__(self):
        """ Evolve chemistry and radiation background. """
        
        if self.pf['tanh_model']:
            self.run_inline_analysis()
            return
        
        t = 0.0
        dt = self.pf['initial_timestep'] * self.pf['time_units']
        z = self.pf['initial_redshift']
        zf = self.pf['final_redshift']
        zfl = self.zfl = self.pf['first_light_redshift']
        
        if self.pf["dzDataDump"] is not None:
            dz = self.pf["dzDataDump"]
        else:
            dz = dt / self.grid.cosm.dtdz(z)
        
        # Read initial conditions
        data_igm = self.grid_igm.data.copy()
        data_cgm = self.grid_cgm.data.copy()

        # Initialize progressbar
        self.tf = self.grid.cosm.LookbackTime(zf, z)
        self.pb = ProgressBar(self.tf, '21-cm (pre-EoR)',
            use=self.pf['progress_bar'])
        self.pb.start()

        self.step = 0

        fields = ['h_1', 'h_2', 'e']
        if self.helium:
            fields.extend(['he_1', 'he_2', 'he_3'])

        # Evolve to final redshift
        while z > zf:
                        
            # kwargs specific to bulk IGM grid patch
            kwargs = {'igm': True, 'return_rc': True}
            
            # Make temporary arrays in order of ascending redshift 
            # (required by interpolation routines we use)
            ztmp = self.history['z'][-1::-1]
            xtmp = self.history['xavg'][-1::-1]

            # Need to pass ionization state to both grids
            to_solver = {}

            # Add all hydrogen (and possibly helium) fractions
            for field in fields:
                to_solver['cgm_%s' % field] = self.history['cgm_%s' % field][-1]
                to_solver['igm_%s' % field] = self.history['igm_%s' % field][-1]

            kwargs.update(to_solver)

            # Compute X-ray background flux
            self.ComputeXRB(z, ztmp, xtmp, **to_solver)
                                    
            # Grab ionization/heating rates from full X-ray background calculation
            if not self.approx_all_xray:
                
                # Sum over sources, but keep sorted by absorbing/donor species
                hlo = np.sum(self.cxrb_hlo, axis=0)
                hhi = np.sum(self.cxrb_hhi, axis=0)
                G1lo = np.sum(self.cxrb_G1lo, axis=0)
                G1hi = np.sum(self.cxrb_G1hi, axis=0)                
                                
                # Interpolate to current time                
                H = [np.interp(z, [self.cxrb_zlo, self.cxrb_zhi], 
                    [hlo[i], hhi[i]]) for i in range(self.grid.N_absorbers)]
                G1 = [np.interp(z, [self.cxrb_zlo, self.cxrb_zhi], 
                    [G1lo[i], G1hi[i]]) for i in range(self.grid.N_absorbers)]
                                
                G2 = np.zeros([self.grid.N_absorbers]*2)
                if self.pf['secondary_ionization'] > 0:                    
                    G2lo = np.sum(self.cxrb_G2lo, axis=0)
                    G2hi = np.sum(self.cxrb_G2hi, axis=0)
                                        
                    for ii in range(self.grid.N_absorbers):
                        for jj in range(self.grid.N_absorbers):
                            G2[ii,jj] = np.interp(z, 
                                [self.cxrb_zlo, self.cxrb_zhi],
                                [G2lo[ii,jj], G2hi[ii,jj]])
                                                                
                if self.pf['secondary_lya']:
                    JXlo = np.sum(self.cxrb_JXlo, axis=0)
                    JXhi = np.sum(self.cxrb_JXhi, axis=0)
                    
                    JX = [np.interp(z, [self.cxrb_zlo, self.cxrb_zhi], 
                        [JXlo[i], JXhi[i]]) for i in range(self.grid.N_absorbers)]          
                else:
                    JX = 0.0
                                      
                # This stuff goes to RadiationField (diffuse sources only)
                kwargs.update({'epsilon_X': np.array(H), 
                    'Gamma': np.array(G1), 
                    'gamma': np.array(G2),
                    'Ja_X': np.array(JX)})
                                  
            # Solve for xe and Tk in the bulk IGM
            data_igm = self.rt_igm.Evolve(data_igm, t=t, dt=dt, z=z, **kwargs)

            # Next up: bubbles
            kwargs.update({'igm': False})
            
            # Gamma etc. are only for the bulk IGM - lose them for bubbles!
            if 'Gamma' in kwargs:
                kwargs.pop('Gamma')
            if 'gamma' in kwargs:
                kwargs.pop('gamma')
            if 'epsilon_X' in kwargs:
                kwargs.pop('epsilon_X')
            if 'Ja_X' in kwargs:
                kwargs.pop('Ja_X')    
                                
            # Solve for the volume filling factor of HII regions
            if self.pf['radiative_transfer'] and (z <= zfl):
                data_cgm = self.rt_cgm.Evolve(data_cgm, t=t, dt=dt, z=z, **kwargs)

            # Increment time and redshift
            zpre = z

            t += dt
            z -= dt / self.grid.cosm.dtdz(z)

            # Evolve LW background and compute Lyman-alpha flux
            if self.pf['radiative_transfer'] and z < self.zfl:

                Ja = []
                for i, rb in enumerate(self.rbs):
                    if rb is None:
                        Ja.append(0.0)
                    elif hasattr(self, 'lwb_Ja'):
                        Ja.append(np.interp(z, self.lwb_z, self.lwb_Ja[i]))
                    else:
                        Ja.append(rb.LymanAlphaFlux(z))

            else:
                Ja = [0.0]

            # Add Ja to history even though it didn't come out of solver
            data_igm['Ja'] = np.array(Ja)

            # SAVE RESULTS
            self.write._update_history(z, zpre, data_igm, data_cgm)

            if z <= zf:
                break

            ##
            # FEEDBACK: Modify Tmin depending on IGM temperature and/or LW flux
            ##
            # Someday!
            
            # Inline analysis: possibly kill calculation if we've passed
            # a turning point, zero-crossing, or critical ionization fraction.
            # See "stop" parameter for more information.             
            if self.pf['track_extrema']:
                stop = self.track.is_stopping_point(self.history['z'],
                    self.history['dTb'])
                if stop:
                    break

            ##
            # TIME-STEPPING FROM HERE ON OUT
            ##                 
                                            
            # Figure out next dt based on max allowed change in evolving fields
            new_dt_igm = \
                self.timestep_igm.Limit(self.rt_igm.chem.q_grid.squeeze(),
                self.rt_igm.chem.dqdt_grid.squeeze(), z=z,
                method=self.pf['restricted_timestep'])
            
            if (z + dt / self.grid.cosm.dtdz(z)) <= self.pf['first_light_redshift'] and \
                self.pf['radiative_transfer']:
                
                new_dt_cgm = \
                    self.timestep_cgm.Limit(self.rt_cgm.chem.q_grid.squeeze(), 
                    self.rt_cgm.chem.dqdt_grid.squeeze(), z=z,
                    method=self.pf['restricted_timestep'])
            else:
                new_dt_cgm = 1e50
                            
            # Limit timestep further based on next DD and max allowed increase
            dt = min(min(new_dt_igm, new_dt_cgm), 2*dt)
            dt = min(dt, self.checkpoints.next_dt(t, dt))
            dt = min(dt, self.pf['max_dt'] * self.pf['time_units'])

            # Limit timestep based on next RD
            if self.checkpoints.redshift_dumps:
                dz = min(self.checkpoints.next_dz(z, dz), self.pf['max_dz'])
                dt = min(dt, dz*self.grid.cosm.dtdz(z))
                
            if self.pf['max_dz'] is not None:
                dt = min(dt, self.pf['max_dz']*self.grid.cosm.dtdz(z))    
                
            # Limit redshift step by next element in flux generator
            if not self.approx_all_xray and (z > self.zmin_igm):                
                dtdz = self.grid.cosm.dtdz(z)
                
                if (z - dt / dtdz) < self.cxrb_zall[max(self.cxrb_llo-1, 0)]:
                    dz = (z - self.cxrb_zall[self.cxrb_llo-1]) * 0.5
                    
                    # Means we're on the last step
                    if dz < 0:
                        dz = z - self.zmin_igm
                                                
                    dt = dz*dtdz
            
            self.pb.update(t)

            # If reionization is ~complete (xavg = 0.99999 by default),
            # just quit, but fill in history down to final_redshift
            if self.history['xavg'][-1] >= self.pf['stop_xavg']:
                
                zrest = np.arange(self.pf['final_redshift'], z, 0.05)[-1::-1]
                self.history['z'].extend(list(zrest))
                
                for key in self.history:
                    if key == 'z':
                        continue

                    v_current = self.history[key][-1]

                    for j, redshift in enumerate(zrest):
                        self.history[key].append(v_current)

                break
                
            self.step += 1

        self.pb.finish()

        tmp = {}
        for key in self.history:
            tmp[key] = np.array(self.history[key])

        self.history = tmp
            
        if self.pf['track_extrema']:
            self.turning_points = self.track.turning_points
    
        self.run_inline_analysis()
    
    def run_inline_analysis(self):    
        
        if (self.pf['inline_analysis'] is None) and \
           (self.pf['auto_generate_blobs'] == False):
            return
            
        tmp = {}
        for key in self.history:
            if type(self.history[key]) is list:
                tmp[key] = np.array(self.history[key])
            else:
                tmp[key] = self.history[key]
            
        self.history = tmp
            
        from ..analysis.InlineAnalysis import InlineAnalysis
        anl = InlineAnalysis(self)
        anl.run_inline_analysis()
        
        self.turning_points = anl.turning_points
        
        self.blobs = anl.blobs
        self.blob_names, self.blob_redshifts = \
            anl.blob_names, anl.blob_redshifts

    @property
    def binfo(self):
        if hasattr(self, 'blob_names'):
            bn = self.blob_names
        else:
            bn = None
        
        if hasattr(self, 'blob_redshifts'):
            bz = self.blob_redshifts
        else:
            bz = None    
        
        return (bn, bz)

    @property
    def blob_shape(self):
        """
        Only accessible *after* the inline analysis has been run.
        """
        if not hasattr(self, '_blob_shape'):
            if hasattr(self, 'blob_names'):
                self._blob_shape = \
                    np.zeros([len(self.blob_redshifts), len(self.blob_names)])
            else:
                self._blob_shape = None

        return self._blob_shape

    def tabulate_blobs(self, z):
        """
        Print blobs at a particular redshift (nicely).
        """
        
        print "-" * 50
        print "par                       value "
        print "-" * 50
        
        for k, par in enumerate(self.blob_names):
            i = self.blob_redshifts.index(z)
            j = list(self.blob_names).index(par)
                        
            print "%-25s %-8.4g" % (par, self.blobs[i,j])
            
    def extract_blob(self, name, z):
        """
        Extract individual result of automatic analysis.
        
        Parameters
        ----------
        name : str
            Name of quantity you'd like returned.
        z : int, float, str
            Redshift of interest. Can pass 'B', 'C', or 'D' to return the 
            quantity of interest at given extrema in global 21-cm signal.
            
        Returns
        -------
        Value of quantity `name` at redshift `z`.
        
        See Also
        --------
        Keyword arguments ``inline_analysis`` and ``auto_generate_blobs``.
        
        """
        i = self.blob_redshifts.index(z)
        j = list(self.blob_names).index(name)
        
        return self.blobs[i,j]
        
    def ComputeXRB(self, z, ztmp, xtmp, **kwargs):
        """
        Compute cosmic X-ray background flux.

        Parameters
        ----------
        z : float
            Current (observer) redshift
        ztmp : numpy.ndarray
            Array of redshifts (in ascending redshift).
        xtmp : numpy.ndarray
            Array of mean ionized fractions correspond to redshifts in ztmp.
        """
        
        if self.approx_all_xray:
            return

        if not self.pf['radiative_transfer']:
            return
            
        if z > self.pf['first_light_redshift']:
            return
                    
        switch = False
                
        # Check to make sure optical depth tables are still valid. 
        # If not, re-initialize UniformBackground instance(s) and
        # prepare to compute the IGM optical depth on-the-fly.
        if self.pre_EoR:
            
            # Keep using optical depth table? If not, proceed to indented block
            if (xtmp[0] > self.pf['EoR_xavg']) or (z < self.cxrb_zall[0]):
                
                self.pb.finish()
                
                if rank == 0 and self.pf['verbose']:
                    if z <= self.zmin_igm:
                        print "\nEoR has begun (@ z=%.4g, x=%.4g) because we've reached the end of the optical depth table." \
                            % (z, xtmp[0])
                    else:
                        print "\nEoR has begun (@ z=%.4g, x=%.4g) by way of xavg > %.4g criterion." \
                            % (z, xtmp[0], self.pf['EoR_xavg'])
                        
                self.pre_EoR = False
                
                # Update parameter file
                Nz = int((np.log10(1.+self.cxrb_zall[self.cxrb_llo]) \
                        - np.log10(1.+self.pf['final_redshift'])) \
                        / self.pf['EoR_dlogx']) + 1
                        
                new_pars = {'initial_redshift': self.cxrb_zall[self.cxrb_llo], 
                    'redshift_bins': Nz, 'load_tau': False}
                
                # Loop over sources and re-initialize
                ct = 0
                self.rbs_old = []
                
                self.cxrb_fhi_EoR = []; self.cxrb_flo_EoR = []
                for i, rb in enumerate(self.rbs):
                    if self.cxrb_gen[i] is None:
                        self.cxrb_fhi_EoR.append(0.0)
                        self.cxrb_flo_EoR.append(0.0)
                        self.rbs_old.append(None)
                        continue
                    
                    # Store last two "pre-EoR" fluxes
                    E_pre, flux_lo = rb.igm.E.copy(), self.cxrb_flo[i].copy()
                    flux_hi = self.cxrb_fhi[i].copy()
                    zlo, zhi = self.cxrb_zlo, self.cxrb_zhi
                    
                    pop = self.pops.pops[i]
                    pop.pf.update(new_pars)
                    
                    self.rbs_old.append(copy.deepcopy(rb))
                    rb.__init__(pop=pop, use_tab=False)

                    fhi_interp = np.interp(rb.igm.E, E_pre, flux_hi)
                    flo_interp = np.interp(rb.igm.E, E_pre, flux_lo)

                    self.cxrb_lhi = rb.igm.L - 1
                    self.cxrb_zhi = rb.igm.z[-1]

                    self.cxrb_llo = rb.igm.L - 2
                    self.cxrb_zlo = rb.igm.z[-2]

                    self.cxrb_fhi_EoR.append(fhi_interp)

                    # Interpolate to current redshift
                    tmp = np.zeros_like(rb.igm.E)
                    for j, nrg in enumerate(rb.igm.E):
                        tmp[j] = np.interp(self.cxrb_zlo, [zlo, zhi], 
                            [flo_interp[j], fhi_interp[j]])

                    self.cxrb_flo_EoR.append(tmp)

                    # Optical depth "on-the-fly"
                    if ct == 0:
                        self.tau_otf_all = np.zeros([rb.igm.L, rb.igm.N])
                        self.tau_otf = np.zeros(rb.igm.N)
                    
                    self.cxrb_gen[i] = rb.XrayFluxGenerator(self.tau_otf,
                        flux0=fhi_interp)

                    # Poke generator once 
                    # (the first redshift is guaranteed to overlap with p-EoR)
                    self.cxrb_gen[i].next()
                    
                    ct += 1
                                        
                    # Poke generator again if redshift resolution very good?
                    
                del self.cxrb_flo, self.cxrb_fhi
                self.cxrb_fhi = self.cxrb_fhi_EoR
                self.cxrb_flo = self.cxrb_flo_EoR

                self._init_XRB(pre_EoR=False, **kwargs)
            
                self.pb = ProgressBar(self.tf, '21-cm (EoR)',
                    use=self.pf['progress_bar'])
                self.pb.start()

        # Loop over UniformBackground instances, sum fluxes
        ct = 0
        new_fluxes = []
        for i, rb in enumerate(self.rbs):
            if self.cxrb_gen[i] is None:
                new_fluxes.append(0.0)
                continue

            # If we don't have fluxes for this redshift yet, poke the
            # generator to get the next set
            if z < self.cxrb_zlo:
                switch = True
                
                # If in EoR, update optical depth reference
                if not self.pre_EoR and ct == 0:

                    # Optical depth between zlo and zlo-dz
                    this_tau = self.ComputeTauOTF(rb)
                    self.tau_otf_all[self.cxrb_llo] = this_tau.copy()
                    self.tau_otf[:] = this_tau
                    
                    if np.any(np.isinf(this_tau)):
                        raise ValueError('infinite optical depth')

                # Now, compute flux
                new_fluxes.append(self.cxrb_gen[i].next())
                
            ct += 1

        # If we just retrieved new fluxes, update attributes accordingly
        if switch:
            znow = self.cxrb_zall[self.cxrb_llo]
            znext = self.cxrb_zall[self.cxrb_llo - 1]
            
            self.cxrb_lhi -= 1
            self.cxrb_llo -= 1

            self.cxrb_zhi = self.cxrb_zlo
            self.cxrb_zlo = znext
            
            self.cxrb_fhi = copy.deepcopy(self.cxrb_flo)
            self.cxrb_flo = copy.deepcopy(new_fluxes)
            
            # Heat and ionization
            self.cxrb_hhi = self.cxrb_hlo
            self.cxrb_G1hi = self.cxrb_G1lo
            self.cxrb_G2hi = self.cxrb_G2lo
                        
            self.cxrb_hlo = np.zeros([self.Nrbs, self.grid.N_absorbers])
            self.cxrb_G1lo = np.zeros([self.Nrbs, self.grid.N_absorbers])
            self.cxrb_G2lo = np.zeros([self.Nrbs, self.grid.N_absorbers, \
                self.grid.N_absorbers])
                            
            for i, rb in enumerate(self.rbs):
                
                for j, absorber in enumerate(self.grid.absorbers):
                    
                    if j > 0 and self.pf['approx_He']:
                        continue
                                        
                    if rb.pop.pf['approx_xrb']:
                        
                        heat_rb = rb.igm.HeatingRate(self.cxrb_zlo, 
                            species=j, return_rc=True, **kwargs)
                        G1_rb = rb.igm.IonizationRateIGM(self.cxrb_zlo, 
                            species=j, return_rc=True, **kwargs)
                        
                        self.cxrb_hlo[i,j] = heat_rb
                        self.cxrb_G1lo[i,j] = G1_rb
                        
                        self.xray_flux[i].append(0.0)
                        self.xray_heat[i].append(self.cxrb_hlo[i])
                        
                        if self.pf['secondary_ionization'] == 0:
                            self.cxrb_G2lo[i,j] = np.zeros(self.grid.N_absorbers)
                            continue
                        
                        G2_rb = rb.igm.SecondaryIonizationRateIGM(self.cxrb_zlo, 
                            species=j, return_rc=True, **kwargs)

                        self.cxrb_G2lo[i,:] = G2_rb

                    self.xray_flux[i].append(self.cxrb_flo[i])
                    
                    self.cxrb_hlo[i,j] = rb.igm.HeatingRate(self.cxrb_zlo, 
                        species=j, xray_flux=self.cxrb_flo[i], return_rc=True, 
                        **kwargs)
                    
                    self.xray_heat[i].append(self.cxrb_hlo[i])    
                    
                    self.cxrb_G1lo[i,j] = rb.igm.IonizationRateIGM(self.cxrb_zlo, 
                        species=j, xray_flux=self.cxrb_flo[i], return_rc=True,
                        **kwargs)
                    
                    if self.pf['secondary_ionization'] > 0:
                        
                        for k, donor in enumerate(self.grid.absorbers):
                            self.cxrb_G2lo[i,j,k] = \
                                rb.igm.SecondaryIonizationRateIGM(self.cxrb_zlo, 
                                species=j, donor=k, xray_flux=self.cxrb_flo[i], 
                                return_rc=True, **kwargs)
                                    
    def ComputeTauOTF(self, rb):
        """
        Compute IGM optical depth on-the-fly (OTF).
        
        Must extrapolate to determine IGM ionization state at next z.
        """

        if self.cxrb_llo <= 1:
            znow = rb.igm.z[1]
            znext = rb.igm.z[0]
        else:
            znow = rb.igm.z[self.cxrb_llo]
            znext = rb.igm.z[self.cxrb_llo-1]

        # Redshift in order of increasing redshift
        zz = np.array(self.history['z'][-1:-4:-1])
        
        # Compute mean hydrogen ionized fraction
        xHI_igm = np.array(self.history['igm_h_1'])[-1:-4:-1]
        xHI_cgm = np.array(self.history['cgm_h_1'])[-1:-4:-1]
        
        xHI_avg = xHI_cgm * xHI_igm
                
        xx = [xHI_avg]
        nf = [rb.igm.cosm.nH]
        
        # Account for helium opacity
        if self.pf['include_He']:
            
            if self.pf['approx_He']:
                xx.extend([xHI_avg, np.zeros_like(xHI_avg)])
                nf.extend([rb.igm.cosm.nHe]*2)
                
            else:    
                xHeI_igm = np.array(self.history['igm_he_1'][-1:-4:-1])
                xHeI_cgm = np.array(self.history['cgm_h_1'][-1:-4:-1])
                xHeI_avg = xHeI_cgm * xHeI_igm
                                                                     
                xHeII_igm = np.array(self.history['igm_he_2'][-1:-4:-1])
                xHeII_cgm = np.array(self.history['cgm_h_2'][-1:-4:-1])
                xHeII_avg = xHeII_cgm * xHeII_igm
                
                xx.extend([xHeI_avg, xHeII_avg])
                nf.extend([rb.igm.cosm.nHe]*2)
        #else:
        #    xx.extend([xHI_avg, np.zeros_like(xHI_avg)])
        #    nf.extend([rb.igm.cosm.nHe]*2)

        tau = np.zeros_like(rb.igm.E)
        for k in range(3):  # absorbers

            if k > 0 and (not self.pf['include_He']):
                continue
                
            # If approximating He opacity, we're neglecting HeII 
            if self.pf['approx_He'] and k == 2:
                continue

            # Interpolate to get current neutral fraction
            xnow = np.interp(znow, zz, xx[k])
            
            # Extrapolate to find neutral fractions at these two redshifts
            m = (xx[k][1] - xx[k][0]) / (zz[1] - zz[0])
            xnext = m * (znext - zz[0]) + xx[k][0]
            
            # Get cross-sections
            snext = rb.igm.sigma_E[k]
            snow = np.roll(rb.igm.sigma_E[k], -1)
            
            # Neutral densities
            nnow = nf[k](znow) * xnow
            nnext = nf[k](znext) * xnext
                                    
            tau += 0.5 * (znow - znext) \
                * (nnext * rb.igm.cosm.dldz(znext) * snext \
                +  nnow  * rb.igm.cosm.dldz(znow)  * snow)

        tau[-1] = 0.0

        return tau
                      
    def save(self, prefix, suffix='pkl', clobber=False):
        """
        Save results of calculation to file.
        
        Parameters
        ----------
        prefix : str
            Prefix of file.
        suffix : str
            Suffix of file, valid options include 'hdf5', 'pkl', and 'npz'.
        clobber : bool
            Overwrite file with same name if one exists?
            
        """
        
        self.write.save(prefix, suffix, clobber)
        

