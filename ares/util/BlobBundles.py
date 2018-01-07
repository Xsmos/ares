"""

ares/util/BlobBundles.py

Author: Jordan Mirocha
Affiliation: UCLA
Created on: Tue Aug  9 14:19:32 PDT 2016

Description: 

"""

import numpy as np
from ..physics.Constants import nu_0_mhz
from .ParameterBundles import ParameterBundle

_gs_hist = ['z', 'cgm_h_2', 'igm_h_2', 'igm_Tk', 'Ja', 'Jlw', 'Ts', 'dTb']
_gs_ext = []
for tp in ['A', 'B', 'C', 'D', 'Bp', 'Cp', 'Dp']:
    for field in _gs_hist:
        _gs_ext.append('{0!s}_{1!s}'.format(field, tp))

_gs_min1d = ['z', 'dTb']

# Add the zero-crossing even though its not an extremum
_gs_ext.append('z_ZC')
# CMB optical depth
_gs_ext.append('tau_e')
# Decoupling redshift
_gs_ext.append('z_dec')
_gs_ext.append('Tk_dec')

_def_z = ('z', np.arange(5, 51, 0.1))
_z_from_freq = ('z', nu_0_mhz / np.arange(25., 210, 1.)[-1::-1] - 1.)

_gs_shape_n = ['hwtqm_diff_C', 'hwhm_diff_C', 'hwqm_diff_C', 
    'fwtqm_C', 'fwhm_C', 'fwqm_C']
_gs_shape_n.extend(['hwtqm_diff_D', 'hwhm_diff_D', 'hwqm_diff_D', 
    'fwtqm_D', 'fwhm_D', 'fwqm_D'])

_gs_shape_f = \
[
 'Width(max_fraction=0.75, peak_relative=True)', 
 'Width(max_fraction=0.5, peak_relative=True)', 
 'Width(max_fraction=0.25, peak_relative=True)',
 'Width(max_fraction=0.75, peak_relative=False)',
 'Width(max_fraction=0.5, peak_relative=False)',
 'Width(max_fraction=0.25, peak_relative=False)',
 'Width(absorption=False, max_fraction=0.75, peak_relative=True)',
 'Width(absorption=False, max_fraction=0.5, peak_relative=True)', 
 'Width(absorption=False, max_fraction=0.25, peak_relative=True)',
 'Width(absorption=False, max_fraction=0.75, peak_relative=False)',
 'Width(absorption=False, max_fraction=0.5, peak_relative=False)',
 'Width(absorption=False, max_fraction=0.25, peak_relative=False)'
]

# Add curvature of turning points too
_gs_shape_n.extend(['curvature_{!s}'.format(tp) for tp in list('BCD')])
_gs_shape_f.extend([None] * 3)
_gs_shape_n.extend(['skewness_{!s}'.format(region) for region in \
    ['absorption', 'emission']])
_gs_shape_n.extend(['kurtosis_{!s}'.format(region) for region in \
    ['absorption', 'emission']])
_gs_shape_f.extend([None] * 4)

# Rate coefficients
_rc_base = ['igm_k_ion', 'igm_k_heat', 'cgm_k_ion']
_species = ['h_1', 'he_1', 'he_2']
_gs_rates = []
_rc_funcs = []
for _name in _rc_base:
    
    for i, spec1 in enumerate(_species):
    
        _save_name = '{0!s}_{1!s}'.format(_name, spec1)
        _gs_rates.append(_save_name)        
        _rc_funcs.append((_name,i))

        # Don't do secondary ionization terms yet
        #for j, spec2 in enumerate(_species):
                    
_extrema = {'blob_names':_gs_ext, 'blob_ivars': None,  'blob_funcs': None}
_rates = {'blob_names':_gs_rates, 'blob_ivars': _def_z,  'blob_funcs': _rc_funcs}
_history = {'blob_names':_gs_hist,'blob_ivars': _def_z,'blob_funcs': None}
_shape = {'blob_names':_gs_shape_n,'blob_ivars': None, 'blob_funcs': _gs_shape_f}
_runtime = {'blob_names': ['count', 'timer', 'rank'], 
    'blob_ivars': None, 'blob_funcs': None}

_He = {'blob_names':['igm_he_1', 'igm_he_2', 'igm_he_3'], 
       'blob_ivars': _def_z,  
       'blob_funcs': None}

# Not a great default way of doing this, since we may have multiple populations, etc.
_sfrd = {'blob_names': ['sfrd{0}'],
         'blob_ivars': _def_z,
         'blob_funcs': ['pops[0].SFRD']}


_emiss = {'blob_names': ['rho_LW{0}', 'rho_LyC{0}', 'rho_sXR{0}', 'rho_hXR{0}'],
          'blob_ivars': _def_z,
          'blob_funcs': ['pops[0]._LuminosityDensity_LW',
                         'pops[0]._LuminosityDensity_LyC',
                         'pops[0]._LuminosityDensity_sXR',
                         'pops[0]._LuminosityDensity_hXR']}

_cxrb = {'blob_names': ['jsxb', 'jhxb'], 
         'blob_ivars': None,
         'blob_funcs': ['medium.field.jxrb(\'soft\')', 'medium.field.jxrb(\'hard\')']}



_blob_n1 = ['galaxy_lf']
_blob_n2 = ['fstar']
_blob_i1 = [('z', np.array([3.8, 4.9, 5.9, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12, 15])),
    ('x', np.arange(-24, -8.8, 0.2))]
_blob_i2 = [('z', np.array([3.8, 4.9, 5.9, 6.9, 7, 7.9, 8, 9, 10, 10.4, 11, 12, 15])),
    ('Mh', 10**np.arange(5., 13.6, 0.1))]
_blob_f1 = ['pops[0].LuminosityFunction']
_blob_f2 = ['pops[0].fstar']

_lf = \
{ 
 'blob_names': [_blob_n1, _blob_n2],
 'blob_ivars': [_blob_i1, _blob_i2],
 'blob_funcs': [_blob_f1, _blob_f2],
}

_blobs = \
{
 'gs': {'basics': _extrema, 'history': _history, 'shape': _shape,
        'runtime': _runtime, 'rates': _rates, 'helium': _He},
 'pop': {'sfrd': _sfrd, 'emissivities': _emiss, 'fluxes': None, 
    'cxrb': _cxrb, 'lf': _lf}
}

_keys = ('blob_names', 'blob_ivars', 'blob_funcs')

class BlobBundle(ParameterBundle):
    def __init__(self, bundle=None, **kwargs):
        ParameterBundle.__init__(self, bundle=bundle, bset=_blobs, **kwargs)

        self._check_shape()

    def _check_shape(self):
        # For a single blob bundle, make sure elements are lists
        for key in _keys:
            if type(self[key]) is not list:
                self[key] = [self[key]]
              
        for key in _keys: 
            if self[key][0] is None:
                continue
                
            if type(self[key][0]) is not list:
                self[key] = [self[key]]

    @property
    def Nb_groups(self):
        if not hasattr(self, '_Nb_groups'):
            ct = 0
            for element in self['blob_names']:
                ct += 1
                    
            self._Nb_groups = max(ct, 1)
            
        return self._Nb_groups            

    def __add__(self, other):
        """
        This ain't pretty, but it does the job.
        """        
                
        # Number of blob groups
        if hasattr(other, 'Nb_groups'):
            Nb_new = other.Nb_groups
        else:
            Nb_new = 1

        Nb_next = self.Nb_groups + Nb_new

        # Don't operate on self (or copy) since some elements might be None
        # which will be a problem for append
        out = {key: [None for i in range(Nb_next)] for key in _keys}
                
        # Need to add another level of nesting on the first go 'round
        for key in _keys:
            for j in range(self.Nb_groups):
                if self[key][j] is None:
                    continue
                                    
                out[key][j] = self[key][j]
        
        for key in _keys:
            for i, j in enumerate(range(self.Nb_groups, Nb_next)):
                if other[key] is None: 
                    continue
                    
                out[key][j] = other[key][i]
        
        return BlobBundle(**out)
        
        
