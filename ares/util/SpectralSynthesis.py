"""

SpectralSynthesis.py

Author: Jordan Mirocha
Affiliation: McGill
Created on: Sat 25 May 2019 09:58:14 EDT

Description: 

"""

import time
import collections
import numpy as np
from ..util import Survey
from ..util import ProgressBar
from ..util import ParameterFile
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from ..physics.Cosmology import Cosmology
from ..physics.Constants import s_per_myr, c, h_p, erg_per_ev

flux_AB = 3631. * 1e-23 # 3631 * 1e-23 erg / s / cm**2 / Hz
nanoJ = 1e-23 * 1e-9

all_cameras = ['wfc', 'wfc3', 'nircam']

def what_filters(z, fset, wave_lo=1300., wave_hi=2600.):
    """
    Given a redshift and a full filter set, return the filters that probe
    the rest UV continuum only.
    """
    
    # Compute observed wavelengths in microns
    l1 = wave_lo * (1. + z) * 1e-4
    l2 = wave_hi * (1. + z) * 1e-4
    
    out = []
    for filt in fset.keys():
        # Hack out numbers
        _x, _y, mid, dx, Tbar = fset[filt]
        
        fhi = mid + dx[0]
        flo = mid - dx[1]
                
        if not ((flo >= l1) and (fhi <= l2)):
            continue
        
        out.append(filt)
        
        
    return out

class SpectralSynthesis(object):
    def __init__(self, **kwargs):
        self.pf = ParameterFile(**kwargs)

    @property
    def cosm(self):
        if not hasattr(self, '_cosm'):
            self._cosm = Cosmology(pf=self.pf, **self.pf)
        return self._cosm

    @property
    def src(self):
        return self._src

    @src.setter
    def src(self, value):
        self._src = value
        
    @property
    def oversampling_enabled(self):
        if not hasattr(self, '_oversampling_enabled'):
            self._oversampling_enabled = True
        return self._oversampling_enabled
        
    @oversampling_enabled.setter
    def oversampling_enabled(self, value):
        self._oversampling_enabled = value

    @property
    def oversampling_below(self):
        if not hasattr(self, '_oversampling_below'):
            self._oversampling_below = 30.
        return self._oversampling_below
        
    @oversampling_below.setter
    def oversampling_below(self, value):
        self._oversampling_below = value
    
    @property
    def force_perfect(self):
        if not hasattr(self, '_force_perfect'):
            self._force_perfect = False
        return self._force_perfect
    
    @force_perfect.setter
    def force_perfect(self, value):
        self._force_perfect = value
        
    @property
    def careful_cache(self):
        if not hasattr(self, '_careful_cache_'):
            self._careful_cache_ = True
        return self._careful_cache_
        
    @careful_cache.setter
    def careful_cache(self, value):
        self._careful_cache_ = value
    
    @property
    def cameras(self):
        if not hasattr(self, '_cameras'):
            self._cameras = {}
            for cam in all_cameras:
                self._cameras[cam] = Survey(cam=cam, 
                    force_perfect=self.force_perfect)

        return self._cameras

    def Slope(self, zobs, spec=None, waves=None, sfh=None, zarr=None, tarr=None,
        tobs=None, cam=None, rest_wave=(1600., 2300.), band=None, hist={},
        return_norm=False, filters=None, filter_set=None, dlam=10., idnum=None,
        method='fit', window=1, extras={}, picky=False):
        """
        Compute slope in some wavelength range or using photometry.
        """
        
        # If no camera supplied, operate directly on spectrum
        if cam is None:

            func = lambda xx, p0, p1: p0 * (xx / 1.)**p1
            
            if waves is None:
                waves = np.arange(rest_wave[0], rest_wave[1]+dlam, dlam)

            owaves, oflux = self.ObserveSpectrum(zobs, spec=spec, waves=waves,
                sfh=sfh, zarr=zarr, tarr=tarr, flux_units='Ang', hist=hist,
                extras=extras, idnum=idnum, window=window)

            rwaves = waves
            ok = np.logical_and(rwaves >= rest_wave[0], rwaves <= rest_wave[1])

            x = owaves[ok==1]
            
            if oflux.ndim == 2:
                batch_mode = True
                y = oflux[:,ok==1].swapaxes(0, 1)
                
                ma = np.max(y, axis=0)
                sl = -2.5 * np.ones(ma.size)
                guess = np.vstack((ma, sl)).T                
            else:    
                batch_mode = False
                y = oflux[ok==1]
                guess = np.array([oflux[np.argmin(np.abs(owaves - 1.))], -2.4])
                        
        else:
            
            if filters is not None:
                assert rest_wave is None, \
                    "Set rest_wave=None if filters are supplied"
            
            # Log-linear fit
            func = lambda x, p0, p1: p0 * (x / 1.)**p1
            
            if type(cam) not in [list, tuple]:
                cam = [cam]
            
            filt = []
            xphot = []
            dxphot = []
            ycorr = []
            for _cam in cam:
                _filters, _xphot, _dxphot, _ycorr = \
                    self.Photometry(sfh=sfh, hist=hist, idnum=idnum, spec=spec,
                    cam=_cam, filters=filters, filter_set=filter_set, waves=waves,
                    dlam=dlam, tarr=tarr, tobs=tobs, extras=extras, picky=picky,
                    zarr=zarr, zobs=zobs, rest_wave=rest_wave, window=window)
            
                filt.extend(list(_filters))
                xphot.extend(list(_xphot))
                dxphot.extend(list(_dxphot))
                ycorr.extend(list(_ycorr))
            
            # No matching filters? Return.    
            if len(filt) == 0:
                if idnum is not None:
                    N = 1
                elif sfh is not None:
                    N = sfh.shape[0]
                else:
                    N = 1    
                    
                if return_norm:
                    return -99999 * np.ones((N, 2))
                else:
                    return -99999 * np.ones(N)
                
            filt = np.array(filt)  
            xphot = np.array(xphot)   
            dxphot = np.array(dxphot)
            ycorr = np.array(ycorr)
                        
            # Sort arrays in ascending wavelength
            isort = np.argsort(xphot)
            
            _x = xphot[isort]
            _y = ycorr[isort]
            
            # Recover flux to do power-law fit    
            xp, xm = dxphot.T
            dx = xp + xm
            
            # Need flux in units of A^-1
           #dnphot = c / ((xphot-xm) * 1e-4) - c / ((xphot + xp) * 1e-4)
           #dwdn = dx * 1e4 / dnphot
            _dwdn = (_x * 1e4)**2 / (c * 1e8)
           
            if rest_wave is not None:
                r = _x * 1e4 / (1. + zobs)
                ok = np.logical_and(r >= rest_wave[0], r <= rest_wave[1])
                x = _x[ok==1]
            else:
                ok = np.ones_like(_x)
                x = _x

            # Be careful in batch mode!
            if ycorr.ndim == 2:
                batch_mode = True
                _f = 10**(_y / -2.5) * flux_AB / _dwdn[:,None]
                y = _f[ok==1]
                ma = np.max(y, axis=0)
                sl = -2.5 * np.ones(ma.size)
                guess = np.vstack((ma, sl)).T
            else:
                batch_mode = False
                _f = 10**(_y / -2.5) * flux_AB / _dwdn
                y = _f[ok==1]   
                ma = np.max(y) 
                guess = np.array([ma, -2.5])                
            
            if ok.sum() == 2 and self.pf['verbose']:
                print("WARNING: Estimating slope from only two points: {}".format(filt[isort][ok==1]))

        ##
        # Fit a PL to points.
        if method == 'fit':
            
            if len(x) < 2:
                if self.pf['verbose']:
                    print("Not enough points to estimate slope")
                
                if batch_mode:
                    corr = np.ones(y.shape[1])
                else:
                    corr = 1
                
                if return_norm:
                    return -99999 * corr, -99999 * corr
                else:
                    return -99999 * corr
            
            if batch_mode:
                N = y.shape[1]
                popt = -99999 * np.ones((2, N))
                pcov = -99999 * np.ones((2, 2, N))
                            
                for i in range(N):
                    
                    if not np.any(y[:,i] > 0):
                        continue
                    
                    try:
                        popt[:,i], pcov[:,:,i] = curve_fit(func, x, y[:,i], 
                            p0=guess[i], maxfev=1000)
                    except RuntimeError:
                        popt[:,i], pcov[:,:,i] = -99999, -99999
                                
            else:
                try:
                    popt, pcov = curve_fit(func, x, y, p0=guess)
                except RuntimeError:
                    popt, pcov = -99999 * np.ones(2), -99999 * np.ones(2)
                        
        elif method == 'diff':
            
            assert cam is None, "Should only use to skip photometry."
                        
            # Remember that galaxy number is second dimension
            logL = np.log(y)
            logw = np.log(x)
            
            if batch_mode:
                # Logarithmic derivative = beta
                beta = (logL[-1,:] - logL[0,:]) / (logw[-1,None] - logw[0,None])
            else:
                beta = (logL[-1] - logL[0]) / (logw[-1] - logw[0])
            
            popt = np.array([-99999, beta])
            
        else:
            raise NotImplemented('help me')
                
        if return_norm:
            return popt
        else:
            return popt[1]
        
    def ObserveSpectrum(self, zobs, spec=None, sfh=None, waves=None,
        flux_units='Hz', tarr=None, tobs=None, zarr=None, hist={}, 
        idnum=None, window=1, extras={}):
        """
        Take an input spectrum and "observe" it at redshift z.
        
        Parameters
        ----------
        wave : np.ndarray
            Rest wavelengths in [Angstrom]
        spec : np.ndarray
            Specific luminosities in [erg/s/A]
        z : int, float
            Redshift.
        
        Returns
        -------
        Observed wavelengths in microns, observed fluxes in erg/s/cm^2/Hz.
        
        """
        
        if spec is None:
            spec = self.Spectrum(sfh, waves, tarr=tarr, zarr=zarr, 
                zobs=zobs, tobs=None, hist=hist, idnum=idnum,
                extras=extras, window=window)
    
        dL = self.cosm.LuminosityDistance(zobs)
        
        if waves is None:
            waves = self.src.wavelengths
            dwdn = self.src.dwdn
            assert len(spec) == len(waves)
        else:
            #freqs = c / (waves / 1e8)
            dwdn = waves**2 / (c * 1e8)
            #tmp = np.abs(np.diff(waves) / np.diff(freqs))
            #dwdn = np.concatenate((tmp, [tmp[-1]]))
    
        # Flux at Earth in erg/s/cm^2/Hz
        f = spec / (4. * np.pi * dL**2)
        
        if flux_units == 'Hz':
            pass
        else:
            f /= dwdn

        return waves * (1. + zobs) / 1e4, f
        
    def _select_filters(self, zobs, cam='wfc3', lmin=912., lmax=3000., tol=0.0, 
        filter_set=None, filters='all'):
        # Get transmission curves
        filter_data = self.cameras[cam]._read_throughputs(filter_set=None, 
            filters='all')
        all_filters = filter_data.keys()    
        
        ok_cent = []
        ok_filters = []
        for filt in filter_data:
            
            # Don't use narrow filters! Maybe come back to this.
            if filt.endswith('N'):
                continue
                
            x, y, cent, dx, Tavg, norm = filter_data[filt]
        
            lo = (cent - dx[1]) * 1e4 / (1. + zobs)
            hi = (cent + dx[0]) * 1e4 / (1. + zobs)
            
            if lo < (lmin - tol):
                continue
            if hi > (lmax + tol):
                continue    
                
            ok_filters.append(filt)
            ok_cent.append(cent)
            
        isort = np.argsort(ok_cent)    
        
        return tuple(np.array(ok_filters)[isort])
                
    def Photometry(self, spec=None, sfh=None, cam='wfc3', filters='all', 
        filter_set=None, dlam=10., rest_wave=None, extras={}, window=1,
        tarr=None, zarr=None, waves=None, zobs=None, tobs=None, band=None, 
        hist={}, idnum=None, flux_units=None, picky=False, lbuffer=200.):
        """
        Just a wrapper around `Spectrum`.

        Returns
        -------
        Tuple containing (in this order):
            - Names of all filters included
            - Midpoints of photometric filters [microns]
            - Width of filters [microns]
            - Apparent magnitudes corrected for filter transmission.

        """
        
        assert (tobs is not None) or (zobs is not None)
        
        if zobs is None:
            zobs = self.cosm.z_of_t(tobs * s_per_myr)
                    
        # Might be stored for all redshifts so pick out zobs            
        if type(filters) == dict:
            assert zobs is not None
            filters = filters[round(zobs)]
                    
        # Get transmission curves
        if cam in self.cameras.keys():
            filter_data = self.cameras[cam]._read_throughputs(filter_set=filter_set, 
                filters=filters)
        else:
            # Can supply spectral windows, e.g., Calzetti+ 1994, in which case
            # we assume perfect transmission but otherwise just treat like
            # photometric filters.
            assert type(filters) in [list, tuple, np.ndarray]
            
            #print("Generating photometry from {} spectral ranges.".format(len(filters)))
            
            wraw = np.array(filters)
            x1 = wraw.min()
            x2 = wraw.max()
            x = np.arange(x1-100, x2+101, 1.) * 1e-4 * (1. + zobs)
                        
            # Note that in this case, the filter wavelengths are in rest-frame
            # units, so we convert them to observed wavelengths before
            # photometrizing everything.
            
            filter_data = {}    
            for _window in filters:
                lo, hi = _window
                
                lo *= 1e-4 * (1. + zobs)
                hi *= 1e-4 * (1. + zobs)
                
                y = np.zeros_like(x)
                y[np.logical_and(x >= lo, x <= hi)] = 1
                mi = np.mean([lo, hi])
                dx = np.array([hi - mi, mi - lo])
                Tavg = 1.
                filter_data[_window] = x, y, mi, dx, Tavg
                            
        all_filters = filter_data.keys()    
            
        # Figure out spectral range we need to model for these filters.
        # Find bluest and reddest filters, set wavelength range with some
        # padding above and below these limits.
        lmin = np.inf
        lmax = 0.0
        ct = 0
        for filt in filter_data:
            x, y, cent, dx, Tavg = filter_data[filt]
                                    
            # If we're only doing this for the sake of measuring a slope, we
            # might restrict the range based on wavelengths of interest, i.e.,
            # we may not use all the filters.
            
            # Right now, will include filters as long as their center is in
            # the requested band. This results in fluctuations in slope 
            # measurements, so to be more stringent set picky=True.
            if rest_wave is not None:
                
                if picky:
                    l = (cent - dx[1]) * 1e4 / (1. + zobs)
                    r = (cent + dx[0]) * 1e4 / (1. + zobs)
                    
                    if (l < rest_wave[0]) or (r > rest_wave[1]):
                        continue

                cent_r = cent * 1e4 / (1. + zobs)
                if (cent_r < rest_wave[0]) or (cent_r > rest_wave[1]):
                    continue

            lmin = min(lmin, cent - dx[1] * 1.2)
            lmax = max(lmax, cent + dx[0] * 1.2)
            ct += 1
            
        # No filters in range requested    
        if ct == 0:    
            return [], [], [], []
                                                
        # Here's our array of REST wavelengths
        if waves is None:
            # Convert from microns to Angstroms, undo redshift.
            lmin = lmin * 1e4 / (1. + zobs)
            lmax = lmax * 1e4 / (1. + zobs)
                
            lmin = max(lmin, self.src.wavelengths.min())
            lmax = min(lmax, self.src.wavelengths.max())
            
            # Force edges to be multiples of dlam
            l1 = lmin - lbuffer
            l1 -= l1 % dlam
            l2 = lmax + lbuffer
            
            waves = np.arange(l1, l2+dlam, dlam)
                
        # Get spectrum first.
        if spec is None:
            spec = self.Spectrum(sfh, waves, tarr=tarr, tobs=tobs,
                zarr=zarr, zobs=zobs, band=band, hist=hist,
                idnum=idnum, extras=extras, window=window)
            
        # Might be running over lots of galaxies
        batch_mode = False 
        if spec.ndim == 2:
            batch_mode = True    

        # Observed wavelengths in micron, flux in erg/s/cm^2/Hz
        wave_obs, flux_obs = self.ObserveSpectrum(zobs, spec=spec, 
            waves=waves, extras=extras, window=window)
            
        # Convert microns to cm. micron * (m / 1e6) * (1e2 cm / m)
        freq_obs = c / (wave_obs * 1e-4)
                    
        # Why do NaNs happen? Just nircam. 
        flux_obs[np.isnan(flux_obs)] = 0.0
        
        # Loop over filters and re-weight spectrum
        xphot = []      # Filter centroids
        wphot = []      # Filter width
        yphot_corr = [] # Magnitudes corrected for filter transmissions.
    
        # Loop over filters, compute fluxes in band (accounting for 
        # transmission fraction) and convert to observed magnitudes.
        for filt in all_filters:

            x, T, cent, dx, Tavg = filter_data[filt]
            
            if rest_wave is not None:
                cent_r = cent * 1e4 / (1. + zobs)
                if (cent_r < rest_wave[0]) or (cent_r > rest_wave[1]):
                    continue
                        
            # Re-grid transmission onto provided wavelength axis.
            T_regrid = np.interp(wave_obs, x, T, left=0, right=0)
            #func = interp1d(x, T, kind='cubic', fill_value=0.0,
            #    bounds_error=False)
            #T_regrid = func(wave_obs)
            
            #T_regrid = np.interp(np.log(wave_obs), np.log(x), T, left=0., 
            #    right=0)
                 
            # Remember: observed flux is in erg/s/cm^2/Hz

            # Integrate over frequency to get integrated flux in band
            # defined by filter.
            if batch_mode:
                integrand = -1. * flux_obs * T_regrid[None,:]                
                _yphot = np.sum(integrand[:,0:-1] * np.diff(freq_obs)[None,:],
                    axis=1)
            else:    
                integrand = -1. * flux_obs * T_regrid
                _yphot = np.sum(integrand[0:-1] * np.diff(freq_obs))
                
                #_yphot = np.trapz(integrand, x=freq_obs)
            
            corr = np.sum(T_regrid[0:-1] * -1. * np.diff(freq_obs), axis=-1)
                                                                               
            xphot.append(cent)
            yphot_corr.append(_yphot / corr)
            wphot.append(dx)
        
        xphot = np.array(xphot)
        wphot = np.array(wphot)
        yphot_corr = np.array(yphot_corr)
        
        # Convert to magnitudes and return
        return all_filters, xphot, wphot, -2.5 * np.log10(yphot_corr / flux_AB)
        
    def Spectrum(self, sfh, waves, tarr=None, zarr=None, window=1,
        zobs=None, tobs=None, band=None, idnum=None, units='Hz', hist={},
        extras={}):
        """
        This is just a wrapper around `Luminosity`.
        """
        
        if sfh.ndim == 2 and idnum is not None:
            sfh = sfh[idnum,:]
        
        batch_mode = sfh.ndim == 2
        time_series = (zobs is None) and (tobs is None)
        
        # Shape of output array depends on some input parameters
        shape = []
        if batch_mode:
            shape.append(sfh.shape[0])
        if time_series:
            shape.append(tarr.size)
        shape.append(len(waves))
            
        spec = np.zeros(shape)
        for i, wave in enumerate(waves):
            slc = (Ellipsis, i) if (batch_mode or time_series) else i
            
            spec[slc] = self.Luminosity(sfh, wave=wave, tarr=tarr, zarr=zarr,
                zobs=zobs, tobs=tobs, band=band, hist=hist, idnum=idnum,
                extras=extras, window=window)
                
        if units in ['A', 'Ang']:
            #freqs = c / (waves / 1e8)
            #tmp = np.abs(np.diff(waves) / np.diff(freqs))
            #dwdn = np.concatenate((tmp, [tmp[-1]]))
            dwdn = waves**2 / (c * 1e8)
            spec /= dwdn    
        
        return spec
        
    def Magnitude(self, sfh, wave=1600., tarr=None, zarr=None, window=1,
        zobs=None, tobs=None, band=None, idnum=None, hist={}, extras={}):
        
        L = self.Luminosity(sfh, wave=wave, tarr=tarr, zarr=zarr, 
            zobs=zobs, tobs=tobs, band=band, idnum=idnum, hist=hist, 
            extras=extras, window=window)
        
        MAB = self.magsys.L_to_MAB(L, z=zobs)
        
        return MAB    
        
    def _oversample_sfh(self, ages, sfh, i):
        """
        Over-sample time axis while stellar populations are young if the time
        resolution is worse than 1 Myr / grid point.
        """
        
        batch_mode = sfh.ndim == 2
        
        # Use 1 Myr time resolution for final stretch.
        # final stretch is determined by `oversampling_below` attribute.
        # This loop determines how many elements at the end of 
        # `ages` are within the `oversampling_below` zone.
        
        ct = 0
        while ages[-1-ct] < self.oversampling_below:
            ct += 1

            if ct + 1 == len(ages):
                break
        
        ifin = -1 - ct                                                            
        ages_x = np.arange(ages[-1], ages[ifin]+1., 1.)[-1::-1]
                                                        
        # Must augment ages and dt accordingly
        _ages = np.hstack((ages[0:ifin], ages_x))
        _dt = np.abs(np.diff(_ages) * 1e6)
        
        if batch_mode:
            xSFR = np.ones((sfh.shape[0], ages_x.size-1))
        else:
            xSFR = np.ones(ages_x.size-1)
            
            
        #print('hey', ages_x.shape, (ages_x.size - 1) / ct, sfh.shape, xSFR.shape)    
        #    
        #print(ages_x)    
            
        # Must allow non-constant SFR within over-sampled region
        # as it may be tens of Myr.
        # Walk back from the end and fill in SFR
        N = (ages_x.size - 1) / ct
        for _i in range(0, ct):
            
            if batch_mode:
                slc = Ellipsis, slice(-1 * N * _i-1, -1 * N * (_i + 1) -1, -1)
            else:    
                slc = slice(-1 * N * _i-1, -1 * N * (_i + 1) -1, -1)
                                    
            if batch_mode:
                _sfh_rs = np.array([sfh[:,-_i-2]]*N).T
                xSFR[slc] = _sfh_rs * np.ones(N)[None,:]
            else:
                xSFR[slc] = sfh[-_i-2] * np.ones(N)
        
        # Need to tack on the SFH at ages older than our 
        # oversampling approach kicks in.
        if batch_mode:
            if ct + 1 == len(ages):
                print(sfh[:,0].shape, xSFR.shape)
                _SFR = np.hstack((sfh[:,0][:,None], xSFR))
            else:
                _SFR = np.hstack((sfh[:,0:i+1][:,0:ifin+1], xSFR))
        else:
                
            if ct + 1 == len(ages):
                _SFR = np.hstack((sfh[0], xSFR))
            else:
                _SFR = np.hstack((sfh[0:i+1][0:ifin+1], xSFR))
        
        return _ages, _SFR
        
    @property
    def _cache_lum_ctr(self):
        if not hasattr(self, '_cache_lum_ctr_'):
            self._cache_lum_ctr_ = 0
        return self._cache_lum_ctr_    
        
    def _cache_lum(self, kwds):
        """
        Cache object for spectral synthesis of stellar luminosity.
        """
        if not hasattr(self, '_cache_lum_'):
            self._cache_lum_ = {}
                  
        notok = -1
        
        t1 = time.time()                
                        
        # If we set order by hand, it greatly speeds things up because
        # more likely than not, the redshift and wavelength are the only
        # things that change and that's an easy logical check to do.
        # Checking that SFHs, histories, etc., is more expensive.
        ok_keys = ('wave', 'zobs', 'tobs', 'idnum', 'sfh', 'tarr', 'zarr', 
            'window', 'band', 'hist', 'extras', 'load')                
                        
        ct = -1
        # Loop through keys to do more careful comparison for unhashable types.
        for keyset in self._cache_lum_.keys():

            ct += 1

            # Remember: keyset is just a number.
            kw, data = self._cache_lum_[keyset]
            
            # Check wavelength first. Most common thing.
            
            if (self.careful_cache == 0) and ('wave' in kw) and ('zobs' in kw):
                if (kw['wave'] == kwds['wave']) and (kw['zobs'] == kwds['zobs']):
                    notok = 0
                    break
                        
            notok = 0
            # Loop over cached keywords, compare to those supplied.
            for key in ok_keys:

                if key not in kwds:
                    notok += 1
                    break
                                    
                #if isinstance(kw[key], collections.Hashable):
                #    if kwds[key] == kw[key]:
                #        continue
                #    else:
                #        notok += 1
                #        break
                #else:
                # For unhashable types, must work on case-by-case basis.
                if type(kwds[key]) != type(kw[key]):
                    notok += 1 
                    break
                elif type(kwds[key]) == np.ndarray:
                    if np.all(kwds[key] == kw[key]):
                        continue
                    else:
                        print("Does this ever happen?")
                        notok += 1
                        break
                elif type(kwds[key]) == dict:
                    if kwds[key] == kw[key]:
                        continue
                    else:
                        
                        #for _key in kwds[key]:
                        #    print(_key, kwds[key][_key] == kw[key][_key])
                        #
                        #raw_input('<enter>')
                        
                        notok += 1
                        break
                else:
                    if kwds[key] == kw[key]:
                        continue
                    else:
                        notok += 1
                        break
                        
            if notok > 0:
                #print(keyset, key)
                continue
                
            # If we're here, load this thing.
            break        
            
        t2 = time.time()    
            
        if notok < 0:
            return kwds, None          
        elif notok == 0:
            #print("Loaded from cache! Took N={} iterations, {} sec to find match".format(ct, t2 - t1))
            # Recall that this is (kwds, data)
            return self._cache_lum_[keyset]
        else:
            return kwds, None    
        
    def Luminosity(self, sfh, wave=1600., tarr=None, zarr=None, window=1,
        zobs=None, tobs=None, band=None, idnum=None, hist={}, extras={},
        load=True):
        """
        Synthesize luminosity of galaxy with given star formation history at a
        given wavelength and time.
        
        Parameters
        ----------
        sfh : np.ndarray
            Array of SFRs. If 1-D, should be same shape as time or redshift
            array. If 2-D, first dimension should correspond to galaxy number
            and second should be time.
        tarr : np.ndarray
            Array of times in ascending order [Myr].
        zarr : np.ndarray
            Array of redshift in ascending order (so decreasing time). Only
            supply if not passing `tarr` argument.
        wave : int, float
            Wavelength of interest [Angstrom]
        window : int, float
            Average over interval about `wave`? [Angstrom]
        zobs : int, float   
            If supplied, luminosity will be return only for an observation 
            at this redshift.
        tobs : int, float   
            If supplied, luminosity will be return only for an observation 
            at this time.
        hist : dict
            Extra information we may need, e.g., metallicity, dust optical 
            depth, etc. to compute spectrum.
        
        Returns
        -------
        Luminosity at wavelength=`wave` in units of erg/s/Hz.
        
        
        """
        
        kw = {'sfh': sfh, 'zobs':zobs, 'tobs': tobs, 'wave':wave, 'tarr':tarr, 
            'zarr': zarr, 'band': band, 'idnum': idnum, 'hist':hist, 
            'extras': extras}        
        
        #kw_tup = tuple(kw.viewitems())
        
        if load:
            _kwds, cached_result = self._cache_lum(kw)
        else:
            self._cache_lum_ = {}
            cached_result = None
            
        if cached_result is not None:
            return cached_result
        
        if sfh.ndim == 2 and idnum is not None:
            sfh = sfh[idnum,:]
                
        # If SFH is 2-D it means we're doing this for multiple galaxies at once.
        # The first dimension will be number of galaxies and second dimension
        # is time/redshift.
        batch_mode = sfh.ndim == 2
                
        # Parse time/redshift information
        if zarr is not None:
            assert tarr is None
            
            tarr = self.cosm.t_of_z(zarr) / s_per_myr
        else:
            zarr = self.cosm.z_of_t(tarr * s_per_myr)
            
        assert np.all(np.diff(tarr) > 0), \
            "Must supply SFH in time-ascending (i.e., redshift-descending) order!"
        
        # Convert tobs to redshift.        
        if tobs is not None:
            zobs = self.cosm.z_of_t(tobs * s_per_myr)
            assert tarr.min() <= tobs <= tarr.max(), \
                "Requested time of observation (`tobs`) not in supplied range!"
            
        # Prepare slice through time-axis.    
        if zobs is None:
            slc = Ellipsis
            izobs = None
        else:
            # Need to be sure that we grab a grid point exactly at or just
            # below the requested redshift (?)
            izobs = np.argmin(np.abs(zarr - zobs))
            if zarr[izobs] > zobs:
                izobs += 1    
                
            if batch_mode:
                #raise NotImplemented('help')
                # Need to slice over first dimension now...
                slc = Ellipsis, slice(0, izobs+1)
            else:    
                slc = slice(0, izobs+1)
                
            assert zarr.min() <= zobs <= zarr.max(), \
                "Requested time of observation (`tobs`) not in supplied range!"
                                
        fill = np.zeros(1)
        tyr = tarr * 1e6
        dt = np.hstack((np.diff(tyr), fill))
        
        # Figure out if we need to over-sample the grid we've got to more
        # accurately solve for young stellar populations.
        oversample = self.oversampling_enabled and (dt[-2] > 1.01e6)
        # Used to also require zobs is not None. Why?
                        
        ##
        # Done parsing time/redshift
        
        # Is this luminosity in some bandpass or monochromatic?
        if band is not None:
            Loft = self.src.IntegratedEmission(band[0], band[1])
            #raise NotImplemented('help!')
            print("Note this now has different units.")
        else:
            Loft = self.src.L_per_SFR_of_t(wave, avg=window)
        #

        # Setup interpolant for luminosity as a function of SSP age.      
        _func = interp1d(np.log(self.src.times), np.log(Loft),
            kind='cubic', bounds_error=False, 
            fill_value=Loft[-1])
            
        # Extrapolate linearly at times < 1 Myr
        _m = (Loft[1] - Loft[0]) / (self.src.times[1] - self.src.times[0])
        L_small_t = lambda age: _m * age + Loft[0]
        
        #L_small_t = Loft[0]
        
        # Extrapolate as PL at t < 1 Myr based on first two
        # grid points
        #m = np.log(Loft[1] / Loft[0]) \
        #  / np.log(self.src.times[1] / self.src.times[0])
        #func = lambda age: np.exp(m * np.log(age) + np.log(Loft[0]))
                
        #if zobs is None:
        Lhist = np.zeros_like(sfh)
        #else:
        #    pass
            # Lhist will just get made once. Don't need to initialize
        
        ##
        # Loop over the history of object(s) and compute the luminosity of 
        # simple stellar populations of the corresponding ages (relative to
        # zobs).
        ##
        
        # Start from initial redshift and move forward in time, i.e., from
        # high redshift to low.
        for i, _tobs in enumerate(tarr):
                        
            # If zobs is supplied, we only have to do one iteration
            # of this loop. This is just a dumb way to generalize this function
            # to either do one redshift or return a whole history.
            if (zobs is not None):
                if (zarr[i] > zobs):
                    continue

            # If we made it here, it's time to integrate over star formation
            # at previous times. First, retrieve ages of stars formed in all 
            # past star forming episodes.
            ages = tarr[i] - tarr[0:i+1]

            # Treat metallicity evolution? If so, need to grab luminosity as 
            # function of age and Z.
            if self.pf['pop_enrichment']:

                if batch_mode:
                    Z = hist['Z'][slc][:,0:i+1]
                else:
                    Z = hist['Z'][slc][0:i+1]

                logA = np.log10(ages)
                logZ = np.log10(Z)
                L_per_msun = self.L_of_Z_t(wave)(logA, logZ, grid=False)
                                        
                # erg/s/Hz
                if batch_mode:
                    Lall = L_per_msun * sfh[:,0:i+1]
                else:
                    Lall = L_per_msun * sfh[0:i+1]
                    
                if oversample:
                    raise NotImplemented('help!')    
                else:
                    _dt = dt[0:i]
                    
                _ages = ages    
            else:    
                
                # If time resolution is >= 2 Myr, over-sample final interval.
                if oversample and len(ages) > 1:                
                    
                    if batch_mode:
                        _ages, _SFR = self._oversample_sfh(ages, sfh[:,0:i+1], i)
                    else:        
                        _ages, _SFR = self._oversample_sfh(ages, sfh[0:i+1], i)
                        
                    _dt = np.abs(np.diff(_ages) * 1e6)
                    
                    # Now, compute luminosity at expanded ages.
                    L_per_msun = np.exp(_func(np.log(_ages)))    

                    # Interpolate linearly at t < 1 Myr    
                    L_per_msun[_ages < 1] = L_small_t(_ages[_ages < 1])   
                    
                    # erg/s/Hz
                    if batch_mode:
                        Lall = L_per_msun * _SFR
                    else:    
                        Lall = L_per_msun * _SFR
                                                               
                else:    
                    L_per_msun = np.exp(np.interp(np.log(ages), 
                        np.log(self.src.times), np.log(Loft), 
                        left=np.log(Loft[0]), right=np.log(Loft[-1])))
                    
                    _dt = dt[0:i]
                                                
                    # Fix early time behavior
                    L_per_msun[ages < 1] = L_small_t(ages[ages < 1])
        
                    _ages = ages
                            
                    # erg/s/Hz
                    if batch_mode:
                        Lall = L_per_msun * sfh[:,0:i+1]
                    else:    
                        Lall = L_per_msun * sfh[0:i+1]
                                                    
                # Correction for IMF sampling (can't use SPS).
                #if self.pf['pop_sample_imf'] and np.any(bursty):
                #    life = self._stars.tab_life
                #    on = np.array([life > age for age in ages])
                #
                #    il = np.argmin(np.abs(wave - self._stars.wavelengths))
                #
                #    if self._stars.aging:
                #        raise NotImplemented('help')
                #        lum = self._stars.tab_Ls[:,il] * self._stars.dldn[il]
                #    else:
                #        lum = self._stars.tab_Ls[:,il] * self._stars.dldn[il]
                #
                #    # Need luminosity in erg/s/Hz
                #    #print(lum)
                #
                #    # 'imf' is (z or age, mass)
                #
                #    integ = imf[bursty==1,:] * lum[None,:]
                #    Loft = np.sum(integ * on[bursty==1], axis=1)
                #
                #    Lall[bursty==1] = Loft


            # Apply local reddening
            #tau_bc = self.pf['pop_tau_bc']
            #if tau_bc > 0:
            #
            #    corr = np.ones_like(_ages) * np.exp(-tau_bc)
            #    corr[_ages > self.pf['pop_age_bc']] = 1
            #
            #    Lall *= corr

            # Integrate over all times up to this tobs            
            if batch_mode:
                if (zobs is not None):
                    Lhist = np.trapz(Lall, dx=_dt, axis=1)
                else:
                    Lhist[:,i] = np.trapz(Lall, dx=_dt, axis=1)
            else:
                if (zobs is not None):
                    Lhist = np.trapz(Lall, dx=_dt)                
                else:    
                    Lhist[i] = np.trapz(Lall, dx=_dt)

            ##
            # In this case, we only need one iteration of this loop.
            ##
            if zobs is not None:
                break
                                   
        ##
        # Redden spectra
        ##
        if 'Sd' in hist:
            
            # Redden away!        
            if np.any(hist['Sd'] > 0) and (band is None):
                # Reddening is binary and probabilistic
                                
                assert 'kappa' in extras
                
                kappa = extras['kappa'](wave=wave)
                                                                
                kslc = idnum if idnum is not None else Ellipsis                
                                
                if idnum is not None:
                    Sd = hist['Sd'][kslc]
                    if type(hist['fcov']) in [int, float, np.float64]:
                        fcov = hist['fcov']
                    else:
                        fcov = hist['fcov'][kslc]
                        
                    rand = hist['rand'][kslc]
                else:
                    Sd = hist['Sd']
                    fcov = hist['fcov']
                    rand = hist['rand']
                                
                tau = kappa * Sd
                                
                clear = rand > fcov
                block = ~clear
                                
                if idnum is not None:
                    Lout = Lhist * clear[izobs] \
                         + Lhist * np.exp(-tau[izobs]) * block[izobs]
                else:    
                    Lout = Lhist * clear[:,izobs] \
                         + Lhist * np.exp(-tau[:,izobs]) * block[:,izobs]
                                 
            else:
                Lout = Lhist.copy()
        else:
            Lout = Lhist.copy()

        #del Lhist, tau, Lall    
        #gc.collect()
        
        ##
        # Sum luminosity of parent halos along merger tree            
        ##
            
        # Don't change shape, just zero-out luminosities of 
        # parent halos after they merge?
        if hist is not None:            
            do_mergers = self.pf['pop_mergers'] and batch_mode
            
            if 'children' in hist:
                if (hist['children'] is not None) and do_mergers:
                    
                    child_iz, child_iM = children.T
                
                    is_central = child_iM == -1
                                             
                    if np.all(is_central == 1):
                        pass
                    else:    
                    
                        print("Looping over {} halos...".format(sfh.shape[0]))
                                    
                        pb = ProgressBar(sfh.shape[0])
                        pb.start()
                
                        # Loop over all 'branches'
                        for i in range(SFR.shape[0]):
                                            
                            # This means the i'th halo is alive and well at the
                            # final redshift, i.e., it's a central
                            if is_central[i]:
                                continue
                             
                            pb.update(i)
                                                                    
                            # At this point, need to figure out which child halos
                            # to dump mass and SFH into...    
                            
                            # Be careful with redshift array. 
                            # We're now working in ascending time, reverse redshift,
                            # so we need to correct the child iz values. We've also
                            # chopped off elements at z < zobs.
                            #iz = Nz0 - child_iz[i]
                            
                            # This `iz` should not be negative despite us having
                            # chopped up the redshift array since getting to this
                            # point in the loop is predicated on being a parent of
                            # another halo, i.e., not surviving beyond this redshift. 
                            
                            # Lout is just 1-D at this point, i.e., just luminosity
                            # *now*. 
                                            
                            # Add luminosity to child halo. Zero out luminosity of 
                            # parent to avoid double counting. Note that nh will
                            # also have been zeroed out but we're just being careful.
                            Lout[child_iM[i]] += 1 * Lout[i]
                            Lout[i] = 0.0
                        
                        pb.finish()
                             
        
        ##
        # Will be unhashable types so just save to a unique identifier
        ##                          
        self._cache_lum_[self._cache_lum_ctr] = kw, Lout
        self._cache_lum_ctr_ += 1
                                    
        # Get outta here.
        return Lout
        