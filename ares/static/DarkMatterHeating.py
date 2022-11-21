import scipy.constants as Cs
import numpy as np
from scipy.special import erf

reduced_H = 0.74
H0 = 100*reduced_H*3.24077929e-20 # s^-1; 74 km/s/Mpc
Omega_m_0 = 0.26
Omega_b_0 = 0.044# 0.0482 # ordinary (baryonic) matter energy density
Omega_dm_0 = Omega_m_0 - Omega_b_0 # cosmic density parameter for non-relativistic matter
rho_crit_0 = 3*H0**2/(8*np.pi*Cs.G)#0.85*1e-26 # kg/m^3 from https://en.wikipedia.org/wiki/Friedmann_equations#Density_parameter
rho_d_0 = Omega_dm_0 * rho_crit_0
rho_b_0 = Omega_b_0 * rho_crit_0
f_dm = 1
f_He = 0#0.08 # n_He/n_H
m_chi = .1*Cs.giga*Cs.eV / Cs.c**2


def baryon_dark_matter_interaction(redshift, baryon_temperature, dark_matter_temperature, electron_ratio, stream_velocity):
    '''
    input: z, Tb, xe, Tchi, v_stream
    '''
    global z, Tb, xe, Tchi, v_stream
    #print('baryon_dark_matter_interaction working...')

    z = redshift
    Tb = baryon_temperature
    Tchi = dark_matter_temperature
    xe = electron_ratio
    v_stream = stream_velocity
    
    # if v_stream == 0:
    #     print('v_stream is 0!')
    # print(z, Tb, Tchi, xe, v_stream)

    Q_b_rate = Q_b_from(Cs.m_p)
    Q_chi_rate = Q_chi_from(Cs.m_p)
    Drag = D()
    #print('baryon_dark_matter_interaction worked properly')

    return {'baryon': Q_b_rate, 'dark matter': Q_chi_rate, 'drag': Drag}

def Q_chi_from(m_t):
    Q_chi = n_H()*(1/(1+f_He))*(m_chi*m_t/(m_chi + m_t)**2)*(sigma_t_mean_0(m_t)/u_th(m_t))*(np.sqrt(2/np.pi)*(np.exp(-r_t(m_t)**2 /2)/u_th(m_t)**2)*Cs.k*(Tb - Tchi) + m_t*F(r_t(m_t))/r_t(m_t)) * Cs.c**4 / Cs.k #* 10**18
    return Q_chi

def Q_b_from(m_t):
    Q_b = n_chi()*(1/(1+f_He))*(m_chi*m_t/(m_chi + m_t)**2)*(sigma_t_mean_0(m_t)/u_th(m_t))*(np.sqrt(2/np.pi)*(np.exp(-r_t(m_t)**2 /2)/u_th(m_t)**2)*Cs.k*(Tchi - Tb) + m_chi*F(r_t(m_t))/r_t(m_t)) * Cs.c**4 / Cs.k
    return Q_b

def Drag_from(m_t):
    Drag = Cs.c**4 * sigma_t_mean_0(m_t)*((m_chi*n_chi()+rho_b())/(m_chi+m_t))*((m_t*n_H())/(rho_b()))*(F(r_t(m_t))/v_stream**2)
    #print(v_stream)
    return Drag

def D():
    D = Drag_from(Cs.m_p)# + Drag_from(Cs.m_e)
    return D

def F(r_t):
    'Is it necessary to taylor expand F for small r_t?'
    #F = np.erf(r_t/np.sqrt(2)) - np.sqrt(2/np.pi)*r_t*np.exp(-r_t**2 / 2)
    
    if np.abs(r_t) <= 0.01:#0:#
        F = r_t**3 * np.sqrt(2/9/np.pi)#taylor expansion
    else:
        F = erf(r_t/np.sqrt(2)) - np.sqrt(2/np.pi)*r_t*np.exp(-r_t**2 / 2)#accurate expression
    
    return F

def r_t(m_t):
    r_t = v_stream/u_th(m_t)
    return r_t

def sigma_t_mean_0(m_t):
    #sigma_t_mean_0 = (2*np.pi*Xi()*(Cs.c*Cs.hbar*Cs.alpha*epsilon)**2)/(mu_chi_t(m_t)**2 * Cs.c**4) #*v_stream**4) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sigma_t_mean_0 = 1e-45 # m^2
    return sigma_t_mean_0

def u_th(m_t):
    u_th = np.sqrt(Cs.k*(Tb/m_t + Tchi/m_chi))
    return u_th

def rho_b():
    rho_b = rho_b_0*(1+z)**3 # w = p/rho = 0.0
    return rho_b

def n_H():
    #n_H = 2e-1*(1+z)**3 # m^-3; 2e-7*(1+z)**3 cm^-3. https://www.uio.no/studier/emner/matnat/astro/AST4320/h14/beskjeder/combinednotesiipostmidterm.pdf
    n_H =  1.6e-1*(1+z)**3# https://ned.ipac.caltech.edu/level5/Madau6/Madau1_1.html
    #print('n_H')
    return n_H

def n_chi():
    n_chi = f_dm*rho_d()/m_chi
    #print(n_chi)
    return n_chi

def rho_d():
    rho_d = rho_d_0*(1+z)**3 # w = p/rho = 0.0
    return rho_d