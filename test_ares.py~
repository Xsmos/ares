import numpy as np
import matplotlib.pyplot as plt

class test_ares():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        if 'dark_matter_mass' in self.kwargs:
            self.m_chi = self.kwargs['dark_matter_mass']
        if 'initial_v_stream' in self.kwargs:
            self.v = self.kwargs['initial_v_stream']

    def run(self):
        self.history['z'] = np.linspace(0, 400)
        self.history['dTb'] = -0.02 * (29000/self.v) * self.history['z']**2 * np.exp(-(self.history['z']/80)**2 * self.m_chi/0.1)
