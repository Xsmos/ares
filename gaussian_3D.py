import numpy as np
import scipy.integrate as integrate

V_rms = 29000 # m/s

def P(v):
    P = np.exp(-v**2 / (2/3*V_rms**2)) / (2*np.pi/3 * V_rms**2)**(3/2) * v**2
    return P

def integrate_gaussian_3d():
    result = integrate.quad(P, -np.infty, np.infty)
    print(result)
    print(result[0] * 2*np.pi * 2)
    return


def test_basic():
    result = integrate.quad(np.sin, 0, np.pi/2)
    print(result)
    return

# print(__name__)
if __name__ == "__main__":
    # print("True")
    test_basic()
    integrate_gaussian_3d()