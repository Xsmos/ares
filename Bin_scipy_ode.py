import scipy.integrate.ode as ode

def dvdt(v):
    3.941261329883112e-14 * v

solver = ode(dvdt).set_integrator('lsoda', nsteps=1e4, atol=1e-8, rtol=1e-8)
solver._integrator.iwork[2] = -1
solver.set_initial_value(v, 0.0).set_f_params(args).set_jac_params(args)
solver.integrate(dt)
