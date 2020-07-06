# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:36:09 2020

@author: Administrator
"""


#importing the required modules
import numpy as np
import matplotlib.pyplot as plt

#initialising the parameters and constants(values in SI)
G = 6.674e-11
M_star = 1.989e30
m_p = 1.898e27
c = 3e8
#This is for lensing of a star within a galaxy by a solar-mass-sized star
D = 1e20
#relative velocity of source star and lensing sysem is assumed
v = 6e5
#impact parameter is assumed
u0 = 0.5

rE_star = np.sqrt(4*G*M_star*D/c**2)  #Einstein radius
tE_star = rE_star/v
t0_star = tE_star

tE_p = np.sqrt(m_p/M_star)*tE_star
#position of perturbation due to planet is assumed
t0_p = 1.4*t0_star

#lensing due to the star
t_s = np.linspace(0, 2*tE_star, 500)
tau_s  = (t_s - t0_star)/tE_star
u_s = np.sqrt(u0**2 + tau_s**2)
A_s = (u_s**2 + 2)/(u_s*np.sqrt(u_s**2 + 4))

#perturbation due to the planet
t_p = np.linspace(t0_p - tE_p, t0_p + tE_p, 1000)
tau_p  = (t_p - t0_p)/tE_p
u_p = np.sqrt(u0**2 + tau_p**2)
A_p = (u_p**2 + 2)/(u_p*np.sqrt(u_p**2 + 4))

#For plotting, only the part of the perturbation that is greater than the magnification
#due to the star, is taken.
#magnification due to the star during the perturbation time
tau_sp  = (t_p - t0_star)/tE_star
u_sp = np.sqrt(u0**2 + tau_sp**2)
A_sp = (u_sp**2 + 2)/(u_sp*np.sqrt(u_sp**2 + 4))

#magnification due to the star excluding the perturbation time
t_pert = t_p[np.where(A_p >= A_sp)]
A_pert = A_p[np.where(A_p >= A_sp)]

t_s1 = np.linspace(0, t_pert[0], 500)
tau_s1  = (t_s1 - t0_star)/tE_star
u_s1 = np.sqrt(u0**2 + tau_s1**2)
A_s1 = (u_s1**2 + 2)/(u_s1*np.sqrt(u_s1**2 + 4))

t_s2 = np.linspace(t_pert[-1], 2*tE_star, 500)
tau_s2  = (t_s2 - t0_star)/tE_star
u_s2 = np.sqrt(u0**2 + tau_s2**2)
A_s2 = (u_s2**2 + 2)/(u_s2*np.sqrt(u_s2**2 + 4))

#plotting
fig = plt.figure(figsize = (8, 6))
plt.plot(t_s1/(60*60*24), A_s1, 'k')
plt.plot(t_pert/(60*60*24), A_pert, 'k')
plt.plot(t_s2/(60*60*24), A_s2, 'k')
plt.xlabel('Time in days')
plt.ylabel('Magnification')
plt.title('Microlensing')
plt.savefig('Gravitational_lensing.png')
plt.show()

