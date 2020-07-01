# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:01:50 2020

@author: Administrator
"""


#For an eccentric orbit
#*********************
#values of K and rot are assumed
#i_star is taken to be pi/2

#importing the required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

#initialising the required parameters
R = 627938.0
rP = 69911.0
a = 778.0e6
P = 12.0*365*24
w = 1.5*np.pi
#eccentricity and amplitude of radial velocity variations are assumed
e = 0.3
K = 0.1
#stellar angular velocity is assumed
rot = 8e5/R
#the sky-projected spin-orbit angle is assumed
angle = 10*np.pi/180

#defining a function to calculate E using Newton-Raphson
def solveE(E, M, e):
    return E - e*np.sin(E) - M

#for the radial velocity curve
t = np.linspace(-P/2, P/2, 500)
M = 2*np.pi*t/P
E = newton(solveE, np.zeros(M.size), args = (M, e))
v = np.arccos((np.cos(E) - e)/(1 - e*np.cos(E)))
#v0 is taken to be 0
vel = K*(np.cos((w + np.pi) + (v + np.pi)) + e*np.cos(w + np.pi))
vel_2 = np.zeros(vel.size)
vel_2[ : int(vel.size/2)] = vel[ : int(vel.size/2)]
vel_2[int(vel.size/2) : ] = -vel[int(vel.size/2) : ] + 2*K*e*np.cos(w + np.pi)

#for the Rossiter-McLaughlin effect
tT = P*np.arcsin(R*(1 + rP/R)/a)/np.pi
tF = np.arcsin(np.sin(tT*np.pi/P)*(1 - rP/R)/(1 + rP/R))*P/np.pi
d = rP**2/R**2
B = np.sqrt(((1 - d**0.5)**2 - (tF/tT)**2*(1 + d**0.5)**2)/(1 - (tF/tT)**2))
factor = np.sqrt(1 - e**2)/(1 + e*np.sin(w))
tt = tT*factor
tf = tF*factor
b = B*(1 - e**2)/(1 + e*np.sin(w))
amp = (2.0/3)*d*rot*np.sqrt(1.0 - b**2)

#ingress
t1 = np.linspace(-tt/2, -tf/2, 500)
M1 = -2*np.pi*t1/P
E1 = newton(solveE, np.zeros(M1.size), args = (M1, e))
v1 = np.arccos((np.cos(E1) - e)/(1 - e*np.cos(E1)))
var1 = np.linspace(0, amp, 500)
vel1 = var1 + K*(np.cos((w + np.pi) + (v1 + np.pi)))

#for full transit
t2 = np.linspace(-tf/2, tf/2, 500)
var2 = amp*np.sin(-np.pi*t2/tf)
M2 = 2*np.pi*t2/P
E2 = newton(solveE, np.zeros(M2.size), args = (M2, e))
v2 = np.arccos((np.cos(E2) - e)/(1 - e*np.cos(E2)))
vel2 = var2 +  K*(np.cos((w + np.pi) + (v2 + np.pi)))

#for egress
t3 = np.linspace(tf/2, tt/2, 500)
var3 = np.linspace(-amp, 0, 500)
M3 = 2*np.pi*t3/P
E3 = newton(solveE, np.zeros(M3.size), args = (M3, e))
v3 = np.arccos((np.cos(E3) - e)/(1 - e*np.cos(E3)))
vel3 = var3 +  K*(np.cos((w + np.pi) + (v3 + np.pi)))

#plotting
fig = plt.figure(figsize = (10, 8))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(t, vel_2, 'k')
ax1.plot(t1, vel1, 'k--', lw = 1)
ax1.plot(t2, vel2, 'k--', lw = 1)
ax1.plot(t3, vel3, 'k--', lw = 1)
ax1.set_title("Radial velocity curve")
ax1.set_ylabel("Radial velocity in m/s")
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(np.linspace(-tt/2 - 0.5, -tt/2, 500), np.zeros(500), 'k')
ax2.plot(t1, var1, 'k')
ax2.plot(t2, var2, 'k')
ax2.plot(t3, var3, 'k')
ax2.plot(np.linspace(tt/2, tt/2 + 0.5, 500), np.zeros(500), 'k')
ax2.set_title("Rossiter-McLaughlin Effect")
ax2.set_xlabel("Time in hours")
ax2.set_ylabel("Radial velocity in m/s")
plt.savefig('Week_3_2.jpg')