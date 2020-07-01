# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 13:17:12 2020

@author: Administrator
"""


#For a circular orbit
#*********************

#Taking i_star = pi/2
#Exaggerated values have been taken for K and rot

#importing the required modules
import numpy as np
import matplotlib.pyplot as plt

#initialising the required parameters
R = 627938.0
rP = 69911.0
a = 778.0e6
P = 12.0*365*24
w = 1.5*np.pi
#amplitude of radial velocity variation is assumed
K = 0.1
#stellar angular velocity is assumed
rot = 8e5/R
#the sky-projected spin-orbit angle is assumed
angle = 10*np.pi/180

#for the radial velocity curve
t = np.linspace(-P/2, P/2, 500)
v = -2*np.pi*t/P
#v0 is taken to be 0
vel = K*(np.cos((w + np.pi) + (v + np.pi)))

#for the Rossiter-McLaughlin effect
tT = P*R*(1 + rP/R)/(np.pi*a)
tF = tT*(1 - rP/R)/(1 + rP/R)
d = rP**2/R**2
b = np.sqrt(((1 - d**0.5)**2 - (tF/tT)**2*(1 + d**0.5)**2)/(1 - (tF/tT)**2))
amp = (2.0/3)*d*rot*np.sqrt(1.0 - b**2)

#for ingress
t1 = np.linspace(-tT/2, -tF/2, 500)
var1 = np.linspace(0, amp, 500)
v1 = -2*np.pi*t1/P
vel1 = var1 + K*(np.cos((w + np.pi) + (v1 + np.pi)))

#for full transit
t2 = np.linspace(-tF/2, tF/2, 500)
var2 = amp*np.sin(-np.pi*t2/tF)
v2 = 2*np.pi*t2/P
vel2 = var2 +  K*(np.cos((w + np.pi) + (v2 + np.pi)))

#for egress
t3 = np.linspace(tF/2, tT/2, 500)
var3 = np.linspace(-amp, 0, 500)
v3 = 2*np.pi*t3/P
vel3 = var3 +  K*(np.cos((w + np.pi) + (v3 + np.pi)))


#plotting
fig = plt.figure(figsize = (10, 8))
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(t, vel, 'k')
ax1.plot(t1, vel1, 'k--', lw = 1)
ax1.plot(t2, vel2, 'k--', lw = 1)
ax1.plot(t3, vel3, 'k--', lw = 1)
ax1.set_title("Radial velocity curve")
ax1.set_ylabel("Radial velocity in m/s")
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(np.linspace(-tT/2 - 0.5, -tT/2, 500), np.zeros(500), 'k')
ax2.plot(t1, var1, 'k')
ax2.plot(t2, var2, 'k')
ax2.plot(t3, var3, 'k')
ax2.plot(np.linspace(tT/2, tT/2 + 0.5, 500), np.zeros(500), 'k')
ax2.set_title("Rossiter-McLaughlin Effect")
ax2.set_xlabel("Time in hours")
ax2.set_ylabel("Radial velocity in m/s")
plt.savefig('Week_3_1.jpg')
