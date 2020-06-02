# -*- coding: utf-8 -*-
"""
Created on Thu May 28 02:22:42 2020

@author: Administrator
"""


#Importing the required modules
import numpy as np
import matplotlib.pyplot as plt

#intensity at the centre = 1
#defining a function for the linear law
def linear(mu, u):
    return (1.0 - u*(1.0 - mu))

#defining a function for the quadratic law
def quadratic(mu, a, b):
    return (1.0 - a*(1.0 - mu) - b*(1.0 - mu)**2)

#defining a function for the non-linear law
def nonlinear(mu, u1, u2, u3, u4):
    return (1.0 - u1*(1.0 - mu**(1.0/2)) - u2*(1.0 - mu) - u3*(1.0 - mu**(3.0/2)) - u4*(1 - mu**2))

#defining a function for the 3-parameter non-linear law
def param3(mu, c2, c3, c4):
    return (1.0 - c2*(1.0 - mu) - c3*(1.0 - mu**(3.0/2)) - c4*(1.0 - mu**2))

#defining an array for theta from 0 to pi/2 radians
theta = np.linspace(0, np.pi/2, 500)

#assigning values for the limb darkening coefficents
#linear
u = 0.7215
#quadratic
a, b = 0.6408, 0.0999
#non-linear
u1, u2, u3, u4 = 0.6928, -0.6109, 1.2029, -0.4366
#3-parameter non-linear
c2, c3, c4 = 1.2877, -0.9263, 0.4009

#mu = cos(theta)
mu = np.cos(theta)

#plotting
theta_deg = theta*180/np.pi
fig = plt.figure(figsize = (10, 8))
plt.plot(theta_deg, linear(mu, u), label = "Linear law", color = 'g')
plt.plot(theta_deg, quadratic(mu, a, b), label = "Quadratic law", color = 'r')
plt.plot(theta_deg, nonlinear(mu, u1, u2, u3, u4), label = "Non-linear law", color = 'k')
plt.plot(theta_deg, param3(mu, c2, c3, c4), label = "3-parameter non-linear law", color = 'yellow')
plt.xlabel("Theta(degrees)")
plt.ylabel("Normalised intensity")
plt.xticks(ticks = np.arange(0, 100, 10))
plt.legend()
plt.title("Luminosities versus angles")
plt.savefig("Week_2_1.png")
plt.show()
