# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 14:16:05 2020

@author: Administrator
"""


#Including the luminosity of the planet

#importing all the required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps

#TRANSIT
#*********

#defining functions for the limb darkening laws
#intensity at the centre is taken to be 1

#linear law
def linear(x, u, R):
    mu =  np.sqrt(1.0 - x**2/R**2)
    return (1.0 - u*(1.0 - mu))

#quadratic law
def quadratic(x, a, b, R):
    mu =  np.sqrt(1.0 - x**2/R**2)
    return (1.0 - a*(1.0 - mu) - b*(1.0 - mu)**2)

#non-linear law
def nonlinear(x, u1, u2, u3, u4, R):
    mu =  np.sqrt(1.0 - x**2/R**2)
    return (1.0 - u1*(1.0 - mu**0.5) - u2*(1.0 - mu) - u3*(1.0 - mu**1.5) - u4*(1.0 - mu**2))

#3-parameter non-linear law
def param3(x, u2, u3, u4, R):
    mu =  np.sqrt(1.0 - x**2/R**2)
    return (1.0 - u2*(1.0 - mu) - u3*(1.0 - mu**1.5) - u4*(1.0 - mu**2))

#defining the parameters
R, rP = 10.0, 1.0
u = 0.7215
a, b = 0.6408, 0.0999
u1, u2, u3, u4 = 0.6928, -0.6109, 1.2029, -0.4366
c2, c3, c4 = 1.2877, -0.9263, 0.4009

#Transit with the linear law:

#calculating the total luminosity of the star without the planet
r_total = np.linspace(0, R, 500)
def L_total_linear(r_total, u, R):
    return (linear(r_total, u, R)*2.0*np.pi*r_total)
lum_total_linear = simps(L_total_linear(r_total, u, R), r_total)

#luminosity during ingress
x2 = np.linspace(-(R + rP), -(R - rP), 500)
def L_2_linear(x2, u, R, rP, lum_total_linear):
    I = linear(R - rP, u, R)
    first = I*rP**2*np.arccos((x2 - R)/rP)/lum_total_linear
    second =  I*rP*(x2 - R)*np.sqrt(1.0 - (x2 - R)**2/rP**2)/lum_total_linear
    return (- first + second + 1.0)
lum_2_linear = L_2_linear(-x2, u, R, rP, lum_total_linear) + 1.0e-1*rP**2/R**2

#luminosity for full transit
x3 = np.linspace(-(R - rP), R - rP, 500)
def L_3_linear(x3, u, R, rP, lum_total_linear):
    L_x = linear(x3, u, R)*np.pi*rP**2
    return (1.0 - L_x/lum_total_linear)
lum_3_linear = L_3_linear(x3, u, R, rP, lum_total_linear) + 1.0e-1*rP**2/R**2

#luminosity during egress
x4 = np.linspace((R - rP), (R + rP), 500)
def L_4_linear(x4, u, R, rP, lum_total_linear):
    I = linear(R - rP, u, R)
    first = I*rP**2*np.arccos((R - x4)/rP)/lum_total_linear
    second =  I*rP*(R - x4)*np.sqrt(1.0 - (R - x4)**2/rP**2)/lum_total_linear
    return (first - second + 1.0 - I*np.pi*rP**2/lum_total_linear)
lum_4_linear = L_4_linear(x4, u, R, rP, lum_total_linear) + 1.0e-1*rP**2/R**2


#Transit with the quadratic law:

#calculating the total luminosity of the star without the planet
def L_total_quad(r_total, a, b, R):
    return (quadratic(r_total, a, b, R)*2.0*np.pi*r_total)
lum_total_quad = simps(L_total_quad(r_total, a, b, R), r_total)

#luminosity during ingress
def L_2_quad(x2, u, R, rP, lum_total_quad):
    I = quadratic(R - rP, a, b, R)
    first = I*rP**2*np.arccos((x2 - R)/rP)/lum_total_quad
    second =  I*rP*(x2 - R)*np.sqrt(1.0 - (x2 - R)**2/rP**2)/lum_total_quad
    return (- first + second + 1.0)
lum_2_quad = L_2_quad(-x2, u, R, rP, lum_total_quad) + 1.0e-1*rP**2/R**2

#luminosity for full transit
def L_3_quad(x3, u, R, rP, lum_total_quad):
    L_x = quadratic(x3, a, b, R)*np.pi*rP**2
    return (1.0 - L_x/lum_total_quad)
lum_3_quad = L_3_quad(x3, u, R, rP, lum_total_quad) + 1.0e-1*rP**2/R**2

#luminosity during egress
def L_4_quad(x4, u, R, rP, lum_total_quad):
    I = quadratic(R - rP, a, b, R)
    first = I*rP**2*np.arccos((R - x4)/rP)/lum_total_quad
    second =  I*rP*(R - x4)*np.sqrt(1.0 - (R - x4)**2/rP**2)/lum_total_quad
    return (first - second + 1.0 - I*np.pi*rP**2/lum_total_quad)
lum_4_quad = L_4_quad(x4, u, R, rP, lum_total_quad) + 1.0e-1*rP**2/R**2


#Transit with the non-linear law:

#calculating the total luminosity of the star without the planet
def L_total_nonlinear(r_total, u1, u2, u3, u4, R):
    return (nonlinear(r_total, u1, u2, u3, u4, R)*2.0*np.pi*r_total)
lum_total_nonlinear = simps(L_total_nonlinear(r_total, u1, u2, u3, u4, R), r_total)

#luminosity during ingress
def L_2_nonlinear(x2, u1, u2, u3, u4, R, rP, lum_total_nonlinear):
    I = nonlinear(R - rP, u1, u2, u3, u4, R)
    first = I*rP**2*np.arccos((x2 - R)/rP)/lum_total_nonlinear
    second =  I*rP*(x2 - R)*np.sqrt(1.0 - (x2 - R)**2/rP**2)/lum_total_nonlinear
    return (- first + second + 1.0)
lum_2_nonlinear = L_2_nonlinear(-x2, u1, u2, u3, u4, R, rP, lum_total_nonlinear) + 1.0e-1*rP**2/R**2

#luminosity for full transit
def L_3_nonlinear(x3, u1, u2, u3, u4, R, rP, lum_total_nonlinear):
    L_x = nonlinear(x3, u1, u2, u3, u4, R)*np.pi*rP**2
    return (1.0 - L_x/lum_total_nonlinear)
lum_3_nonlinear = L_3_nonlinear(x3, u1, u2, u3, u4, R, rP, lum_total_nonlinear) + 1.0e-1*rP**2/R**2

#luminosity during egress
def L_4_nonlinear(x4, u1, u2, u3, u4, R, rP, lum_total_nonlinear):
    I = nonlinear(R - rP, u1, u2, u3, u4, R)
    first = I*rP**2*np.arccos((R - x4)/rP)/lum_total_nonlinear
    second =  I*rP*(R - x4)*np.sqrt(1.0 - (R - x4)**2/rP**2)/lum_total_nonlinear
    return (first - second + 1.0 - I*np.pi*rP**2/lum_total_nonlinear)
lum_4_nonlinear = L_4_nonlinear(x4, u1, u2, u3, u4, R, rP, lum_total_nonlinear) + 1.0e-1*rP**2/R**2


#Transit with the 3-parameter non-linear law:

#calculating the total luminosity of the star without the planet
def L_total_3param(r_total, c2, c3, c4, R):
    return (param3(r_total, c2, c3, c4, R)*2.0*np.pi*r_total)
lum_total_3param = simps(L_total_3param(r_total, c2, c3, c4, R), r_total)

#luminosity during ingress
def L_2_3param(x2, c2, c3, c4, R, rP, lum_total_3param):
    I = param3(R - rP, c2, c3, c4, R)
    first = I*rP**2*np.arccos((x2 - R)/rP)/lum_total_3param
    second =  I*rP*(x2 - R)*np.sqrt(1.0 - (x2 - R)**2/rP**2)/lum_total_3param
    return (- first + second + 1.0)
lum_2_3param = L_2_3param(-x2, c2, c3, c4, R, rP, lum_total_3param) + 1.0e-1*rP**2/R**2

#luminosity for full transit
def L_3_3param(x3, c2, c3, c4, R, rP, lum_total_3param):
    L_x = param3(x3, c2, c3, c4, R)*np.pi*rP**2
    return (1.0 - L_x/lum_total_3param)
lum_3_3param = L_3_3param(x3, c2, c3, c4, R, rP, lum_total_3param) + 1.0e-1*rP**2/R**2

#luminosity during egress
def L_4_3param(x4, c2, c3, c4, R, rP, lum_total_3param):
    I = param3(R - rP, c2, c3, c4, R)
    first = I*rP**2*np.arccos((R - x4)/rP)/lum_total_3param
    second =  I*rP*(R - x4)*np.sqrt(1.0 - (R - x4)**2/rP**2)/lum_total_3param
    return (first - second + 1.0 - I*np.pi*rP**2/lum_total_3param)
lum_4_3param = L_4_3param(x4, c2, c3, c4, R, rP, lum_total_3param) + 1.0e-1*rP**2/R**2


#luminosity before ingress
x1 = np.linspace(-10.0*R, -(R + rP), 500)
lum_1_transit = np.linspace(1.0 + 1.0e-4/2, 1.0, 500) + 1.0e-1*rP**2/R**2

#luminosity after egress
x5 = np.linspace((R + rP), 10.0*R, 500)
lum_5_transit = np.linspace(1.0, 1.0 + 1.0e-4/2, 500) + 1.0e-1*rP**2/R**2


#OCCULTATION
#************

#luminosity before ingress
lum_5_occultation = np.linspace(1.0 + 1.0e-4, 1.0 + 1.0e-4/2, 500) + 1.0e-1*rP**2/R**2

#luminosity during ingress
def L_in(x4, R, rP):
    first = (1.0e-4/np.pi + 1.0e-1/(np.pi*R**2))*np.arccos((R - x4)/rP)
    second = (1.0e-4/(np.pi*rP) + 1.0e-1*rP/(np.pi*R**2))*(R - x4)*np.sqrt(1.0 - (R - x4)**2/rP**2)
    return (1.0 + first - second)
lum_in = L_in(x4, R, rP)

#luminosity when the planet is fully covered by the star
lum_occ = np.ones(x3.size)

#luminosity during egress
def L_out(x2, R, rP):
    first = (1.0e-4/np.pi + 1.0e-1/(np.pi*R**2))*np.arccos((x2 - R)/rP)
    second = (1.0e-4/(np.pi*rP) + 1.0e-1*rP/(np.pi*R**2))*(x2 -  R)*np.sqrt(1.0 - (x2 - R)**2/rP**2)
    return (1.0 - first + second + 1.0e-4 + 1.0e-1*rP**2/R**2)
lum_out = L_out(-x2, R, rP)

#luminosity after egress
lum_1_occultation = np.linspace(1.0 + 1.0e-4/2, 1.0 + 1.0e-4, 500) + 1.0e-1*rP**2/R**2

#plotting
fig = plt.figure(figsize = (20, 14))
#transit
#Linear law
plt.plot(x2, lum_2_linear, 'g', label = 'Linear law')
plt.plot(x3, lum_3_linear, 'g')
plt.plot(x4, lum_4_linear, 'g')
#Quadratic law
plt.plot(x2, lum_2_quad, 'r', label = "Quadratic law")
plt.plot(x3, lum_3_quad, 'r')
plt.plot(x4, lum_4_quad, 'r')
#Non-linear law
plt.plot(x2, lum_2_nonlinear, 'b', label = "Non-linear law")
plt.plot(x3, lum_3_nonlinear, 'b')
plt.plot(x4, lum_4_nonlinear, 'b')
#3-parameter non-linear law
plt.plot(x2, lum_2_3param, 'cyan', label = "3-parameter non-linear law")
plt.plot(x3, lum_3_3param, 'cyan')
plt.plot(x4, lum_4_3param, 'cyan')

plt.plot(x1, lum_1_transit, color = 'k', linewidth = 1)
plt.plot(x5, lum_5_transit, color = 'k', linewidth = 1)
#occultation
plt.plot(x1, lum_1_occultation, color = 'k', linewidth = 1)
plt.plot(x2, lum_out, color = 'k', linewidth = 1)
plt.plot(x3, lum_occ, color = 'k', linewidth = 1)
plt.plot(x4, lum_in, color = 'k', linewidth = 1)
plt.plot(x5, lum_5_occultation, color = 'k', linewidth = 1)

plt.legend()
plt.xlabel("Distance between the centres of the star and the planet (x)")
plt.ylabel("Luminosity")
plt.title("Full Light Curve for a Circular Orbit (Including the Planet's Luminosity)")
plt.savefig("Week_2_4.png")
plt.show()

'''The reflected luminosity of the planet is taken to be 10^(-4) times the star's
luminosity.
The intensity of the planet is taken to be 10^(-1) times the star's average intensity, and is
assumed to be constant for the entire planet.'''