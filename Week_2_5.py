# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 13:53:19 2020

@author: Administrator
"""


#Transit light curve with eccentricity
#This does not work for exaggerated values of eccentricity

#importing the required modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.optimize import newton

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
R, rP = 627938.0, 69911.0
u = 0.7215
a, b = 0.6408, 0.0999
u1, u2, u3, u4 = 0.6928, -0.6109, 1.2029, -0.4366
c2, c3, c4 = 1.2877, -0.9263, 0.4009
A = 778.0e6
P = 12.0*365*24
w, e = 1.5*np.pi, 0.01

tT = P*np.arcsin(R*(1.0 + rP/R)/A)/np.pi
tF = np.arcsin(np.sin(tT*np.pi/P)*(1.0 - rP/R)/(1.0 + rP/R))*P/np.pi
factor = np.sqrt(1.0 - e**2)/(1 + e*np.sin(w))
tt = tT*factor
tf = tF*factor

#defining a function to find E using Newton-Raphson
def solveE(E, M, e):
    return E - e*np.sin(E) - M

#Transit with the linear law:

#calculating the total luminosity of the star without the planet
r_total = np.linspace(0, R, 500)
def L_total_linear(r_total, u, R):
    return (linear(r_total, u, R)*2.0*np.pi*r_total)
lum_total_linear = simps(L_total_linear(r_total, u, R), r_total)

#luminosity before ingress
t1 = np.linspace(-tt/2 - 0.25, -tt/2, 500)
def L_1_linear(x1, lum_total_linear):
    return np.ones(x1.size)
lum_1_linear = L_1_linear(t1, lum_total_linear)

#luminosity during ingress
t2 = np.linspace(-tt/2, -tf/2, 500)
M2 = -2*np.pi*t2/P
E2 = newton(solveE, np.zeros(M2.size), args = (M2, e))
v2 = np.arccos((np.cos(E2) - e)/(1 - e*np.cos(E2)))
r2 = A*(1 - e**2)/(1 + e*np.cos(v2))
x2 = r2*np.cos(2*np.pi - w - v2)
def L_2_linear(x2, u, R, rP, lum_total_linear):
    I = linear(R - rP, u, R)
    arg = (x2 - R)/rP
    arg2 = np.zeros(arg.size)
    arg2[np.where(abs(arg) < 1)] = arg[np.where(abs(arg) < 1)]
    for i in range(arg.size):
        if abs(arg[i]) >= 1:
            arg2[i] = int(arg[i])
    first = I*rP**2*np.arccos(arg2)/lum_total_linear
    second =  I*rP**2*arg2*np.sqrt(1.0 - arg2**2)/lum_total_linear
    return (- first + second + 1.0)
lum_2_linear = L_2_linear(x2, u, R, rP, lum_total_linear)

#luminosity for full transit
t3 = np.linspace(-tf/2, tf/2, 500)
M3 = 2*np.pi*t3/P
E3 = newton(solveE, np.zeros(M3.size), args = (M3, e))
v3 = np.arccos((np.cos(E3) - e)/(1 - e*np.cos(E3)))
r3 = A*(1 - e**2)/(1 + e*np.cos(v3))
x3 = r3*np.cos(2*np.pi - w - v3)
def L_3_linear(x3, u, R, rP, lum_total_linear):
    L_x = linear(x3, u, R)*np.pi*rP**2
    return (1.0 - L_x/lum_total_linear)
lum_3_linear = L_3_linear(x3, u, R, rP, lum_total_linear)

#luminosity during egress
t4 = np.linspace(tf/2, tt/2, 500)
M4 = 2*np.pi*t4/P
E4 = newton(solveE, np.zeros(M4.size), args = (M4, e))
v4 = np.arccos((np.cos(E4) - e)/(1 - e*np.cos(E4)))
r4 = A*(1 - e**2)/(1 + e*np.cos(v4))
x4 = r4*np.cos(2*np.pi - w - v4)
def L_4_linear(x4, u, R, rP, lum_total_linear):
    I = linear(R - rP, u, R)
    arg = (R - x4)/rP
    arg2 = np.zeros(arg.size)
    arg2[np.where(abs(arg) < 1)] = arg[np.where(abs(arg) < 1)]
    for i in range(arg.size):
        if abs(arg[i]) >= 1:
            arg2[i] = int(arg[i])
    first = I*rP**2*np.arccos(arg2)/lum_total_linear
    second =  I*rP**2*arg2*np.sqrt(1.0 - arg2**2)/lum_total_linear
    return (first - second + 1.0 - I*np.pi*rP**2/lum_total_linear)
lum_4_linear = L_4_linear(x4, u, R, rP, lum_total_linear)

#luminosity after egress
t5 = np.linspace(tt/2, tt/2 + 0.25, 500)
def L_5_linear(x5, lum_total_linear):
    return np.ones(x5.size)
lum_5_linear = L_5_linear(t5, lum_total_linear)

#Transit with the quadratic law:

#calculating the total luminosity of the star without the planet
def L_total_quad(r_total, a, b, R):
    return (quadratic(r_total, a, b, R)*2.0*np.pi*r_total)
lum_total_quad = simps(L_total_quad(r_total, a, b, R), r_total)

#luminosity before ingress
def L_1_quad(x1, lum_total_quad):
    return np.ones(x1.size)
lum_1_quad = L_1_quad(t1, lum_total_quad)

#luminosity during ingress
def L_2_quad(x2, u, R, rP, lum_total_quad):
    I = quadratic(R - rP, a, b, R)
    arg = (x2 - R)/rP
    arg2 = np.zeros(arg.size)
    arg2[np.where(abs(arg) < 1)] = arg[np.where(abs(arg) < 1)]
    for i in range(arg.size):
        if abs(arg[i]) >= 1:
            arg2[i] = int(arg[i])
    first = I*rP**2*np.arccos(arg2)/lum_total_quad
    second =  I*rP**2*arg2*np.sqrt(1.0 - arg2**2)/lum_total_quad
    return (- first + second + 1.0)
lum_2_quad = L_2_quad(x2, u, R, rP, lum_total_quad)

#luminosity for full transit
def L_3_quad(x3, u, R, rP, lum_total_quad):
    L_x = quadratic(x3, a, b, R)*np.pi*rP**2
    return (1.0 - L_x/lum_total_quad)
lum_3_quad = L_3_quad(x3, u, R, rP, lum_total_quad)

#luminosity during egress
def L_4_quad(x4, u, R, rP, lum_total_quad):
    I = quadratic(R - rP, a, b, R)
    arg = (R - x4)/rP
    arg2 = np.zeros(arg.size)
    arg2[np.where(abs(arg) < 1)] = arg[np.where(abs(arg) < 1)]
    for i in range(arg.size):
        if abs(arg[i]) >= 1:
            arg2[i] = int(arg[i])
    first = I*rP**2*np.arccos(arg2)/lum_total_quad
    second =  I*rP**2*arg2*np.sqrt(1.0 - arg2**2)/lum_total_quad
    return (first - second + 1.0 - I*np.pi*rP**2/lum_total_quad)
lum_4_quad = L_4_quad(x4, u, R, rP, lum_total_quad)

#luminosity after egress
def L_5_quad(x5, lum_total_quad):
    return np.ones(x5.size)
lum_5_quad = L_5_quad(t5, lum_total_quad)

#Transit with the non-linear law:

#calculating the total luminosity of the star without the planet
def L_total_nonlinear(r_total, u1, u2, u3, u4, R):
    return (nonlinear(r_total, u1, u2, u3, u4, R)*2.0*np.pi*r_total)
lum_total_nonlinear = simps(L_total_nonlinear(r_total, u1, u2, u3, u4, R), r_total)

#luminosity before ingress
def L_1_nonlinear(x1, lum_total_nonlinear):
    return np.ones(x1.size)
lum_1_nonlinear = L_1_nonlinear(t1, lum_total_nonlinear)

#luminosity during ingress
def L_2_nonlinear(x2, u1, u2, u3, u4, R, rP, lum_total_nonlinear):
    I = nonlinear(R - rP, u1, u2, u3, u4, R)
    arg = (x2 - R)/rP
    arg2 = np.zeros(arg.size)
    arg2[np.where(abs(arg) < 1)] = arg[np.where(abs(arg) < 1)]
    for i in range(arg.size):
        if abs(arg[i]) >= 1:
            arg2[i] = int(arg[i])
    first = I*rP**2*np.arccos(arg2)/lum_total_nonlinear
    second =  I*rP**2*arg2*np.sqrt(1.0 - arg2**2)/lum_total_nonlinear
    return (- first + second + 1.0)
lum_2_nonlinear = L_2_nonlinear(x2, u1, u2, u3, u4, R, rP, lum_total_nonlinear)

#luminosity for full transit
def L_3_nonlinear(x3, u1, u2, u3, u4, R, rP, lum_total_nonlinear):
    L_x = nonlinear(x3, u1, u2, u3, u4, R)*np.pi*rP**2
    return (1.0 - L_x/lum_total_nonlinear)
lum_3_nonlinear = L_3_nonlinear(x3, u1, u2, u3, u4, R, rP, lum_total_nonlinear)

#luminosity during egress
def L_4_nonlinear(x4, u1, u2, u3, u4, R, rP, lum_total_nonlinear):
    I = nonlinear(R - rP, u1, u2, u3, u4, R)
    arg = (R - x4)/rP
    arg2 = np.zeros(arg.size)
    arg2[np.where(abs(arg) < 1)] = arg[np.where(abs(arg) < 1)]
    for i in range(arg.size):
        if abs(arg[i]) >= 1:
            arg2[i] = int(arg[i])
    first = I*rP**2*np.arccos(arg2)/lum_total_nonlinear
    second =  I*rP**2*arg2*np.sqrt(1.0 - arg2**2)/lum_total_nonlinear
    return (first - second + 1.0 - I*np.pi*rP**2/lum_total_nonlinear)
lum_4_nonlinear = L_4_nonlinear(x4, u1, u2, u3, u4, R, rP, lum_total_nonlinear)

#luminosity after egress
def L_5_nonlinear(x5, lum_total_nonlinear):
    return np.ones(x5.size)
lum_5_nonlinear = L_5_nonlinear(t5, lum_total_nonlinear)

#Transit with the 3-parameter non-linear law:

#calculating the total luminosity of the star without the planet
def L_total_3param(r_total, c2, c3, c4, R):
    return (param3(r_total, c2, c3, c4, R)*2.0*np.pi*r_total)
lum_total_3param = simps(L_total_3param(r_total, c2, c3, c4, R), r_total)

#luminosity before ingress
def L_1_3param(x1, lum_total_3param):
    return np.ones(x1.size)
lum_1_3param = L_1_3param(t1, lum_total_3param)

#luminosity during ingress
def L_2_3param(x2, c2, c3, c4, R, rP, lum_total_3param):
    I = param3(R - rP, c2, c3, c4, R)
    arg = (x2 - R)/rP
    arg2 = np.zeros(arg.size)
    arg2[np.where(abs(arg) < 1)] = arg[np.where(abs(arg) < 1)]
    for i in range(arg.size):
        if abs(arg[i]) >= 1:
            arg2[i] = int(arg[i])
    first = I*rP**2*np.arccos(arg2)/lum_total_3param
    second =  I*rP**2*arg2*np.sqrt(1.0 - arg2**2)/lum_total_3param
    return (- first + second + 1.0)
lum_2_3param = L_2_3param(x2, c2, c3, c4, R, rP, lum_total_3param)

#luminosity for full transit
def L_3_3param(x3, c2, c3, c4, R, rP, lum_total_3param):
    L_x = param3(x3, c2, c3, c4, R)*np.pi*rP**2
    return (1.0 - L_x/lum_total_3param)
lum_3_3param = L_3_3param(x3, c2, c3, c4, R, rP, lum_total_3param)

#luminosity during egress
def L_4_3param(x4, c2, c3, c4, R, rP, lum_total_3param):
    I = param3(R - rP, c2, c3, c4, R)
    arg = (R - x4)/rP
    arg2 = np.zeros(arg.size)
    arg2[np.where(abs(arg) < 1)] = arg[np.where(abs(arg) < 1)]
    for i in range(arg.size):
        if abs(arg[i]) >= 1:
            arg2[i] = int(arg[i])
    first = I*rP**2*np.arccos(arg2)/lum_total_3param
    second =  I*rP**2*arg2*np.sqrt(1.0 - arg2**2)/lum_total_3param
    return (first - second + 1.0 - I*np.pi*rP**2/lum_total_3param)
lum_4_3param = L_4_3param(x4, c2, c3, c4, R, rP, lum_total_3param)

#luminosity after egress
def L_5_3param(x5, lum_total_3param):
    return np.ones(x5.size)
lum_5_3param = L_5_3param(t5, lum_total_3param)

#plotting
#Linear law
fig = plt.figure(figsize = (15, 7))
plt.plot(t1, lum_1_linear, color = 'g', label = "Linear law")
plt.plot(t2, lum_2_linear, 'g')
plt.plot(t3, lum_3_linear, 'g')
plt.plot(t4, lum_4_linear, 'g')
plt.plot(t5, lum_5_linear, 'g')
#Quadratic law
plt.plot(t1, lum_1_quad, color = 'r', label = "Quadratic law")
plt.plot(t2, lum_2_quad, 'r')
plt.plot(t3, lum_3_quad, 'r')
plt.plot(t4, lum_4_quad, 'r')
plt.plot(t5, lum_5_quad, 'r')
#Non-linear law
plt.plot(t1, lum_1_nonlinear, color = 'b', label = "Non-linear law")
plt.plot(t2, lum_2_nonlinear, 'b')
plt.plot(t3, lum_3_nonlinear, 'b')
plt.plot(t4, lum_4_nonlinear, 'b')
plt.plot(t5, lum_5_nonlinear, 'b')
#3-parameter non-linear law
plt.plot(t1, lum_1_3param, color = 'cyan', label = "3-parameter non-linear law")
plt.plot(t2, lum_2_3param, 'cyan')
plt.plot(t3, lum_3_3param, 'cyan')
plt.plot(t4, lum_4_3param, 'cyan')
plt.plot(t5, lum_5_3param, 'cyan')

plt.legend()
plt.xlabel("Time in hours")
plt.ylabel("Luminosity")
plt.title("Transit with Limb Darkening for an Eccentric Orbit")
plt.savefig("Week_2_5.png")
plt.show()