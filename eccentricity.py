# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:32:49 2020

@author: Ananya
"""

#importing all the required modules
import numpy as np
from scipy.optimize import newton
#assigning values
a = 778.0e6
P = 12.0*365*24
R = 627938.0
rP = 69911.0
tT = 30.0
tF = 24.0
e = 0.0489
w = 1.5*np.pi
#modifying tT and tF
factor = np.sqrt(1.0 - e**2)/(1.0 - e*np.sin(w))
tt, tf = tT*factor, tF*factor
#for egress
t = np.linspace(tf/2, tt/2, 500)
T = 0
M = 2.0*np.pi*(t - T)/P
def solveE(E, M, e):
    return E - e*np.sin(E) - M
E = newton(solveE, np.zeros(M.size), args = (M, e))
tan = np.sqrt(1.0 - e**2)*np.sin(E)/(np.cos(E) - e)
b = a*np.sqrt(1.0 - e**2)
v = np.arctan(tan)
#calculating projected distance
x = -a*np.cos(v + w)*np.cos(w) - b*np.sin(v + w)*np.sin(w)
arg = (R - x)/rP