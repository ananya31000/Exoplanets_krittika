# -*- coding: utf-8 -*-
"""
Created on Mon May 25 16:46:38 2020

@author: Administrator
"""


#Importing the required packages
import numpy as np
import matplotlib.pyplot as plt

#This program takes in tT(total transit duration), tF(transit duration with the
#planet disk fully superimposed on the stellar disk), d(transit depth), and
#R(stellar radius) as user inputs.
tT = float(input("Enter total transit duration:"))
tF = float(input("Enter tF:"))
d = float(input("Enter transit depth:"))
R = float(input("Enter stellar radius:"))

#Defining impact parameter b
b = np.sqrt(((1.0 - np.sqrt(d))**2 - (tF/tT)**2*(1.0 + np.sqrt(d))**2)/(1.0 - (tF/tT)**2))

#1st part of the light curve: When the planet is not projected on the stellar disk
y1 = np.linspace(-0.5 - R*np.sqrt((1.0 + np.sqrt(d))**2 - b**2), -R*np.sqrt((1.0 + np.sqrt(d))**2 - b**2), 500)
def L_1(y1, d, R):
    return np.ones(y1.size)
lum_1 = L_1(y1, d, R)

#2nd part of the light curve: Ingress
#The first element of the term array raised an error in the arccos function despite being
#equal to 1.000e00. So it was converted to integer form.
y2 = np.linspace(-R*np.sqrt((1.0 + np.sqrt(d))**2 - b**2), -R*np.sqrt((1.0 - np.sqrt(d))**2 - b**2), 500)
def L_2(y2, d, R):
    x = np.sqrt(y2**2/R**2 + b**2) - 1.0
    term = x/np.sqrt(d)
    term[0] = int(term[0])
    first = (d/np.pi)*np.arccos(term)
    second = (np.sqrt(d)/np.pi)*x*np.sqrt(abs(1.0 - x**2/d))
    return (-first + second + 1.0)
lum_2 = L_2(y2, d, R)

#3rd part of the light curve: When the planet is fully projected on the stellar disk
y3 = np.linspace(-R*np.sqrt((1.0 - np.sqrt(d))**2 - b**2), R*np.sqrt((1.0 - np.sqrt(d))**2 - b**2), 500)
def L_3(y3, d, R):
    return np.ones(y3.size)*(1 - d)
lum_3 = L_3(y3, d, R)

#4th part of the light curve: Egress
#The last element of the term array raised an error in the arccos function despite being
#equal to 1.000e00. So it was converted to integer form.
y4 = np.linspace(R*np.sqrt((1.0 - np.sqrt(d))**2 - b**2), R*np.sqrt((1.0 + np.sqrt(d))**2 - b**2), 500)
def L_4(y4, d, R):
    x = (1.0 - np.sqrt(y4**2/R**2 + b**2))
    term = x/np.sqrt(d)
    term[-1] = int(term[-1])
    first = (d/np.pi)*np.arccos(term)
    second = (np.sqrt(d)/np.pi)*x*np.sqrt(abs(1.0 - x**2/d))
    return (first - second + 1.0 - d)
lum_4 = L_4(y4, d, R)

#5th part of the light curve: When the planet has left the stellar disk
y5 = np.linspace(R*np.sqrt((1.0 + np.sqrt(d))**2 - b**2), R*np.sqrt((1.0 + np.sqrt(d))**2 - b**2) + 0.5, 500)
def L_5(y5, d, R):
    return np.ones(y5.size)
lum_5 = L_5(y5, d, R)

#plotting
fig = plt.figure(figsize = (15, 5))
plt.plot(y1, lum_1, 'k')
plt.plot(y2, lum_2, 'k')
plt.plot(y3, lum_3, 'k')
plt.plot(y4, lum_4, 'k')
plt.plot(y5, lum_5, 'k')
plt.xlabel("Distance of the planet's centre from conjunction (y)")
plt.ylabel("Normalised luminosity L")
plt.title("Simple Light Curve for Transit")
plt.savefig("Week_1.png")
plt.show()

'''OUTPUT:
Enter total transit duration:0.75

Enter tF:0.5

Enter transit depth:0.01

Enter stellar radius:1.0'''
