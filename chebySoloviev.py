# -*- coding: utf-8 -*-

import chebyFitFuncs as chb
import numpy as np
import matplotlib.pyplot as plt

def Soloviev(r,z):
    R1 = 0 #Soloviev Parameter
    ri = 0.15 #inner radius of plasma surface
    re = 0.6 #outer radius of plasma surface
    rt = 0.33 #r at the top plasma point
    zt = 0.35 #z at the top plasma point
    R2 = np.sqrt(ri**2 + re**2 - R1**2)
    Z2 = np.sqrt((zt**2 * R2**2)/(R2**2 + R1**2 - 2*rt**2))
    psi = (1-((r/R2)**2)-((z/Z2)**2))*((r**2) - (R1**2))
    return psi

R1 = 0 #Soloviev Parameter
ri = 0.15 #inner radius of plasma surface
re = 0.6 #outer radius of plasma surface
rt = 0.33 #r at the top plasma point
zt = 0.35 #z at the top plasma point
R2 = np.sqrt(ri**2 + re**2 - R1**2)
Z2 = np.sqrt((zt**2 * R2**2)/(R2**2 + R1**2 - 2*rt**2))
n = 5 #order of chebyshev representation

coeffs = chb.coeff2D(R1,R2,-Z2,Z2,n,Soloviev)
coeffs[abs(coeffs)<1e-10]=0

rvals = np.linspace(R1,R2,100)
zvals = np.linspace(-Z2,Z2,100)
gridPTS = np.meshgrid(rvals,zvals)
solnGrid = np.zeros([len(rvals),len(zvals)])
for ir in range(len(rvals)):
    for iz in range(len(zvals)):
        solnGrid[ir,iz] = chb.fit2D(R1, R2, -Z2, Z2, n, coeffs, gridPTS[0][ir,iz], gridPTS[1][ir,iz])
        
solnGrid[solnGrid<0]=-.01
plt.contourf(rvals,zvals,solnGrid,10, cmap='Purples')
plt.xlabel('r')
plt.ylabel('z')
plt.colorbar()
plt.title('Soloviev solution to the grad-shafranov equation on LTX \n using Chebyshev polynomials')
plt.show()