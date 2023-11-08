# -*- coding: utf-8 -*-
''' The purpose of this script is to test the chevychev fitting functions, provided in the file ChebyFitFuncs.py. That file should be in the same directory as this one when it is run. This will find the fits for polynomials as a verification. Other user-defined functions can be input below. The script will print all the values of the fits at the prescribed points for confirmation and human check that the values are within precision.

'''


from ChebyFitFuncs import chebycoeff,chebyft,chebycoeffder,chebycoeffint,cheby2coeff,cheby2ft,cheby2coeffpartial_y,cheby2coeffpartial_x,cheby2coeffpartial_xy

# Define the bounds of our test. Ensure that a<b and c<d
a = -.14
b = .64
c = -1
d = 1

# Define the values of x and y, within the boundaries above, you would like to test
x = -.7
y = .5

# Define the order of the chebychev approximation being made
n = 5

# Define our 1D test function to test our approximations
def fitfunc(x):
    return x**5
def intfunc(x,a):
    return 1/3*x**3 - 1/3*a**3
def derfunc(x):
    return 3*x**2

# Define our 2D test function and its derivatives
def fit2Dfunc(x,y):
    return x**2 + y**2
def fit2Dfuncder_x(x,y):
    return 2*x
def fit2Dfuncder_y(x,y):
    return y*2
def fit2Dfuncder_xy(x,y):
    return 0


# Test that the values of the function are correct
ccoff = chebycoeff(a,b,n,fitfunc) # find the 1D chebychev coefficients
valfit1D = chebyft(a,b,n,ccoff,x) # evaluate the 1D chebychev model
valreal1D = fitfunc(x)
print('Check that the values of the function agree')
print(str(valreal1D) + '~=\n' + str(valfit1D))
print(str(valreal1D-valfit1D))


# Derivative check
cder = chebycoeffder(a, b, n, ccoff) # find the coefficients for the derivative
derfit1D = chebyft(a,b,n,cder,x) # evaluate the fit using the derivative coeffieicents
derreal1D = derfunc(x)
print('Check that the derivative of the function agrees')
print(str(derreal1D) + '~=\n' + str(derfit1D))
print(str(derreal1D - derfit1D))

'''
# Integral check
cint = chebycoeffint(a,b,n,ccoff) # find the integral coefficients using the polynomial coefficients
intfit1D = chebyft(a,b,n,cint,x) # find the vale of the integral evaluated at x
intreal1D = intfunc(x,a)
print('Check that the integral of the function agree')
print(str(intreal1D) + '~=\n' + str(intfit1D))
'''

# 2D coefficient check
'''def fitfunc2Check(x,y):
    return x**2
def fitfunc2Check1D(x):
    return x**2
cij = cheby2coeff(a,b,c,d,n,fitfunc2Check)
ccomp = chebycoeff(a,b,n,fitfunc2Check1D)
print('Check that the coefficients of the 2D chebychev representation reduce to the 1D case')
print('2D '+str(cij[0:3,0]))
print('Compared to '+str(ccomp[0:3]))'''
'''

# 2D value check

cij = cheby2coeff(a,b,c,d,n,fit2Dfunc) # find the matrix of coefficients for a 2D chebychev fit
valfit2D = cheby2ft(a,b,c,d,n,cij,x,y) # Evaluate the chebychev model at the prescribed point
valreal2D = fit2Dfunc(x,y)
print('Check the 2D chebychev evaluation values')
print(str(valreal2D) +'~= \n'+str(valfit2D))

cderx = cheby2coeffpartial_x(a,b,c,d,n,fit2Dfunc) # Find the derivative coefficients with respect to x
valderxFit = cheby2ft(a,b,c,d,n,cderx,x,y) # Evaluate the partial wrt x at the prescribed points
valderxReal = fit2Dfuncder_x(x,y)
cdery = cheby2coeffpartial_y(a,b,c,d,n,fit2Dfunc) # Find the derivative coefficients wrt y
valderyFit = cheby2ft(a,b,c,d,n,cdery,x,y) # Evaluate the partial wrt y at the prescribed points
valderyReal = fit2Dfuncder_y(x,y)
cderxy = cheby2coeffpartial_xy(a,b,c,d,n,fit2Dfunc) # Find the derivative ccoefficients wrt x and y
valderxyFit = cheby2ft(a,b,c,d,n,cderxy,x,y) # Find the value of the fit using these coefficients
valderxyReal = fit2Dfuncder_xy(x,y)
print('Partial x check')
print(str(valderxReal)+'~= \n'+str(valderxFit))
print('Partial y check')
print(str(valderyReal)+'~= \n'+str(valderyFit))
print('Partial xy check')
print(str(valderxyReal)+'~= \n'+str(valderxyFit))'''