# -*- coding: utf-8 -*-
''' This code contains all the functions used for chebychev polynomial fitting. Provided that there is an analytical function as the input, these functions will find the chebychev polynomial representation for that function.

'''
import numpy as np

def chebycoeff(a,b,n,func):
    ''' a - The lower bound of the interval \n
    b - The upper bound of the interval \n
    n - The order of the polynomial approximation \n
    func - The function being approximated
    returns \n
    c - The coefficients used in the chebyshev polynomial
    '''
    f = np.zeros(n)
    c = np.zeros(n)
    bma = 0.5*(b-a)
    bpa = 0.5*(b+a)
    frac = 2/n
    
    for iloop in range(n):
        y = np.cos(np.pi*(iloop+0.5)/n)
        f[iloop] = func(y*bma+bpa)
    for iterm in range(n):
        Sum = 0
        for isum in range(n):
            Sum += f[isum]*np.cos(np.pi*iterm*(isum+0.5)/n) # could vectorize instead of nested for loops. This structure is more friendly with C though.
        c[iterm] = frac*Sum
    c[0]/=2 # Due to the orthogonal product
    return c

def chebyft(a,b,n,c,x):
    '''a - The lower bound of the interval \n
    b - The upper bound of the interval\n
    n - The order of the polynomial approximation \n
    c - The chebyshev coefficients \n
    x - A point at which the chebyshev oolynomial is evaluated at. I'm not sure if this is exactly what we want out of this code. Wouldn't we rather have the function itself for rapid evaluation for any x? Also, I chose to code in a C friendly way, avoiding vectorized computations where you could input an array of x-values and it would spit out the array of y values. This might be more straightfoward to implement in CUDA since it is C based. \n
    returns \n
    val - The value of the chebychev approximation at the point x'''
    if (x-a)*(x-b)>0:
        raise Exception('x is outside the range of routine chebyft')
    xtrans = (2*x-a-b)/(b-a)
    xtranst2 = 2*xtrans
    clenloopvals = np.arange(n-1,0,-1)
    d = np.zeros(n+2)
    for iclen in clenloopvals:
        d[iclen] = xtranst2*d[iclen+1]-d[iclen+2]+c[iclen]
    val = xtrans*d[1]-d[2]+c[0]
    return val

def chebycoeffder(a,b,n,c):
    ''' a - lower bound \n
    b - upper bound \n
    n - order of approximation \n 
    c - the chebychev coeffieicents for the fit \n
    returns \n
    cder - the coefficients for the derivative of the chebyshev approximation'''
    cder = np.zeros(n)
    cder[n-2] = 2*(n-1)*c[n-1]
    loopvals = np.arange(n-3,-1,-1)
    for ider in loopvals:
        cder[ider] = cder[ider+2]+2*(ider+1)*c[ider+1]
    cder[0]/=2
    cder = cder*2/(b-a)
    return cder
    
def chebycoeffint(a,b,n,c):
    ''' a - lower bound \n
    b - upper bound \n
    n - order of approximation \n
    c - the chebychev coefficients for the fit \n
    returns \n
    cint - the constants for a chebyshev fit for the integral of the function '''
    con = 0.25*(b-a)
    loopvals = np.arange(1,n-1,1)
    cint = np.zeros(n)
    Sum = 0
    fac = 1
    c[0]*=2 # To account for the factor due to orthogonality. otherwise it would be an if statement
    for i in loopvals:
        cint[i]=con*(c[i-1]-c[i+1])/i
        Sum += fac*cint[i]
        fac = -fac
    cint[n-1]=con*c[n-2]/(n-1)
    Sum+= fac*cint[n-1]
    cint[0]=Sum
    return cint
    
def cheby2coeff(a,b,c,d,n,func):
    '''a - x lower bound \n
    b - x upper bound \n
    c - y lower bound \n
    d - y upper bound \n
    n - order of fit \n
    func - the function to approximate \n
    returns \n
    c - an nxn array of chebychev coefficients'''
    f = np.zeros([n,n])
    xk=0
    yk=0
    bpa = 0.5*(b+a)
    bma = 0.5*(b-a)
    dpc = 0.5*(d+c)
    dmc = 0.5*(d-c)
    for i in range(n):
        xk = np.cos(np.pi*(i+.5)/n)*bma+bpa
        for j in range(n):
            yk = np.cos(np.pi*(j+.5)/n)*dmc+dpc
            f[i,j]=func(xk,yk)
    cij = np.zeros([n,n])
    for iterm in range(n):
        for jterm in range(n):
            if iterm==0 and jterm==0:
                fac = 1/n**2
            elif iterm==0 or jterm==0:
                fac = 2/n**2
            else:
                fac = 4/n**2
            Sum = 0
            for isum in range(n):
                for jsum in range(n):
                    Sum += f[isum,jsum]*\
                        np.cos(np.pi*iterm*(isum+0.5)/n)*\
                        np.cos(np.pi*jterm*(jsum+0.5)/n)
            cij[iterm,jterm] = fac*Sum
    return cij

def cheby2ft(a,b,c,d,n,cij,x,y):
    '''a - x lower bound \n
    b - x upper bound \n
    c - y lower bound \n
    d - y upper bound \n
    n - order of approximation \n
    cij - the chebychev coefficients \n
    x - the x value of the estimate \n
    y - the y value of the estimate\n
    returns \n
    val - value of the fit at (x,y)'''
    if x<a or x>b:
        raise Exception('x is outside the range of routine cheby2ft')
    elif y<c or y>d :
        raise Exception('y is outside the range of routine cheby2ft')
    g = np.zeros(n)
    for i in range(n):
        g[i] = chebyft(c,d,n,cij[i,:],y)
    val = chebyft(a,b,n,g,x)
    return val
    
def cheby2coeffpartial_x(a,b,c,d,n,func):
    '''a - x lower bound \n
    b - x upper bound \n
    c - y lower bound \n
    d - y upper bound \n
    n - order of approximation \n
    func - the function which to approximate \n
    returns\n
    dx_cij - the chebychev coefficients derivative wrt x
    '''
    cij = cheby2coeff(a,b,c,d,n,func)
    dxcij = np.zeros(np.shape(cij))
    for i in range(n):
        dxcij[:,i]=chebycoeffder(a,b,n,cij[:,i])
    return dxcij

def cheby2coeffpartial_y(a,b,c,d,n,func):
    '''a - x lower bound \n
    b - x upper bound \n
    c - y lower bound \n
    d - y upper bound \n
    n - order of approximation \n
    func - the function which to approximate \n
    returns\n
    dy_cij - the chebychev coefficients derivative wrt y
    '''
    cij = cheby2coeff(a,b,c,d,n,func)
    dycij = np.zeros(np.shape(cij))
    for i in range(n):
        dycij[i,:]=chebycoeffder(c,d,n,cij[i,:])
    return dycij

def cheby2coeffpartial_xy(a,b,c,d,n,func):
    '''a - x lower bound \n
    b - x upper bound \n
    c - y lower bound \n
    d - y upper bound \n
    n - order of approximation \n
    func - the function which to approximate \n
    returns\n
    dxdy_cij - the chebychev coefficients derivative wrt x and y
    '''
    cij = cheby2coeff(a,b,c,d,n,func)
    dycij = np.zeros(np.shape(cij))
    for i in range(n):
        dycij[i,:]=chebycoeffder(c,d,n,cij[i,:])
    dydxcij = np.zeros(np.shape(cij))
    for i in range(n):
        dydxcij[:,i]=chebycoeffder(a,b,n,dycij[:,i])
    return dydxcij


