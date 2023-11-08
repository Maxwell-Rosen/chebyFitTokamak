import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import sympy as sm
import numpy.linalg as lin
import chebyFitFuncs as chb


x = sm.symbols("x")
y = sm.symbols("y")
c1 = sm.symbols("c1")
c2 = sm.symbols("c2")
c3 = sm.symbols("c3")
c4 = sm.symbols("c4")
c5 = sm.symbols("c5")
c6 = sm.symbols("c6")
c7 = sm.symbols("c7")
c = np.r_[c1, c2, c3, c4, c5, c6, c7]


# Equation 8 is psi through psi7
def psi(x, y):
    return (
        x**4 / 8
        + A * (0.5 * x**2 * sm.log(x) - x**4 / 8)
        + c[0] * psi1(x, y)
        + c[1] * psi2(x, y)
        + c[2] * psi3(x, y)
        + c[3] * psi4(x, y)
        + c[4] * psi5(x, y)
        + c[5] * psi6(x, y)
        + c[6] * psi7(x, y)
    )


def psi1(x, y):
    return 1

def psi2(x, y):
    return x**2

def psi3(x, y):
    return y**2 - x**2 * sm.log(x)

def psi4(x, y):
    return x**4 - 4 * x**2 * y**2

def psi5(x, y):
    return (
        2 * y**4
        - 9 * y**2 * x**2
        + 3 * x**4 * sm.log(x)
        - 12 * x**2 * y**2 * sm.log(x)
    )

def psi6(x, y):
    return x**6 - 12 * x**4 * y**2 + 8 * x**2 * y**4

def psi7(x, y):
    return (
        8 * y**6
        - 120 * y**4 * x**2
        + 75 * y**2 * x**4
        - 15 * x**6 * sm.log(x)
        + 180 * x**4 * y**2 * sm.log(x)
        - 120 * x**2 * y**4 * sm.log(x)
    )

def psi_x(x0, y0):
    deriv = sm.diff(psi(x, y), x)
    return deriv.subs(x, x0).subs(y, y0)

def psi_xx(x0, y0):
    deriv = sm.diff(sm.diff(psi(x, y), x), x)
    return deriv.subs(x, x0).subs(y, y0)

def psi_y(x0, y0):
    deriv = sm.diff(psi(x, y), y)
    return deriv.subs(x, x0).subs(y, y0)

def psi_yy(x0, y0):
    deriv = sm.diff(sm.diff(psi(x, y), y), y)
    return deriv.subs(x, x0).subs(y, y0)

# NSTX Params
epsilon = 0.78
kappa = 2.0
delta = 0.35
alpha = np.arcsin(delta)
x_sep = 1 - 1.1 * delta * epsilon
y_sep = 1.1 * kappa * epsilon
q_star = 2.0
A = 0.0

N1 = -((1 + alpha) ** 2) / (epsilon * kappa**2)
N2 = (1 - alpha) ** 2 / (epsilon * kappa**2)
N3 = -kappa / (epsilon * np.cos(alpha) ** 2)


def get_constant_coeff(expr):
    """Helper function for getting constant coefficient"""
    return expr.as_independent(c1, c2, c3, c4, c5, c6, c7)[0]


# Boundary Equations. Equation 10
expressions = np.zeros(7, dtype=object)
expressions[0] = psi(1 + epsilon, 0)
expressions[1] = psi(1 - epsilon, 0)
expressions[2] = psi(x_sep, y_sep)
expressions[3] = psi_x(x_sep, y_sep)
expressions[4] = psi_y(x_sep, y_sep)
expressions[5] = psi_yy(1 + epsilon, 0) + N1 * psi_x(1 + epsilon, 0)
expressions[6] = psi_yy(1 - epsilon, 0) + N2 * psi_x(1 - epsilon, 0)

# Lefthand side Matrix to multiply column vector c
L = np.zeros((7, 7))
for i in range(0, len(expressions)):
    for j in range(0, len(c)):
        L[i, j] = expressions[i].coeff(c[j])

# Righthand side values
R = np.zeros(7)
for i in range(0, len(expressions)):
    R[i] = -get_constant_coeff(expressions[i])

c = lin.inv(L).dot(R)

print(psi(x, y))

##My work begins here. Put in a chebyshev decomposition for psi.
# Step 1: Get the chebyshev coefficients for psi
# Step 1.5: Create a grid and evaluate the chebyshev representation on the grid
# Step 2: Plot psi before, after, and the difference in a 3 pannel plot


# Inverse solver target functions
# Solve for y given x
def func(y, xf, psif):
    return psif - float(psi(xf, y[0]))


def jac(y, xf, psif):
    return float(-psi_y(xf, y[0]))


# Solve for x given y
def funcx(x, yf, psif):
    return psif - float(psi(x[0], yf))


def jacx(x, yf, psif):
    return float(-psi_x(x[0], yf))


load = False
R0 = 1.0

xgrid = R0 * np.arange(0, 2.5, 0.05)
ygrid = R0 * np.arange(-3.5, 3.5, 0.05)
X, Y = np.meshgrid(xgrid, ygrid)

if load:
    psiplot = np.load("./psiplot.npy")
else:
    psiplot = np.zeros((xgrid.shape[0], ygrid.shape[0]))
    for i in range(0, xgrid.shape[0]):
        for j in range(0, ygrid.shape[0]):
            psiplot[i, j] = float(psi(xgrid[i] / R0, ygrid[j] / R0))
    np.save("psiplot", psiplot)


fig, ax = plt.subplots()
# CS = ax.contour(X, Y, psiplot.T, levels =  np.arange(-0.04,0.5,0.005))
CS = ax.contour(X, Y, psiplot.T, levels=np.arange(-0.001 * 10, 0.002 * 10, 0.001))
ax.clabel(CS, inline=True, fontsize=10)
fig.colorbar(CS)
ax.set_xlabel("$R/R_0$")
ax.set_ylabel("$Z/R_0$")
plt.show()

psif = -0.001
candidates = np.zeros((200, 2))
for i, xf in enumerate(np.linspace(R0 * 0.5, R0 * 1.0, num=200)):
    # out = least_squares(func,x0=R0*0.5,jac = jac, bounds = (R0*0.5,R0*0.8), args = (xf,psif))
    out = least_squares(
        func, x0=R0 * 0.5, jac=jac, bounds=(R0 * 0.1, R0 * 1.8), args=(xf, psif)
    )
    candidates[i] = np.r_[xf, out.x]
maxind = np.argmax(candidates[:, 1])
Rtop, Ztop = candidates[maxind]
ax.scatter(Rtop, Ztop)

candidates2 = np.zeros((200, 2))
for i, xf in enumerate(np.linspace(R0 * 0.5, R0 * 1.0, num=200)):
    # out = least_squares(func,x0=R0*0.5,jac = jac, bounds = (R0*0.5,R0*0.8), args = (xf,psif))
    out = least_squares(
        func, x0=-R0 * 0.5, jac=jac, bounds=(-R0 * 1.8, -R0 * 0.1), args=(xf, psif)
    )
    candidates2[i] = np.r_[xf, out.x]
minind = np.argmin(candidates2[:, 1])
Rbot, Zbot = candidates2[minind]
ax.scatter(Rbot, Zbot)
