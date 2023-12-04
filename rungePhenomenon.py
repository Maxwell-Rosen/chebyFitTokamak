import numpy as np
import matplotlib.pyplot as plt
import chebyFitFuncs as chb

psi = lambda R,Z: 1/(1+25*R**2)

Rmin, Rmax = -1,1
Zmin, Zmax = -1,1
R = np.linspace(Rmin, Rmax, 1000)
Z = np.linspace(Zmin, Zmax, 3)
R_grid, Z_grid = np.meshgrid(R,Z)

legScale = lambda x: x*np.log(x)**2/np.log(np.log(x))
print(legScale(40))
print(40*40)

chebOrder = np.arange(2,26,2, dtype=int)
if len(chebOrder)%2 == 1:
    chebOrder = np.append(chebOrder, chebOrder[-1]+1)

print(chebOrder)
psiplot = np.zeros((R.shape[0], Z.shape[0]))
psiplot_cheby = np.zeros((chebOrder.shape[0], R.shape[0], Z.shape[0]))


for i in range(0, R.shape[0]):
    for j in range(0, Z.shape[0]):
        psiplot[i, j] = float(psi(R[i], Z[j]))

for ic in range(len(chebOrder)):
    coeff = chb.coeff2D(Rmin, Rmax, Zmin, Zmax, chebOrder[ic], psi)
    print(ic)
    for i in range(0, R.shape[0]):
        for j in range(0, Z.shape[0]):
            psiplot_cheby[ic, i, j] = chb.fit2D(Rmin, Rmax\
                    ,Zmin, Zmax, chebOrder[ic],coeff,R[i],Z[j])

# Make a 4x4 subgrid plotting the psiplot and psiplot_cheby for each chebOrder but take a slice of Z=0
fig, axs = plt.subplots(2, int(chebOrder.shape[0]/2), figsize=(15, 5*chebOrder.shape[0]))
for ic in range(len(chebOrder)):
    CS1 = axs[ic%2, ic//2].plot(R, psiplot[:,0], label='psi')
    CS2 = axs[ic%2, ic//2].plot(R, psiplot_cheby[ic,:,0], label='cheby')
    axs[ic%2, ic//2].set_xlabel("$R/R_0$")
    axs[ic%2, ic//2].set_ylabel("$\psi$")
    axs[ic%2, ic//2].set_title("Chebyshev order = %d" % chebOrder[ic])
    axs[ic%2, ic//2].legend()
plt.show()