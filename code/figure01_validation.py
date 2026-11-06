"""
============================================================
Figure 1 Validation
Comparison of Allen-Eggers exponential atmosphere with the
U.S. Standard Atmosphere (1976)

Author : Divyanshu
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Allen-Eggers atmosphere
# ---------------------------------------------------------

RHO0 = 0.0034           # slug/ft^3
BETA = 1.0 / 22000.0    # 1/ft


def rho_allen(h_ft):
    return RHO0 * np.exp(-BETA * h_ft)


# ---------------------------------------------------------
# U.S. Standard Atmosphere (1976)
# ---------------------------------------------------------

g0 = 9.80665
R = 287.05287
T0 = 288.15
P0 = 101325.0

H = np.array([0,11000,20000,32000,47000,51000,71000,84852],dtype=float)
L = np.array([-0.0065,0.0,0.0010,0.0028,0.0,-0.0028,-0.0020])


def rho_standard(h_ft):

    h = np.asarray(h_ft)*0.3048

    rho = np.zeros_like(h)

    for i,height in enumerate(h):

        T = T0
        P = P0

        for j in range(len(L)):

            if height <= H[j+1]:

                dh = height-H[j]

                if L[j]==0:

                    P *= np.exp(-g0*dh/(R*T))

                else:

                    T2 = T+L[j]*dh
                    P *= (T2/T)**(-g0/(R*L[j]))
                    T = T2

                break

            else:

                dh = H[j+1]-H[j]

                if L[j]==0:

                    P *= np.exp(-g0*dh/(R*T))

                else:

                    T2 = T+L[j]*dh
                    P *= (T2/T)**(-g0/(R*L[j]))
                    T = T2

        rho_si = P/(R*T)

        rho[i] = rho_si*0.00194032

    return rho


# ---------------------------------------------------------
# Compute
# ---------------------------------------------------------

h = np.linspace(0,280000,600)

rho_AE = rho_allen(h)
rho_STD = rho_standard(h)

error = 100*(rho_AE-rho_STD)/rho_STD


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

plt.rcParams.update({

    "font.family":"serif",
    "font.size":12,
    "axes.labelsize":13,
    "axes.linewidth":1.0,
    "xtick.direction":"in",
    "ytick.direction":"in",
    "savefig.dpi":600

})

fig,(ax1,ax2)=plt.subplots(
    2,
    1,
    figsize=(6.5,8),
    sharex=True,
    gridspec_kw={"height_ratios":[2,1]}
)

# ---------------------------------------------------------
# Density
# ---------------------------------------------------------

ax1.semilogy(
    h/1000,
    rho_AE,
    linewidth=2.2,
    label=r'Allen-Eggers: $\rho=\rho_0e^{-\beta y}$'
)

ax1.semilogy(
    h/1000,
    rho_STD,
    '--',
    linewidth=2.2,
    label='U.S. Standard Atmosphere (1976)'
)

ax1.axvspan(
    20,
    180,
    color='lightgray',
    alpha=0.25,
    label='Reported validity range'
)

ax1.set_ylabel(r'Density (slug ft$^{-3}$)')
ax1.set_ylim(1e-7,1e-2)
ax1.set_xlim(0,280)

ax1.set_xticks(np.arange(0,281,40))

ax1.grid(which='major',linewidth=0.7)
ax1.grid(which='minor',linestyle=':',linewidth=0.4)

ax1.legend(loc='upper right')


# ---------------------------------------------------------
# Error
# ---------------------------------------------------------

ax2.plot(
    h/1000,
    error,
    color='black',
    linewidth=2
)

ax2.axhline(
    0,
    color='red',
    linestyle='--'
)

ax2.axvspan(
    20,
    180,
    color='lightgray',
    alpha=0.25
)

ax2.set_xlabel(r'Altitude, ft $\times10^{3}$')
ax2.set_ylabel('Error (%)')

ax2.set_xlim(0,280)
ax2.set_xticks(np.arange(0,281,40))

ax2.grid(True)

plt.tight_layout()

plt.savefig("../figures/Figure01_Validation.png")

plt.show()
