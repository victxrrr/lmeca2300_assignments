# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 18:41:23 2025

@author: maxod
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 09:50:12 2025

@author: maxod
"""

### Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
from scipy.special import hankel2
from math import ceil

### Physical constants
rho0 = 1.2 # air, 20°C, 1 atm
f = 10 # [hz]
w = 2*np.pi*f # air -> 343 = w/k
k = w/343
lambda_ = (2*np.pi)/k

### A segment
class Edge:
    def __init__(self,r1,r2):
        self.r1 = r1
        self.r2 = r2
    def norm(self):
        r = self.r1-self.r2
        return(np.sqrt(r[0]**2 + r[1]**2))
    def __integrate_hankel(self, a, b, m, case):
        """
        Aux. function that integrates Hankel function from a to b

        @param a: lower intergration bound
        @param b: upper integration bound
        @param case: symmetric/general case
        @param m: number of integration points
        """
        s = np.linspace(a, b, m)
        if case == 1:
            diff = s[:, np.newaxis] * (self.r2 - self.obs)
        else:
            diff = self.obs - (self.r1 + s[:, np.newaxis] * (self.r2 - self.r1))
        rho = np.linalg.norm(diff, axis=1)
        singularity_mask = k*rho <= lambda_/1000

        evals = np.zeros_like(rho, dtype=np.complex128)
        cst = 1 - 1j*(2/np.pi)*(np.euler_gamma - np.log(2))
        evals[singularity_mask] = cst
        subset_rho = rho[~singularity_mask]
        evals[~singularity_mask] = hankel2(0, k*subset_rho) + 1j*(2/np.pi)*np.log(k*subset_rho)

        h = (b - a) / m

        evals[0] *= 1/2
        evals[-1] *= 1/2
        first_term = h*np.sum(evals)
        if case == 1:
            if a == 0:
                last = 0
            else:
                last = a*(np.log(a)-1)
            second_term = -1j*(2/np.pi)*((b-a)*(np.log(k)+np.log(np.linalg.norm(self.r2-self.obs))) + (b*(np.log(b)-1) - last))
        else:
            pass

        return first_term + second_term
    
    def integral(self, Vn, obs):
        """
        Parameters
        ----------
        Vn : float
            constant normal velocity w.r.t interface
        obs : 2d array
            grid of observation points.

        Returns
        -------
        l'intégrale.

        """
        m=20000
        obs = np.atleast_2d(obs) # (N,2)

        s = np.linspace(0, 1, m) # (m,)
        diff = obs[:, np.newaxis, :] - (self.r1 + s[:, np.newaxis] * (self.r2-self.r1)) # (N,m,2)
        rho = np.linalg.norm(diff, axis=-1) # (N,m)

        H = hankel2(0, k * rho) # (N,m)
        if np.any(np.isnan(H)):
            raise ValueError("Hankel function evaluated at singular point, change geometry or grid.")
        H[:, 0] *= 1/2
        H[:, -1] *= 1/2
        integral = w * rho0 * Vn * self.norm() * np.sum(H, axis=1)/4 / m # (N,)
        return integral
        
        
    def integrate_sym(self, Vn, obs):
        """
        Integrates complex amplitude for an obsever point ON the segment, and thus takes advantage of symmetry.

        @param Vn: constant normal velocity
        @param obs: coord. of observer
        """
        self.obs = obs
        L1 = np.linalg.norm(obs - self.r1)
        L2 = np.linalg.norm(obs - self.r2)

        sbar = np.fmin(L1, L2) / np.fmax(L1, L2)

        m = 40
        if L1 == 0:
            return (Vn*w*rho0*L2/4)*self.__integrate_hankel(0, 1, m, 1)
        elif L1 < L2:
            return (Vn*w*rho0/4)*((L2+L2)*self.__integrate_hankel(0, sbar, m, 1) + (L2)*self.__integrate_hankel(sbar, 1, m, 1))
        else:
            return (Vn*w*rho0*L1/2)*self.__integrate_hankel(0, 1, m, 1)
        
        
r1 = np.array([-lambda_/10, 0])
r2 = np.array([lambda_/10, 0])
# r1 = np.array([1, 0])
# r2 = np.array([1+lambda_/10, 0])
edge = Edge(r1, r2)

pressure_nodes = np.linspace(r1,(r1+r2)/2,100)

phasors = [edge.integrate_sym(1, obs) for obs in pressure_nodes]

temp = np.flip(phasors)
phasors = np.hstack((phasors,temp[1:]))
phasors = np.array(phasors, dtype=np.complex128)
temp = np.flip(-pressure_nodes,0)
pressure_nodes = np.vstack((pressure_nodes,temp[1:]))
np.savetxt("phasors100.txt", phasors)

plt.rcParams["animation.html"] = "jshtml"
plt.ioff()

fig, ax = plt.subplots()
delta_t = (1/f)/10

def animate(t):
    plt.cla()
    pressures = np.real(phasors * np.exp(1j*t*w*delta_t))
    plt.plot(pressure_nodes[:, 0], pressures, linewidth=2)
    plt.plot([r1[0], r2[0]], [r1[1], r2[1]], 'k', linewidth=2)
    ax.set_title("Pressure field $\Delta$t = %f mseg"%delta_t)
    plt.xlim(r1[0]-lambda_/100, r2[0]+lambda_/100)
    plt.ylim(-150, 150)
    plt.xlabel("$x$")
    plt.ylabel("pressure")
    fig.tight_layout()
    
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=100, interval=50)
plt.show()


gridsize = 101
y = np.linspace(-lambda_/40, lambda_/40,gridsize)
x = np.zeros_like(y)
x[:] = pressure_nodes[75,0]
coord_array = np.column_stack((x, y))
plt.scatter(coord_array[:, 0], coord_array[:, 1], c='gray', marker='.', s=1)
plt.scatter(pressure_nodes[:, 0], pressure_nodes[:,1],color = 'red')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis('equal')
plt.title("Geometry overview")
plt.show()

phasors_2 = [edge.integral(1, obs) for obs in coord_array]
phasors_2 = np.array(phasors_2, dtype=np.complex128)
plt.rcParams["animation.html"] = "jshtml"
plt.ioff()

fig, ax = plt.subplots()
delta_t = (1/f)/10

def animate2(t):
    plt.cla()
    pressures_2 = np.real(phasors_2 * np.exp(1j*t*w*delta_t))
    pressures =  np.real(phasors[75] * np.exp(1j*t*w*delta_t))
    plt.plot(coord_array[:, 1], pressures_2, linewidth=2)
    plt.scatter(0, pressures,color = 'red')
    ax.set_title("Pressure field $\Delta$t = %f seg"%delta_t)
    plt.xlim(y[0],y[-1])
    plt.ylim(-200, 200)
    plt.xlabel("$y$")
    plt.ylabel("pressure")
    fig.tight_layout()
    
ani = matplotlib.animation.FuncAnimation(fig, animate2, frames=100, interval=50)
plt.show()
