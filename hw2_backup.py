# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:32:25 2025

@author: maxod
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2
from math import ceil
rho0 = 1.2 # air, 20°C, 1 atm
f = 100 #[hz]
w = 2*np.pi*f# air -> 343 = w/k
k = w/343
longueur_onde = (2*np.pi)/k
m = 50

class Edge:
    def __init__(self,r1,r2):
        self.r1 = r1
        self.r2 = r2

    def norm(self):
        r = self.r1-self.r2
        return(np.sqrt(r[0]**2 + r[1]**2))

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
        obs = np.atleast_2d(obs) # (N,2)

        s = np.linspace(0, 1, m) # (m,)
        diff = obs[:, np.newaxis, :] - (self.r1 + s[:, np.newaxis] * (self.r2-self.r1)) # (N,m,2)
        rho = np.linalg.norm(diff, axis=-1) # (N,m)

        H = hankel2(0, k * rho) # (N,m)
        if np.any(np.isnan(H)):
            raise ValueError("Hankel function evaluated at singular point, change geometry or grid.")
        
        H[:, 0] *= 1/2
        H[:, -1] *= 1/2

        integral = w * rho0 * Vn * self.norm() * np.sum(H, axis=1) / m # (N,)
        return integral
        
def circle_maker(rayon,center):
    """
    Parameters
    ----------
    rayon : int
        rayon en cellule.
    center : (int,int)
        index matrice du centre.
    Returns
    -------
    liste des edges du pourtour de la section.
    """
    
    circonf = 2*np.pi*rayon
    long_edge = longueur_onde/10
    n_edges = ceil(circonf/long_edge)
    
    angle = 360/n_edges/180*np.pi # DONE : adapt this such that each edge's length is 2pi/10k
    r1 = np.array(center)+np.array((rayon,0))
    all_edges = []
    current_angle = angle
    
    #création des edges
    for i in range(n_edges + 1):
        r2 = center + np.array((rayon*np.cos(current_angle), rayon*np.sin(current_angle)))
        edge = Edge(r1, r2)
        all_edges.append(edge)
        r1 = r2
        current_angle = current_angle + angle
            
    return all_edges
    
        
edges = circle_maker(1, (0,0))


velocity_distribution = 0.5 * np.ones(len(edges))
for i in range(len(edges)):
    mid_point_x = (edges[i].r1[0]+edges[i].r2[0])/2
    mid_point_y = (edges[i].r1[1]+edges[i].r2[1])/2
    
    angle = np.arctan2(mid_point_x,mid_point_y)
    velocity_distribution[i] = velocity_distribution[i]*(np.cos(angle)**2)
    
#velocity_distribution = 0.5 * np.ones(len(edges)) # constant velocity distribution

# create grid
gridsize = 50
x = np.linspace(-5*longueur_onde, 5*longueur_onde, gridsize)
X, Y = np.meshgrid(x, x)
coord_array = np.column_stack((X.ravel(), Y.ravel()))

for i in range(len(edges)):
    plt.plot([edges[i].r1[0], edges[i].r2[0]], [edges[i].r1[1], edges[i].r2[1]], "-", linewidth=2)
    plt.scatter(coord_array[:, 0], coord_array[:, 1], c='gray', marker='+', s=1)

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis('equal')
plt.title("Geometry overview")
plt.show()

grid_integrals = np.zeros((gridsize * gridsize, len(edges)), dtype=complex)

for i in range(len(edges)):
    grid_integrals[:,i] = edges[i].integral(velocity_distribution[i], coord_array)
grid_integrals = np.sum(grid_integrals, axis=-1) # sum up all contributions

import matplotlib.pyplot as plt
import matplotlib.animation

plt.rcParams["animation.html"] = "jshtml"
plt.ioff()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

def animate(t):
    plt.cla()
    pressures = np.real(grid_integrals * np.exp(1j* t/2))
    ax.plot_surface(X, Y, pressures.reshape(X.shape))

    plt.xlim(-5*longueur_onde, 5*longueur_onde)
    plt.ylim(-5*longueur_onde, 5*longueur_onde)
    ax.set_zlim(-1000, 1000)
    ax.set_title("Pressure field")
    fig.tight_layout()
    

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=50, interval=10)
plt.show()