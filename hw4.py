# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 14:23:39 2025

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
f = 1/32 # [hz]
w = 2*np.pi*f # air -> 343 = w/k
k = w/343
lambda_ = (2*np.pi)/k

### A segment
class Edge:
    def __init__(self,r1,r2):
        self.r1 = np.asarray(r1)
        self.r2 = np.asarray(r2)

    def __integrate_hankel(self, a, b, m, case):
        """
        Aux. function that integrates Hankel function from a to b via singularity extraction

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
        print(L1-L2)
        print(sbar)
        if L1 == 0:
            print("CASE 1")
            return (Vn*w*rho0*L2/4)*self.__integrate_hankel(0, 1, m, 1)
        elif L1 <= L2:
            print("CASE 2")
            return (Vn*w*rho0/4)*((L2+L2)*self.__integrate_hankel(0, sbar, m, 1) + (L2)*self.__integrate_hankel(sbar, 1, m, 1))
        elif L1-L2<10**-3:
            print("CASE 3")
            return (Vn*w*rho0*L1/2)*self.__integrate_hankel(0, 1, m, 1)
        else:
            print("Not treated")
        
    def integrate_gen(self, Vn, obs,m=100):
        """
        Integrates for general observer point off the segment. Can be vectorized.

        @param Vn: constant normal velocity
        @param obs: coord. of observer (or vector of coords)
        @param m: number of integration points
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
        integral = w * rho0 * Vn * np.linalg.norm(self.r2 - self.r1) * np.sum(H, axis=1)/ (4*m) # (N,)
        return integral
    
### the shape (to change)
# for now, it is simply a square
shape = [Edge([-1, -1], [1, -1]),
         Edge([1, -1], [1, 1]),
         Edge([1, 1], [-1, 1]),
         Edge([-1, 1], [-1, -1])]

# scale it
for s in shape:
    s.r1 = lambda_/20 * s.r1
    s.r2 = lambda_/20 * s.r2

# another geometry
def circle_maker(radius,center):
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
    
    circonf = 2*np.pi*radius
    long_edge = lambda_/10
    n_edges = ceil(circonf/long_edge)
    
    angle = 360/n_edges/180*np.pi
    r1 = np.array(center)+np.array((radius,0))
    all_edges = []
    current_angle = angle
    
    #création des edges
    for i in range(n_edges + 1):
        r2 = center + np.array((radius*np.cos(current_angle), radius*np.sin(current_angle)))
        edge = Edge(r1, r2)
        all_edges.append(edge)
        r1 = r2
        current_angle = current_angle + angle
            
    return all_edges

shape = circle_maker(5000, (0,0))

# show geometry
for s in shape:
    plt.plot([s.r1[0], s.r2[0]], [s.r1[1], s.r2[1]])
plt.show()

def build_system(shape, p_inc):
    """
    build linear system Ax = b as described above

    @param shape: a collection of Edge's
    @param p_inc: function that yields the complex amplitude of the incident field
    """

    N = len(shape)
    A = np.zeros((N, N), dtype=complex)
    b = np.zeros(N, dtype=complex)

    for i in range(N):
        edge = shape[i]
        midpoint = 0.5 * (edge.r1 + edge.r2)
        for j in range(N):
            edge = shape[j]
            if i == j:
                A[i,j] = edge.integrate_sym(1, midpoint)
            else:
                A[i,j] = edge.integrate_gen(1, midpoint, 100)[0]
        b[i] = -p_inc(midpoint)

    return A,b 

### build system          
P0 = 1
u = [np.sqrt(2)/2, np.sqrt(2)/2]
p_inc = lambda r : P0 * np.exp(1j * k * (np.asarray(r) @ np.asarray(u)))
A,b = build_system(shape, p_inc)

### solve system
xi = np.linalg.solve(A,b)

### Create grid
gridsize = 50
bound = lambda_ * 2
x = np.linspace(-bound, bound, gridsize)
X, Y = np.meshgrid(x, x)
coord_array = np.column_stack((X.ravel(), Y.ravel()))

### Normalize abs(xi) for colormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

xi_abs = np.abs(xi)
norm = mcolors.Normalize(vmin=np.min(xi_abs), vmax=np.max(xi_abs))  
cmap = cm.viridis  # Choose a colormap

### Plot
plt.figure(dpi=150)
for i in range(len(shape)):
    color = cmap(norm(xi_abs[i]))  
    plt.plot([shape[i].r1[0], shape[i].r2[0]], [shape[i].r1[1], shape[i].r2[1]], "-", color=color, linewidth=2)
    plt.scatter(coord_array[:, 0], coord_array[:, 1], c='gray', marker="+", s=.75)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
plt.colorbar(sm, label=r"$|x_i|$",ax=plt.gca())
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis('equal')
plt.title("Geometry overview with velocity distribution")
plt.show()

grid_radiated = np.zeros((gridsize * gridsize, len(shape)), dtype=complex)

for i in range(len(shape)):
    grid_radiated[:,i] = shape[i].integrate_gen(xi[i], coord_array)
grid_radiated = np.sum(grid_radiated, axis=-1) # sum up all contributions

grid_incident = p_inc(coord_array)

grid_total = grid_radiated + grid_incident 


import matplotlib.pyplot as plt


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the 3D surface
ax.plot_surface(X, Y, np.real(grid_incident.reshape(X.shape)),cmap=cm.coolwarm)
ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-2, 2),
       xlabel='X', ylabel='Y', zlabel='amplitude',title = 'onde incidente')

plt.show()


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the 3D surface
ax.plot_surface(X, Y, np.real(grid_radiated.reshape(X.shape)),cmap=cm.coolwarm)
ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-2, 2),
       xlabel='X', ylabel='Y', zlabel='amplitude',title = 'onde réfracter')

plt.show()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the 3D surface
ax.plot_surface(X, Y, np.real(grid_total.reshape(X.shape)),cmap=cm.coolwarm)
ax.set(xlim=(-bound, bound), ylim=(-bound, bound), zlim=(-2, 2),
       xlabel='X', ylabel='Y', zlabel='amplitude,',title = 'onde total')

plt.show()



import matplotlib.pyplot as plt
import matplotlib.animation

plt.rcParams["animation.html"] = "jshtml"
plt.ioff()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
delta_t = (1/f)/10

pressures = np.real(grid_total * np.exp(1j*w*0*delta_t))
surf = ax.plot_surface(X, Y, pressures.reshape(X.shape),cmap=cm.coolwarm)
ax.set_xlim(-1*bound, 1*bound)
ax.set_ylim(-1*bound, 1*bound)
ax.set_zlim(-1, 1)
ax.set_title("Pressure field total")
fig.tight_layout()
def animate(t):
    global surf
    surf.remove()
    pressures = np.real(grid_total * np.exp(1j*w*t*delta_t))
    surf = ax.plot_surface(X, Y, pressures.reshape(X.shape),cmap=cm.coolwarm)
    return(surf,)
    

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=50, interval=10)
plt.show()

import matplotlib.pyplot as plt
import matplotlib.animation

plt.rcParams["animation.html"] = "jshtml"
plt.ioff()

fig, ax = plt.subplots()
delta_t = (1/f)/10
im = ax.imshow(np.real(grid_total).reshape(X.shape))
fig.colorbar(im, ax=ax, label='Interactive colorbar')
ax.set_title("Pressure field total")
def animate2(t):
    pressures = np.real(grid_total * np.exp(1j*w*t*delta_t))
    im = ax.imshow(pressures.reshape(X.shape))
    fig.tight_layout()
    return(im,)

ani = matplotlib.animation.FuncAnimation(fig, animate2, frames=50, interval=10)
plt.show()


