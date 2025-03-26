# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 16:32:57 2025

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
f = 100 # [hz]
w = 2*np.pi*f # air -> 343 = w/k
k = w/343
lambda_ = (2*np.pi)/k
    
class Point:
    def __init__(self,x,y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
    def integrate_gen(self, Vn, obs):
        """
        Integrates for general observer ,point source.

        @param Vn: constant normal velocity
        @param obs: coord. of observer (or vector of coords)
        """
        obs = np.atleast_2d(obs) # (N,2)
        
        diff = obs - (self.x,self.y)
        rho = np.linalg.norm(diff, axis=-1) # (N,m)
        H = hankel2(0, k * rho) # (N,m)
        
        if np.any(np.isnan(H)):
            raise ValueError("Hankel function evaluated at singular point, change geometry or grid.")
        integral = w * rho0 * Vn  * H/4 # (N,)
        return integral
    def integrate_decenter(self,P_obs,Vn, obs):
        """
        Integrates for decenter observer ,point source.

        @param Vn: constant normal velocity
        @param obs: coord. of observer (or vector of coords)
        @param 
        """
        m = 50
        n=21
        def e(l):
            if l==0:
                return 1
            return 2
        beta = np.linspace(0, 2*np.pi,m)
        l = np.arange(0,n,1)
        obs = np.atleast_2d(obs) # (N,2)
        
        R0 = np.linalg.norm((self.x-P_obs.x,self.y-P_obs.y))
        phi0 = np.arctan2(P_obs.y-self.y,P_obs.x-self.x)
        #print(phi0*180/3.14)
        r0 = obs - (P_obs.x,P_obs.y) # (N,2)
        temp = np.zeros((n,len(r0)),dtype=complex)
        temp2 = np.zeros((m,len(r0)),dtype=complex)
        for i in range(len(beta)):
            u_beta = [np.cos(beta[i]), np.sin(beta[i])]
            for j in l:
                temp[j,:] = hankel2(j, k * R0)*e(l[j])*((-1j)**l[j])*np.cos(l[j]*(beta[i]-phi0))*np.exp(1j * k * (np.asarray(r0) @ np.asarray(u_beta))) # (n,N)
                #print(temp)
            temp2[i,:] = np.sum(temp,axis = 0)  #(m,N)
            #print(temp2)
        H = np.sum(temp2,axis=0)
        if np.any(np.isnan(H)):
            raise ValueError("Hankel function evaluated at singular point, change geometry or grid.")
        integral = w * rho0 * Vn  * H/4 /(2*np.pi) # (N,)
        return integral
    
    
point_source = Point(-10*lambda_, -10*lambda_)
point_obs = Point(0,0)
gridsize = 50
x = np.linspace(-2*lambda_, 2*lambda_, gridsize)
X, Y = np.meshgrid(x, x)
coord_array = np.column_stack((X.ravel(), Y.ravel()))
grid_integrals = point_source.integrate_decenter(point_obs,1, coord_array)

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.animation

plt.rcParams["animation.html"] = "jshtml"
plt.ioff()
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
delta_t = (1/f)/10

pressures = np.real(grid_integrals * np.exp(1j*w*0*delta_t))
surf = ax.plot_surface(X, Y, pressures.reshape(X.shape),cmap=cm.coolwarm)
ax.set_xlim(-2*lambda_, 2*lambda_)
ax.set_ylim(-2*lambda_, 2*lambda_)
ax.set_zlim(-200, 200)
ax.set_title("Pressure field total")
fig.tight_layout()
def animate(t):
    global surf
    surf.remove()
    pressures = np.real(grid_integrals * np.exp(1j*w*t*delta_t))
    surf = ax.plot_surface(X, Y, pressures.reshape(X.shape),cmap=cm.coolwarm)
    fig.tight_layout()
    return(surf,)
    

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=50, interval=10)
plt.show()