# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 14:32:25 2025

@author: maxod
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hankel2
from math import ceil,floor

omega = 1
k = 5
rho0 = 1
m = 100

class Facette:
    def __init__(self,r1,r2):
        self.r1 = r1
        self.r2 = r2
    def norm(self):
        r = self.r1-self.r2
        return(np.sqrt(r[0]**2 + r[1]**2))
    def integrale(self, Vn,source):
        """
        Parameters
        ----------
        Vn : float
            vitesse normal.
        source : (int,int)
            point observateur.

        Returns
        -------
        l'intégrale.

        """
        # à faire
        s = np.linspace(0, 1, m)
        diff = source - (self.r1 + s[:, np.newaxis] * (self.r2-self.r1))
        rho = np.sqrt(diff[:, 0]**2 + diff[:, 1]**2)

        H = hankel2(0, k * rho)
        H[0] *= 1/2
        H[-1] *= 1/2

        return omega * rho0 * Vn * self.norm() * np.sum(H) / m

        
def cercle_maker(n_facette,rayon,center,size_matirix):
    """
    Parameters
    ----------
    n_facette : int
        nombre de faces.
    rayon : int
        rayon en cellule.
    center : (int,int)
        index matrice du centre.
    size_matrix : (int,int)
        taille de la matrice en cellule
    Returns
    -------
    matrice avec borne et liste des facette.
    """
    matrix_geo = np.zeros((size_matirix[0],size_matirix[1]))
    matrix_geo[:] = np.nan
    angle = 360/n_facette/180*np.pi
    r1 = np.array(center)+np.array((rayon,0))
    all_facette = []
    current_angle = angle
    
    #création des facettes
    for i in range(n_facette + 1):
        r2 = center + np.array((rayon*np.cos(current_angle), rayon*np.sin(current_angle)))
        facette = Facette(r1, r2)
        all_facette.append(facette)
        
        # #mise de la facette dans la géométrie
        # index = i
        # rho = r2-r1
        # maxx = np.max(np.abs(rho))
        # s = np.linspace(0, 1,maxx+1)
        # for i in range(len(s)):
        #     r = r1 + s[i]*rho
        #     if (current_angle<=90/180*np.pi):
        #         matrix_geo[int(ceil(r[0])),int(r[1])] = index
        #     elif (current_angle<=180/180*np.pi):
        #         matrix_geo[int((r[0])),int(r[1])] = index
        #     elif (current_angle<=270/180*np.pi):
        #         matrix_geo[int((r[0])),ceil(r[1])] = index
        #     else:
        #         matrix_geo[ceil((r[0])),ceil(r[1])] = index
            
        #passe à la facette suivante
        r1 = r2
        current_angle = current_angle + angle
        

    for i in range(size_matirix[0]):
        a = False
        b = False
        c = False
        start = -1
        end = -1
        for j in range(size_matirix[1]):
            if(c):
                matrix_geo[i][start:end] = -1
                break
            if(not np.isnan(matrix_geo[i][j]) and not a):
                a = True
            elif(not b and np.isnan(matrix_geo[i][j]) and a):
                b =True
                start = j
            elif(not np.isnan(matrix_geo[i][j]) and b):
                end = j
                c=True
        
                
            
    return(matrix_geo,all_facette)
    
        
_, f = cercle_maker(30, 10, (25,25), (50,50))

for facette in f:

    plt.plot([facette.r1[0], facette.r2[0]], [facette.r1[1], facette.r2[1]], "o-")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis('equal')

plt.grid(True)
plt.show()
    