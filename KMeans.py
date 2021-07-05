# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 16:46:52 2021

@author: Γιάννης
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import KMeans

n_samples = 1500
random_state = 170

np.random.seed(0)
#make and plot circles 
noisy_circles_x,noisy_circles_y = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
plt.scatter(noisy_circles_x[:,0], noisy_circles_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

km = KMeans(n_clusters = 2, init = 'random',
            n_init = 10, max_iter = 300, tol = 1e-04, random_state = random_state)
y_km = km.fit_predict(noisy_circles_x)
plt.scatter(noisy_circles_x[y_km == 0,0], noisy_circles_x[y_km == 0,1], s = 50, c = 'green', marker = 'o', edgecolor = 'green', label = 'cluster1')
plt.scatter(noisy_circles_x[y_km == 1,0], noisy_circles_x[y_km == 1,1], s = 50, c = 'red', marker = 'd', edgecolor = 'red', label = 'cluster2')
plt.show()


#make and plot moons
noisy_moons_x,noisy_moons_y = datasets.make_moons(n_samples=n_samples, noise=.05)
plt.scatter(noisy_moons_x[:,0], noisy_moons_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

km = KMeans(n_clusters = 2, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = random_state)
y_km = km.fit_predict(noisy_moons_x)
plt.scatter(noisy_moons_x[y_km == 0,0], noisy_moons_x[y_km == 0,1], s = 50, c = 'green', marker = 'o', edgecolor = 'green', label = 'cluster1')
plt.scatter(noisy_moons_x[y_km == 1,0], noisy_moons_x[y_km == 1,1], s = 50, c = 'red', marker = 'd', edgecolor = 'red', label = 'cluster2')
plt.show()

#make and plot blobs
blobs_x,blobs_y = datasets.make_blobs(n_samples=n_samples, random_state=8)
plt.scatter(blobs_x[:,0], blobs_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

km = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = 8)
y_km = km.fit_predict(blobs_x)
plt.scatter(blobs_x[y_km == 0,0], blobs_x[y_km == 0,1], s = 50, c = 'green', marker = 'o', edgecolor = 'green', label = 'cluster1')
plt.scatter(blobs_x[y_km == 1,0], blobs_x[y_km == 1,1], s = 50, c = 'red', marker = 'd', edgecolor = 'red', label = 'cluster2')
plt.scatter(blobs_x[y_km == 2,0], blobs_x[y_km == 2,1], s = 50, c = 'blue', marker = '*', edgecolor = 'blue', label = 'cluster3')
plt.show()

#make and plot no structure 
no_structure_x,no_structure_y = np.random.rand(n_samples, 2), None
plt.scatter(no_structure_x[:,0], no_structure_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

km = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = random_state)
y_km = km.fit_predict(no_structure_x)
plt.scatter(no_structure_x[y_km == 0,0], no_structure_x[y_km == 0,1], s = 50, c = 'green', marker = 'o', edgecolor = 'green', label = 'cluster1')
plt.scatter(no_structure_x[y_km == 1,0], no_structure_x[y_km == 1,1], s = 50, c = 'red', marker = 'd', edgecolor = 'red', label = 'cluster2')
plt.scatter(no_structure_x[y_km == 2,0], no_structure_x[y_km == 2,1], s = 50, c = 'blue', marker = '*', edgecolor = 'blue', label = 'cluster3')
plt.show()

# blobs with varied variances
varied_x,varied_y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)
plt.scatter(varied_x[:,0], varied_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

km = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = random_state)
y_km = km.fit_predict(varied_x)
plt.scatter(varied_x[y_km == 0,0], varied_x[y_km == 0,1], s = 50, c = 'green', marker = 'o', edgecolor = 'green', label = 'cluster1')
plt.scatter(varied_x[y_km == 1,0], varied_x[y_km == 1,1], s = 50, c = 'red', marker = 'd', edgecolor = 'red', label = 'cluster2')
plt.scatter(varied_x[y_km == 2,0], varied_x[y_km == 2,1], s = 50, c = 'blue', marker = '*', edgecolor = 'blue', label = 'cluster3')
plt.show()

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples,random_state = random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

km = KMeans(n_clusters = 3, init = 'random', n_init = 10, max_iter = 300, tol = 1e-04, random_state = random_state)
y_km = km.fit_predict(X_aniso)
plt.scatter(X_aniso[y_km == 0,0], X_aniso[y_km == 0,1], s = 50, c = 'green', marker = 'o', edgecolor = 'green', label = 'cluster1')
plt.scatter(X_aniso[y_km == 1,0], X_aniso[y_km == 1,1], s = 50, c = 'red', marker = 'd', edgecolor = 'red', label = 'cluster2')
plt.scatter(X_aniso[y_km == 2,0], X_aniso[y_km == 2,1], s = 50, c = 'blue', marker = '*', edgecolor = 'blue', label = 'cluster3')
plt.show()


