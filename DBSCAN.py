# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 18:35:52 2021

@author: Γιάννης
"""

import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
from sklearn.cluster import DBSCAN

n_samples = 1500
random_state = 170

np.random.seed(0)
#make and plot circles 
noisy_circles_x,noisy_circles_y = datasets.make_circles(n_samples=n_samples, factor=.5,noise=.05)
plt.scatter(noisy_circles_x[:,0], noisy_circles_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

xx = DBSCAN(eps = 0.1, min_samples = 9, metric = 'euclidean').fit(noisy_circles_x)
plt.figure(figsize = (10,7))
plt.scatter(noisy_circles_x[xx.labels_ == 0,0], noisy_circles_x[xx.labels_ == 0,1], s = 50, c = 'red')
plt.scatter(noisy_circles_x[xx.labels_ == 1,0], noisy_circles_x[xx.labels_ == 1,1], s = 50, c = 'green')
plt.show()

#make and plot moons
noisy_moons_x,noisy_moons_y = datasets.make_moons(n_samples=n_samples, noise=.05)
plt.scatter(noisy_moons_x[:,0], noisy_moons_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

xx = DBSCAN(eps = 0.1, min_samples = 9, metric = 'euclidean').fit(noisy_moons_x)
plt.figure(figsize = (10,7))
plt.scatter(noisy_moons_x[xx.labels_ == 0,0], noisy_moons_x[xx.labels_ == 0,1], s = 50, c = 'red')
plt.scatter(noisy_moons_x[xx.labels_ == 1,0], noisy_moons_x[xx.labels_ == 1,1], s = 50, c = 'green')
plt.show()

#make and plot blobs
blobs_x,blobs_y = datasets.make_blobs(n_samples=n_samples, random_state=8)
plt.scatter(blobs_x[:,0], blobs_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

xx = DBSCAN(eps = 0.45, min_samples = 9, metric = 'euclidean').fit(blobs_x)
plt.figure(figsize = (10,7))
plt.scatter(blobs_x[xx.labels_ == 0,0], blobs_x[xx.labels_ == 0,1], s = 50, c = 'red')
plt.scatter(blobs_x[xx.labels_ == 1,0], blobs_x[xx.labels_ == 1,1], s = 50, c = 'green')
plt.scatter(blobs_x[xx.labels_ == 2,0], blobs_x[xx.labels_ == 2,1], s = 50, c = 'blue')
plt.show()

#make and plot no structure 
no_structure_x,no_structure_y = np.random.rand(n_samples, 2), None
plt.scatter(no_structure_x[:,0], no_structure_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

xx = DBSCAN(eps = 0.45, min_samples = 9, metric = 'euclidean').fit(no_structure_x)
plt.figure(figsize = (10,7))
plt.scatter(no_structure_x[xx.labels_ == 0,0], no_structure_x[xx.labels_ == 0,1], s = 50, c = 'red')
plt.scatter(no_structure_x[xx.labels_ == 1,0], no_structure_x[xx.labels_ == 1,1], s = 50, c = 'green')
plt.scatter(no_structure_x[xx.labels_ == 2,0], no_structure_x[xx.labels_ == 2,1], s = 50, c = 'blue')
plt.show()

# blobs with varied variances
varied_x,varied_y = datasets.make_blobs(n_samples=n_samples,cluster_std=[1.0, 2.5, 0.5],random_state=random_state)
plt.scatter(varied_x[:,0], varied_x[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

xx = DBSCAN(eps = 0.8, min_samples = 9, metric = 'euclidean').fit(varied_x)
plt.figure(figsize = (10,7))
plt.scatter(varied_x[xx.labels_ == 0,0], varied_x[xx.labels_ == 0,1], s = 50, c = 'red')
plt.scatter(varied_x[xx.labels_ == 1,0], varied_x[xx.labels_ == 1,1], s = 50, c = 'green')
plt.scatter(varied_x[xx.labels_ == 2,0], varied_x[xx.labels_ == 2,1], s = 50, c = 'blue')
plt.show()

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples,random_state = random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
plt.scatter(X_aniso[:,0], X_aniso[:,1], c = 'white', marker = 'o', edgecolor = 'black', s = 50)
plt.show()

xx = DBSCAN(eps = 0.3, min_samples = 9, metric = 'euclidean').fit(X_aniso)
plt.figure(figsize = (10,7))
plt.scatter(X_aniso[xx.labels_ == 0,0], X_aniso[xx.labels_ == 0,1], s = 50, c = 'red')
plt.scatter(X_aniso[xx.labels_ == 1,0], X_aniso[xx.labels_ == 1,1], s = 50, c = 'green')
plt.scatter(X_aniso[xx.labels_ == 2,0], X_aniso[xx.labels_ == 2,1], s = 50, c = 'blue')
plt.show()