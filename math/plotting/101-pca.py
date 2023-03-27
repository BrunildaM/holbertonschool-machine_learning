#!/usr/bin/env python3
""" plot the given PCA data as a 3D scatter plot """

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure(figsize=(12, 8), dpi=80)
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
