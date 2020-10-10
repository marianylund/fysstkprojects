import numpy as np
from random import random, seed
from imageio import imread
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from RegLib.HelperFunctions import plot_3d_graph
from PROJECT_SETUP import TERRAIN_PATH, SAVE_FIG

# f): Introducing real data and preparing the data analysis 

N = 1000
terrain = imread(TERRAIN_PATH)
terrain = terrain[:N,:N]

# Creates mesh of image pixels
x = np.linspace(0,1, np.shape(terrain)[0])
y = np.linspace(0,1, np.shape(terrain)[1])
x_mesh, y_mesh = np.meshgrid(x,y)
z = terrain
smoothed_z = savgol_filter(terrain, 77, 2)

# Show the terrain
plt.figure()
plt.title('Terrain over Norway')
plt.imshow(z, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

plt.figure()
plt.title('Terrain over Norway (Smoothed)')
plt.imshow(smoothed_z, cmap='gray')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

dpi = 100
if SAVE_FIG:
    dpi = 150

plot_3d_graph(x_mesh, y_mesh, z, "Norway Terrain Data", "Elevation of terrain", dpi=dpi, formatter='%.f',z_line_ticks=6, view_azim=-50, set_limit=False, save_fig=SAVE_FIG)


plot_3d_graph(x_mesh, y_mesh, smoothed_z, "Norway Terrain Data Smoothed (Savgol filter)", "Elevation of terrain", dpi=dpi, formatter='%.f',z_line_ticks=6, view_azim=-50, set_limit=False, save_fig=SAVE_FIG)