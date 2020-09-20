import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Credit to Jon Dahl and Michael
from numba import jit
@jit
def create_X(x, y, n, debug = False):
    if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)          # Number of elements in beta                                                               
    X = np.ones((N,l))

    for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                    X[:,q+k] = (x**(i-k))*(y**k)
    if debug:
        print("X.shape: ", X.shape)
    return X

def FrankeFunction(x,y, noise_strength = 0.0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, 1, size=x.shape[0]) * noise_strength

def plot_3d_franke(x, y, z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def create_mesh(n, random_mesh = False, seed = None):
    if random_mesh:
        if seed is not None:
            np.random.seed(seed)
        x = np.random.rand(n)
        y = np.random.rand(n)
    else:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
    return np.meshgrid(x, y)

def plot_test_train(polydegree, values_to_plot, debug = False):
    for val in values_to_plot:
        if debug:
            print(val, values_to_plot[val])
        plt.plot(polydegree, values_to_plot[val], label=val)
    plt.legend()
    plt.show()