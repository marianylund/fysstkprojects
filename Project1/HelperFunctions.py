import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
from numpy.random import randint
from scipy.stats import norm
from imageio import imread
import seaborn as sns
import os, sys
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

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

def image_path(fig_id, FIGURE_ID = "Results/FigureFiles"):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id, DATA_ID = "DataFiles/"):
    return os.path.join(DATA_ID, dat_id)

def save_figure(fig_id, FIGURE_ID = "Results/FigureFiles"):
    plt.savefig(image_path(fig_id, FIGURE_ID) + ".png", format='png', bbox_inches = 'tight')

def FrankeFunction(x,y, noise_strength = 0.0):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4 + np.random.normal(0, 1, size=x.shape[0]) * noise_strength

def create_frankie_data(seed = 3155, N = 20, noise_strength = 0.1):
    np.random.seed(seed)

    x, y = create_mesh(N, random_mesh = True, seed = seed)
    z_franke = FrankeFunction(x, y, noise_strength)
    z = np.ravel(z_franke)
    return x, y, z

def create_terrain_data(N = 1000, path = 'DataFiles/SRTM_data_Norway_2.tif'):
    terrain = imread(path)
    terrain = terrain[:N,:N]
    # Creates mesh of image pixels
    x = np.linspace(0,1, np.shape(terrain)[0])
    y = np.linspace(0,1, np.shape(terrain)[1])
    x_mesh, y_mesh = np.meshgrid(x,y)

    z = terrain
    return x_mesh, y_mesh, z


def plot_3d_graph(x, y, z, title, z_title = "Z", dpi = 150, formatter = '%.02f', z_line_ticks = 10, view_azim = -35, set_limit = True, save_fig = False):
    fig = plt.figure(dpi=dpi)
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, 
                        linewidth=0, antialiased=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(z_title)
    # Customize the z axis.
    if set_limit:
        ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(z_line_ticks))
    ax.zaxis.set_major_formatter(FormatStrFormatter(formatter))
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.view_init(elev=40., azim=view_azim)
    plt.subplots_adjust(top = 1.4, right=1.4)
    plt.title(title, loc='left', fontsize=22, fontweight=1)
    if save_fig:
        save_figure(title)
    plt.show()

def confidence_interval(X, z, beta, noise_strength, N, percentile = 1.95, title = "Confidence Intervals of beta", save_fig = False):
    sns.set_style("whitegrid")
    cov = np.var(z)*np.linalg.pinv(X.T @ X)
    std_beta = np.sqrt(np.diag(cov))
    CI = percentile*std_beta

    plt.errorbar(range(len(beta)), beta, CI, fmt='.k', elinewidth=1, capsize=3, label=r'$\beta_j \pm ' + str(percentile) + ' \sigma$')
    plt.legend()
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.figtext(0.1, -0.1, "Noise Strength: " + str(noise_strength) + "\nNumber of samples: " + str(N), ha="left", fontsize=7)
    if save_fig:
        save_figure(title + str(N) + str(noise_strength))
    plt.show()

def plot_test_train_model_complexity(polydegree, values_to_plot, N = -1, trials = -1, sample_count = -1, noise_strength = 0.1, save_fig = False):
    plt.style.use('seaborn-darkgrid')
    plt.plot(polydegree, values_to_plot["Train"], label="Train sample")
    plt.plot(polydegree, values_to_plot["Test"], label="Test sample")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Prediction Error")
    plt.legend()
    if(N != -1 and trials != -1 and sample_count != -1):
        plt.figtext(0.1, -0.1, "Noise Strength: " + str(noise_strength) + "\nNumber of samples: " + str(N) + "\nTrials: " + str(trials) + "\nBoostrap samples: " + str(sample_count), ha="left", fontsize=7)
    if save_fig:
        save_figure("TestTrainErrorAsModelComplexity" + str(N) + str(trials) + str(sample_count) + str(noise_strength).replace(".", ""))
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


def bootstrap(data, statistic, R, analysis = False):
    t = np.zeros(R); n = len(data); inds = np.arange(n); t0 = time()
    # non-parametric bootstrap         
    for i in range(R):
        t[i] = statistic(data[randint(0,n,n)])

    if analysis:   
        print("Runtime: %g sec" % (time()-t0)); print("Bootstrap Statistics :")
        print("original           bias      std. error")
        print("%8g %8g %14g %15g" % (statistic(data), np.std(data), np.mean(t), np.std(t)))
    return t

def test_bootstrap(X):
    stat = lambda data: np.mean(data)
    t = bootstrap(X, stat, 100)
    # the histogram of the bootstrapped  data                                                                                                    
    n, binsboot, patches = plt.hist(t, 50, normed=1, facecolor='red', alpha=0.75)

    # add a 'best fit' line  
    y = norm.pdf( binsboot, np.mean(t), np.std(t))
    lt = plt.plot(binsboot, y, 'r--', linewidth=1)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    #plt.axis([99.5, 100.6, 0, 3.0])
    plt.grid(True)

    plt.show()