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
from scipy.signal import savgol_filter
from RegLib.load_save_data import load_best_checkpoint

warnings.filterwarnings("ignore", message="Numba")

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

def get_best_dict(output_dir):
    best_data_dict = load_best_checkpoint(output_dir)
    assert best_data_dict != None, "No best data dictionary was found in: " + str(output_dir)
    m, s = best_data_dict["Proccess_time"]
    print(f'Best model. Step: {best_data_dict["Step"]}, eval: {best_data_dict["Test_eval"]:.2f}. Time: {m:.0f}:{s:.0f}')
    return best_data_dict

def image_path(fig_id, FIGURE_ID = "Results/FigureFiles"):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id, DATA_ID = "DataFiles/"):
    return os.path.join(DATA_ID, dat_id)

def save_figure(fig_id, FIGURE_ID = "Results/FigureFiles"):
    new_fid_id = fig_id.replace(",", "").replace(" ", "").replace("=", "")
    plt.savefig(image_path(new_fid_id, FIGURE_ID) + ".png", format='png', bbox_inches = 'tight')

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

    terrain = savgol_filter(terrain, 77, 2)
    z = terrain.ravel()

    return x_mesh, y_mesh, z

def plot_3d_graph(x, y, z, title, z_title = "Z", dpi = 150, formatter = '%.02f', z_line_ticks = 10, view_azim = -35, set_limit = True, save_fig = False):
    fig = plt.figure(dpi=dpi)
    ax = fig.gca(projection='3d')
    # Plot the surface.
    surf = ax.plot_surface(x, y, z, 
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
    else: 
        plt.show()

    plt.cla()

def confidence_interval(X, z, beta, noise_strength, N, info_to_add = {}, percentile = 1.95, title = "Confidence Intervals of beta", save_fig = False):
    sns.set_style("whitegrid")
    cov = np.var(z)*np.linalg.pinv(X.T @ X)
    std_beta = np.sqrt(np.diag(cov))
    CI = percentile*std_beta

    plt.errorbar(range(len(beta)), beta, CI, fmt='.k', elinewidth=1, capsize=3, label=r'$\beta_j \pm ' + str(percentile) + ' \sigma$')
    plt.legend()
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    info_str, title_info = parse_info_for_plot(info_to_add)
    
    if info_str != "":
        plt.figtext(0.1, -0.1, info_str, ha="left", fontsize=7)

    if save_fig:
        save_figure(title + title_info)
    else:
        plt.show()
        print(info_str)
    plt.cla()

def parse_info_for_plot(info_to_add):
    info_str = ""
    title_info = ""
    for i in info_to_add:
        info_str += (i + str(info_to_add[i]) + "\n")
        title_info += str(info_to_add[i]).replace(".", "")
    return info_str, title_info

def plot_values_with_info(polydegree, values_to_plot, title = "TestTrainErrorAsModelComplexity", xlabel = "Polynomial Degree", ylabel = "Prediction Error", info_to_add = {}, xscale = "linear", save_fig = False, scatter = False):
    plt.style.use('seaborn-darkgrid')
    scatter_label = 'o' if scatter else ''
    for val in values_to_plot:
        plt.plot(polydegree, values_to_plot[val], scatter_label, label=val)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    info_str, title_info = parse_info_for_plot(info_to_add)
    
    if info_str != "":
        plt.figtext(0.1, -0.1, info_str, ha="left", fontsize=7)

    plt.xscale(xscale)

    if save_fig:
        save_figure(title + title_info)
    else:
        plt.show()
        print(info_str)

    plt.cla()

def plot_values_with_two_y_axis(steps, values_to_plot_y1, values_to_plot_y2, title = "SGD", xlabel = "Step", y1_label = "Prediction Error", y2_label = "Learning Rate", info_to_add = {}, xscale = "linear", ylimit = None, save_fig = False, scatter = False):
    plt.style.use('seaborn-darkgrid')
    fig, ax1 = plt.subplots()

    scatter_label = 'o' if scatter else ''

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1_label)
    for val in values_to_plot_y1:
        ax1.plot(steps, values_to_plot_y1[val], scatter_label, label=val)

    if ylimit != None:
        ax1.set_ylim(ylimit[0], ylimit[1])

    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel(y2_label, color=color)
    ax2.set_yscale("log")
    for val in values_to_plot_y2:
        ax2.plot(steps, values_to_plot_y2[val], scatter_label, label=val, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.legend()

    info_str, title_info = parse_info_for_plot(info_to_add)
    
    plt.title(title, loc='left', fontsize=12, fontweight=0)

    if info_str != "":
        plt.figtext(0.1, -0.1, info_str, ha="left", fontsize=7)

    plt.xscale(xscale)
    plt.tight_layout() 
    if save_fig:
        save_figure(title) #+ title_info)
        plt.cla()
    else:
        plt.show()
        print(info_str)

def plot_bias_variance_analysis(polydegree, values_to_plot, title = "BiasVarTradeoff",  xlabel = "Polynomial Degree", ylabel = "Prediction Error", info_to_add = {}, xscale = "linear",  save_fig = False):
    plt.style.use('seaborn-darkgrid')
    y1 = values_to_plot["Variance"]
    y2 = values_to_plot["Bias^2"]
    labels = ["Variance", "Bias^2"]
    fig, ax = plt.subplots()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.stackplot(polydegree, y1, y2, labels=labels)
    ax.plot(polydegree, values_to_plot["MSE"], color="black", label = "MSE")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xscale(xscale)
    
    info_str, title_info = parse_info_for_plot(info_to_add)
    if info_str != "":
        plt.figtext(0.1, -0.1, info_str, ha="left", fontsize=7)


    if save_fig:
        save_figure(title + title_info)
    else:
        plt.show()
        print(info_str)

    plt.cla()

def mupltiple_line_plot(polydegree, values_to_plot, plot_labels, subtitle, info_to_add = {}, xlabel = "Polynomial degree", ylabel = "Prediction Error", ylim = [0, 100], xscale = "linear", save_fig = False):
    # Initialize the figure
    plt.style.use('seaborn-darkgrid')
    # create a color palette
    palette = plt.get_cmap('Set1')

    num_of_plots = len(values_to_plot)
    assert len(plot_labels) == num_of_plots, "plot_labels != values_to_plot"

    for plot_num in range(0, num_of_plots):
        # Find the right spot on the plot
        plt.subplot(np.ceil(num_of_plots/2), 2, plot_num+1)
        
        for val in values_to_plot[plot_num]:
            # Plot the lineplot
            plt.plot(polydegree, values_to_plot[plot_num][val], label=val)

        if(plot_num == 1):
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.ylim(ylim[0], ylim[1])
        ax = plt.gca()
        # Not ticks everywhere
        if plot_num in range(num_of_plots) :
            ax.axes.xaxis.set_ticklabels([])
        if plot_num not in range(0, num_of_plots, 2):
            ax.axes.yaxis.set_ticklabels([])
        plt.xscale(xscale)

        # Add title
        plt.title(plot_labels[plot_num], loc='left', fontsize=12, fontweight=0, color=palette(plot_num) )
    
    # general title
    plt.suptitle(subtitle, fontsize=14, fontweight=0, color='black', style='italic', y=1.02)
    plt.figtext(0.5, 0.02, xlabel, ha='center', va='center')
    plt.figtext(0.02, 0.5, ylabel, ha='center', va='center', rotation='vertical')
    plt.tight_layout()

    info_str, title_info = parse_info_for_plot(info_to_add)
    if info_str != "":
        plt.figtext(0.1, -0.1, info_str, ha="left", fontsize=7)

    if save_fig:
        save_figure(subtitle + title_info)
    else:
        plt.show()
        print(info_str)

    plt.cla()

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

def progressBar(current, total, msg = "", barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))
    progress_msg = 'Progress: [%s%s] %d %% ' % (arrow, spaces, percent) + msg
    print(progress_msg, end='\r')