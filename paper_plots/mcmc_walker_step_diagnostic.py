import numpy as np
import matplotlib.mlab as mlab
import matplotlib.patches as patches
import matplotlib.pyplot as plt; plt.ion()
from scipy.stats import norm
import os, pickle

def read_pickle(fname):
    with open(f'{fname}.pkl', 'rb') as handle:
        x = pickle.load(handle)
    return x

def fit_gaussian(hist_data, ax):
    (mu, sigma) = norm.fit(hist_data)
    x = np.linspace(-10, 10, 100)
    y = norm.pdf(x, mu, sigma)
    ax.plot(x, y, 'k--', linewidth=2)
    ax.text(0.95, 0.95, r"$\sigma = $" + f"{sigma:.3f}", transform=ax.transAxes, ha='right', va='top')

    return sigma

if __name__ == "__main__":
    walker_arr = np.array([4, 6, 8])
    iterations_arr = np.array([200, 400, 600, 800, 1000, 1500, 2000])

    # compute times
    comp_times = np.array([[21.61, 17.05, 12.31, 10.35, 8.19, 6.25, 4.34],
                           [30.38, 24.07, 16.73, 14.20, 10.88, 8.44, 5.55],
                           [43.98, 28.51, 20.2, 17.03, 16.0, 10.13, 6.39]])
    
    comp_times = np.flip(comp_times, axis=1)

    # creating array to store the sigma for each case
    sigma_vy_arr = np.zeros_like(comp_times)
    sigma_vz_arr = np.zeros_like(comp_times)

    # loading the reference data dictionary
    current_file_directory = os.path.dirname(__file__)
    data_dir = f'{current_file_directory}/mcmc_walker_step'
    time_str = '20200126_071201_073054'
    ref_dict = read_pickle(f'{data_dir}/scipy_vdf_rec_data_8_2000_{time_str}')
    Ntimes = len(ref_dict.keys())

    u_corr_ref = np.asarray([ref_dict[i]['u_corr'] for i in range(Ntimes)])
    time_arr = np.asarray([ref_dict[i]['time'] for i in range(Ntimes)])

    fig, ax = plt.subplots(len(iterations_arr), 6, figsize=(10,10), sharex=True, sharey=True)

    for walker_idx, walker in enumerate(walker_arr):
        for iteration_idx, iteration in enumerate(iterations_arr):
            data_dict = read_pickle(f'{data_dir}/scipy_vdf_rec_data_{walker}_{iteration}_{time_str}')

            u_corr = np.asarray([data_dict[i]['u_corr'] for i in range(Ntimes)])

            ax[iteration_idx,2*walker_idx].hist(u_corr[:,1] - u_corr_ref[:,1], bins=15, density=True,
                                     range=(-10,10), color='blue', alpha=0.6)
            ax[iteration_idx,2*walker_idx+1].hist(u_corr[:,2] - u_corr_ref[:,2], bins=15, density=True,
                                     range=(-10,10), color='gray', alpha=0.6)
            sigma_vy = fit_gaussian(u_corr[:,1] - u_corr_ref[:,1], ax[iteration_idx,2*walker_idx])
            sigma_vz = fit_gaussian(u_corr[:,2] - u_corr_ref[:,2], ax[iteration_idx,2*walker_idx+1])
            ax[iteration_idx,2*walker_idx].set_ylim([0, 0.3])
            ax[iteration_idx,2*walker_idx+1].set_ylim([0, 0.3])
            ax[iteration_idx,0].set_ylabel(f'{iteration}', fontweight='bold', fontsize=14)

            # storing the sigma values in the arrays
            sigma_vy_arr[walker_idx, iteration_idx] = sigma_vy
            sigma_vz_arr[walker_idx, iteration_idx] = sigma_vz

    # Draw a horizontal lines at those coordinates
    arrow_len = 0.08

    p1a = patches.FancyArrowPatch((0.07, 0.97), (0.07+arrow_len*0.8,0.97), arrowstyle='<|-', mutation_scale=20, transform=fig.transFigure, color="black")
    text1 = plt.text(s='nwalkers = 4', x=0.165, y=0.965, transform=fig.transFigure, fontweight='bold')
    p1b = patches.FancyArrowPatch((0.36-arrow_len*0.85,0.97), (0.36,0.97), arrowstyle='-|>', mutation_scale=20, transform=fig.transFigure, color="black")
    fig.add_artist(p1a)
    fig.add_artist(text1)
    fig.add_artist(p1b)

    p1a = patches.FancyArrowPatch((0.38, 0.97), (0.38+arrow_len*0.8,0.97), arrowstyle='<|-', mutation_scale=20, transform=fig.transFigure, color="black")
    text1 = plt.text(s='nwalkers = 6', x=0.47, y=0.965, transform=fig.transFigure, fontweight='bold')
    p1b = patches.FancyArrowPatch((0.67-arrow_len*0.85,0.97), (0.67,0.97), arrowstyle='-|>', mutation_scale=20, transform=fig.transFigure, color="black")
    fig.add_artist(p1a)
    fig.add_artist(text1)
    fig.add_artist(p1b)

    p1a = patches.FancyArrowPatch((0.69, 0.97), (0.69+arrow_len*0.8,0.97), arrowstyle='<|-', mutation_scale=20, transform=fig.transFigure, color="black")
    text1 = plt.text(s='nwalkers = 8', x=0.78, y=0.965, transform=fig.transFigure, fontweight='bold')
    p1b = patches.FancyArrowPatch((0.98-arrow_len*0.85,0.97), (0.98,0.97), arrowstyle='-|>', mutation_scale=20, transform=fig.transFigure, color="black")
    fig.add_artist(p1a)
    fig.add_artist(text1)
    fig.add_artist(p1b)

    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.04, top=0.95)

    plt.savefig('MCMC_convergence_test.pdf')

    #--------- Making the convergence plot for Vy ---------------#
    plt.figure(figsize=(8,8))
    # plotting the sigma against computational time
    color_arr = np.array(['blue', 'red', 'black'])
    ls_arr = np.array(['solid', 'dashed', 'dotted'])
    marker_arr = np.array(['o', 'v', '^', '*', 'x', '+', 's'])

    for i in range(len(walker_arr)):
        plt.plot(sigma_vy_arr[i], comp_times[-1,-1] / comp_times[i], ls=ls_arr[i], color=color_arr[i], label=f'Walkers = {walker_arr[i]}')
        for j in range(len(iterations_arr)):
            plt.scatter(sigma_vy_arr[i,j], comp_times[-1,-1] / comp_times[i,j], marker=marker_arr[j], color=color_arr[i], label=iterations_arr[j])
    
    plt.grid(True)
    plt.legend()
    plt.xlabel(r'Spread in convergence of $V_y$', fontweight='bold', fontsize=14)
    plt.ylabel(r'Speedup w.r.t 8 walkers 2000 iterations', fontweight='bold', fontsize=14)

    #--------- Making the convergence plot for Vz ---------------#
    plt.figure(figsize=(8,8))
    # plotting the sigma against computational time
    color_arr = np.array(['blue', 'red', 'black'])
    ls_arr = np.array(['solid', 'dashed', 'dotted'])
    marker_arr = np.array(['o', 'v', '^', '*', 'x', '+', 's'])

    for i in range(len(walker_arr)):
        plt.plot(sigma_vz_arr[i], comp_times[-1,-1] / comp_times[i], ls=ls_arr[i], color=color_arr[i], label=f'Walkers = {walker_arr[i]}')
        for j in range(len(iterations_arr)):
            plt.scatter(sigma_vz_arr[i,j], comp_times[-1,-1] / comp_times[i,j], marker=marker_arr[j], color=color_arr[i], label=iterations_arr[j])
    
    plt.grid(True)
    plt.legend()
    plt.xlabel(r'Spread in convergence of $V_z$', fontweight='bold', fontsize=14)
    plt.ylabel(r'Speedup w.r.t 8 walkers 2000 iterations', fontweight='bold', fontsize=14)




            