import numpy as np
import matplotlib.pyplot as plt; plt.ion()

def color_gyrogrids(gvdf_tstamp):
    # extracting the grids
    vpara = gvdf_tstamp.vpara_nonan
    vperp = gvdf_tstamp.vperp_nonan

    # normalizing the colormap
    colorvals = gvdf_tstamp.G_k_n / gvdf_tstamp.G_k_n.max()

    plt.figure()
    # coloring them by cycling over the bases
    for i in range(gvdf_tstamp.G_k_n.shape[0]):
        plt.scatter(gvdf_tstamp.vperp_nonan, gvdf_tstamp.vpara_nonan, c=colorvals[i,:], cmap='seismic', vmin=-0.4, vmax=0.4)
        plt.scatter(-gvdf_tstamp.vperp_nonan, gvdf_tstamp.vpara_nonan, c=colorvals[i,:], cmap='seismic', vmin=-0.4, vmax=0.4)
        plt.savefig(f'Figures/presentation_plots/basis_{i}.png')
        plt.xlabel(r'$v_{\perp}$ [km/s]', fontsize=16)
        plt.ylabel(r'$v_{||}$ [km/s]', fontsize=16)

def plot_changing_grids(gvdf_tstamp, tidx):
    # getting the unit magnetic field vector
    b_hat = gvdf_tstamp.b_span[tidx] / np.linalg.norm(gvdf_tstamp.b_span[tidx])

    # plotting the vector in 3D

    # Create a figure with 1 row, 2 columns
    fig, (ax3d, ax2d) = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={ 'projection': '3d' })

    # Plot the 3D arrow
    ax3d.quiver(0, 0, 0, b_hat[0], b_hat[1], b_hat[2], length=1.0, normalize=True, color='r')
    ax3d.set_xlim([-1, 1])
    ax3d.set_ylim([-1, 1])
    ax3d.set_zlim([-1, 1])
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title("3D Unit Vector")


    # Plot the 2D data
    ax2d = fig.add_subplot(1, 2, 2)  # Redefine as 2D axes
    ax2d.scatter(gvdf_tstamp.vperp_nonan, gvdf_tstamp.vpara_nonan, color='k', s=4)
    ax2d.scatter(-gvdf_tstamp.vperp_nonan, gvdf_tstamp.vpara_nonan, color='k', s=4)
    ax2d.set_xlabel(r'$v_{\perp}$ [km/s]', fontsize=16)
    ax2d.set_ylabel(r'$v_{||}$ [km/s]', fontsize=16)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.savefig(f'Figures/presentation_plots/b_hat3D_images/b_hat_{tidx}.png')
    plt.close()


