import numpy as np
import os, sys, importlib
import matplotlib.pyplot as plt; plt.ion()
from gdf.src import hybrid_ubulk

# loading the config file to obtain the desired gvdf dictionary
current_file_dir = os.path.dirname(__file__)
# expects that the init file will be in the local directory of this script
config_file = 'init_plot_setup'

# importing the config file
config = importlib.import_module(config_file, package=current_file_dir)
gvdf_tstamp = hybrid_ubulk.run(config)

# plotting the Bsplines
sortidx = np.argsort(gvdf_tstamp.super_rfac)
rgrid = gvdf_tstamp.super_rfac[sortidx]
minidx = np.argmin(np.abs(rgrid - gvdf_tstamp.rfac_nonan.min()))
maxidx = np.argmin(np.abs(rgrid - gvdf_tstamp.rfac_nonan.max()))
rgrid = rgrid[minidx:maxidx]
B_i_n = gvdf_tstamp.super_B_i_n[:,sortidx][:,minidx:maxidx]

plt.figure()
for i in range(len(B_i_n)):
    plt.plot(rgrid, B_i_n[i])
plt.grid(True)
plt.axvline(gvdf_tstamp.rfac_nonan.min(), ls='--', color='k')
plt.axvline(gvdf_tstamp.rfac_nonan.max(), ls='--', color='k')
