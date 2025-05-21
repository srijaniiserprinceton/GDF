import numpy as np
import os, sys, importlib
import matplotlib.pyplot as plt; plt.ion()
from gdf.src import hybrid_ubulk

# loading the config file to obtain the desired gvdf dictionary
current_file_dir = os.path.dirname(__file__)
config_file = 'init_plot_setup'

# importing the config file
config = importlib.import_module(config_file, package=current_file_dir)
gvdf_tstamp = hybrid_ubulk.run(config)
