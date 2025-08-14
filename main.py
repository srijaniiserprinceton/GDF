import sys, importlib
import numpy as np

from gdf.src_GL import VDF_rec_PSP
from gdf.src_GL import misc_funcs as misc_fn

# importing the config file provided at command line
config_file = sys.argv[1]
config = importlib.import_module(config_file).config
gvdf_tstamp = VDF_rec_PSP.run(config)
