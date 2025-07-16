import sys, importlib

from gdf.src import VDF_rec_PSP
from gdf.src import misc_funcs as misc_fn

# importing the config file provided at command line
config_file = sys.argv[1]
config = importlib.import_module(config_file).config
gvdf_tstamp = VDF_rec_PSP.run(config)
