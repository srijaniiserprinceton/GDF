import sys, importlib
from gdf.src import hybrid_ubulk

# config file
config_file = sys.argv[1]

# importing the config file
config = importlib.import_module(config_file)
gvdf_tstamp = hybrid_ubulk.run(config)