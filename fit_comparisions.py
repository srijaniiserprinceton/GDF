# This file is dedicated to generating figures that compare the density and velocity with:
#
# 1. SBR method
# 2. BiMax Fit
# 3. SPAN-i moment
# 4: QTN Density


import numpy as np
import cdflib
import matplotlib.pyplot as plt
import os, sys
import pickle
from datetime import datetime

import functions as fn

if __name__ == "__main__":
    trange = ['2024-12-24T10:00:00', '2024-12-24T12:00:00']
    start_time = datetime.strptime(trange[0], '%Y-%m-%dT%H:%M:%S')
    end_time   = datetime.strptime(trange[1], '%Y-%m-%dT%H:%M:%S')
    
    credentials = fn.load_config('./config.json')
    creds = [credentials['psp']['sweap']['username'], credentials['psp']['sweap']['password']]

    den_rec = pickle.load(open('/home/michael/Research/GDF/Outputs/den_rec_data_32_to_533.pkl', 'rb'))
    vel_rec = pickle.load(open('/home/michael/Research/GDF/Outputs/v_rec_data_32_to_533.pkl', 'rb'))

    n_rec = np.array([den_rec[i] for i in range(500)])
    v_rec = np.array([vel_rec[i] for i in range(500)])


    cdfdata = cdflib.cdf_to_xarray('/home/michael/Research/GDF/biMax_Fits/spp_swp_spi_sf00_fits_2024-12-24_v00.cdf', to_datetime=True)
    fit_sel = cdfdata.sel(Epoch=slice(start_time, end_time))

    mom_data = fn.init_psp_moms(trange, CREDENTIALS=creds, CLIP=True)