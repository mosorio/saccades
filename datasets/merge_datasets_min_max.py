import os
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from matplotlib import pyplot as plt
import argparse
import os

# Set your data directory
DATA_DIR = '/Users/mosorio/Documents/ChrisWork/saccades/datasets/image_sets_min_max/'

# Define the expected filenames for merging

# --- Training data ---
E01file = 'num3-7_nl-0.74_BCsame_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_100.nc'
E03file = 'num3-7_nl-0.74_BCsame_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_100.nc'
F01file = 'num3-7_nl-0.74_FGsame_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_100.nc'
F03file = 'num3-7_nl-0.74_FGsame_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_100.nc'

E01 = xr.open_dataset(DATA_DIR + E01file)
E03 = xr.open_dataset(DATA_DIR + E03file)
F01 = xr.open_dataset(DATA_DIR + F01file)
F03 = xr.open_dataset(DATA_DIR + F03file)
merged = xr.concat([E01, E03, F01, F03], dim='image')
merged.to_netcdf(DATA_DIR + 'num3-7_nl-0.74_BCFGsame_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_12_400.nc')


# --- Test data ---
E01file = 'num3-7_nl-0.74_BCsame_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_10.nc'
F01file = 'num3-7_nl-0.74_FGsame_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_10.nc'
E03file = 'num3-7_nl-0.74_BCsame_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_10.nc'
F03file = 'num3-7_nl-0.74_FGsame_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_10.nc'

E01 = xr.open_dataset(DATA_DIR + E01file)
E03 = xr.open_dataset(DATA_DIR + E03file)
F01 = xr.open_dataset(DATA_DIR + F01file)
F03 = xr.open_dataset(DATA_DIR + F03file)
merged = xr.concat([E01, E03, F01, F03], dim='image')
merged.to_netcdf(DATA_DIR + 'num3-7_nl-0.74_BCFGsame_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_12_40.nc')


