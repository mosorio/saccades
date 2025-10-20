import os
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from matplotlib import pyplot as plt
import argparse

DATA_DIR = 'datasets/image_sets/'

p = argparse.ArgumentParser()
p.add_argument("--n_glimpses", "-g", type=int, default=12)
p.add_argument("--min_num", "-nmin", type=int, default=1)
p.add_argument("--max_num", "-nmax", type=int, default=5)
args = p.parse_args()



### Training data
# E01file = 'num1-5_nl-0.9_ESUZsame_distract123_grid6_policy-humanlike_lum[0.1, 0.4, 0.7]_logpolar_12_10000.nc'
# F01file = 'num1-5_nl-0.9_FCKJsame_distract123_grid6_policy-humanlike_lum[0.1, 0.4, 0.7]_logpolar_12_10000.nc'
# E03file = 'num1-5_nl-0.9_ESUZsame_distract123_grid6_policy-humanlike_lum[0.3, 0.6, 0.9]_logpolar_12_10000.nc'
# F03file = 'num1-5_nl-0.9_FCKJsame_distract123_grid6_policy-humanlike_lum[0.3, 0.6, 0.9]_logpolar_12_10000.nc'
E01file = f'num{args.min_num}-{args.max_num}_nl-0.74_BCDEsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_{args.n_glimpses}_10000.nc'
F01file = f'num{args.min_num}-{args.max_num}_nl-0.74_FGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_{args.n_glimpses}_10000.nc'
E03file = f'num{args.min_num}-{args.max_num}_nl-0.74_BCDEsame_distract012_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_{args.n_glimpses}_10000.nc'
F03file = f'num{args.min_num}-{args.max_num}_nl-0.74_FGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_{args.n_glimpses}_10000.nc'
E01 = xr.open_dataset(DATA_DIR + E01file)
E03 = xr.open_dataset(DATA_DIR + E03file)
F01 = xr.open_dataset(DATA_DIR + F01file)
F03 = xr.open_dataset(DATA_DIR + F03file)
merged = xr.concat([E01, E03, F01, F03], dim='image')
merged.to_netcdf(DATA_DIR + f'num{args.min_num}-{args.max_num}_nl-0.74_BCDEFGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_{args.n_glimpses}_40000.nc')


### Testing data
# E01file = f'num1-5_nl-0.9_ESUZsame_distract123_grid6_policy-humanlike_lum[0.1, 0.4, 0.7]_logpolar_{args.n_glimpses}_1000.nc'
# F01file = f'num1-5_nl-0.9_FCKJsame_distract123_grid6_policy-humanlike_lum[0.1, 0.4, 0.7]_logpolar_{args.n_glimpses}_1000.nc'
# E03file = f'num1-5_nl-0.9_ESUZsame_distract123_grid6_policy-humanlike_lum[0.3, 0.6, 0.9]_logpolar_{args.n_glimpses}_1000.nc'
# F03file = f'num1-5_nl-0.9_FCKJsame_distract123_grid6_policy-humanlike_lum[0.3, 0.6, 0.9]_logpolar_{args.n_glimpses}_1000.nc'
E01file = f'num{args.min_num}-{args.max_num}_nl-0.74_BCDEsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_{args.n_glimpses}_1000.nc'
F01file = f'num{args.min_num}-{args.max_num}_nl-0.74_FGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_{args.n_glimpses}_1000.nc'
E03file = f'num{args.min_num}-{args.max_num}_nl-0.74_BCDEsame_distract012_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_{args.n_glimpses}_1000.nc'
F03file = f'num{args.min_num}-{args.max_num}_nl-0.74_FGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_{args.n_glimpses}_1000.nc'
E01 = xr.open_dataset(DATA_DIR + E01file)
E03 = xr.open_dataset(DATA_DIR + E03file)
F01 = xr.open_dataset(DATA_DIR + F01file)
F03 = xr.open_dataset(DATA_DIR + F03file)
merged = xr.concat([E01, E03, F01, F03], dim='image')
merged.to_netcdf(DATA_DIR + f'num{args.min_num}-{args.max_num}_nl-0.74_BCDEFGHJsame_distract012_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_{args.n_glimpses}_4000.nc')



