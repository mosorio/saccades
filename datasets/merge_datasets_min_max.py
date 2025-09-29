import os
import numpy as np
import pandas as pd
import xarray as xr
from itertools import product
from matplotlib import pyplot as plt
import argparse
import os
import re

# Set your data directory
DATA_DIR = '/Users/mosorio/Documents/ChrisWork/saccades/datasets/image_sets_min_max/'

# def merge_datasets(size_filter):
#     """
#     Merge all datasets in DATA_DIR matching size_filter.
#     Automatically constructs the merged filename by keeping the
#     original naming convention and replacing the size with the new total.
#     """
#     # Find matching files
#     files = [f for f in os.listdir(DATA_DIR) if f.endswith(size_filter)]
#     if not files:
#         raise ValueError(f"No files found in {DATA_DIR} with filter {size_filter}")
#     print(f"Found {len(files)} files to merge: {files}")

#     datasets, total_size = [], 0
#     for f in files:
#         ds = xr.open_dataset(os.path.join(DATA_DIR, f))
#         datasets.append(ds)
#         # Extract size (last number before .nc)
#         match = re.search(r"_(\d+)\.nc$", f)
#         if match:
#             total_size += int(match.group(1))

#     # Concatenate
#     merged = xr.concat(datasets, dim="image")

#     # Use first filename as template
#     template = files[0]
#     merged_name = re.sub(r"_(\d+)\.nc$", f"_{total_size}.nc", template)

#     out_path = os.path.join(DATA_DIR, merged_name)
#     merged.to_netcdf(out_path)

#     print(f"Merged dataset saved to {out_path}")


# if __name__ == "__main__":
#     # Merge training (100000 each → 400000, etc.)
#     merge_datasets(size_filter="_100.nc")

#     # Merge test (1000 each → 4000, etc.)
#     merge_datasets(size_filter="_10.nc")



# --- Training data ---
E01file = 'num3-7_nl-0.74_BCDEdistinct-0.3_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_10000.nc'
E03file = 'num3-7_nl-0.74_BCDEdistinct-0.3_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_10000.nc'
F01file = 'num3-7_nl-0.74_FGHJdistinct-0.3_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_10000.nc'
F03file = 'num3-7_nl-0.74_FGHJdistinct-0.3_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_10000.nc'

E01 = xr.open_dataset(DATA_DIR + E01file)
E03 = xr.open_dataset(DATA_DIR + E03file)
F01 = xr.open_dataset(DATA_DIR + F01file)
F03 = xr.open_dataset(DATA_DIR + F03file)
merged = xr.concat([E01, E03, F01, F03], dim='image')
merged.to_netcdf(DATA_DIR + 'num3-7_nl-0.74_BCDEFGHJdistinct-0.3_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_12_40000.nc')


# --- Test data ---
E01file = 'num3-7_nl-0.74_BCDEdistinct-0.3_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_1000.nc'
F01file = 'num3-7_nl-0.74_FGHJdistinct-0.3_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7]_logpolar_12_1000.nc'
E03file = 'num3-7_nl-0.74_BCDEdistinct-0.3_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_1000.nc'
F03file = 'num3-7_nl-0.74_FGHJdistinct-0.3_grid6_policy-cheat+jitter_lum[0.3, 0.6, 0.9]_logpolar_12_1000.nc'

E01 = xr.open_dataset(DATA_DIR + E01file)
E03 = xr.open_dataset(DATA_DIR + E03file)
F01 = xr.open_dataset(DATA_DIR + F01file)
F03 = xr.open_dataset(DATA_DIR + F03file)
merged = xr.concat([E01, E03, F01, F03], dim='image')
merged.to_netcdf(DATA_DIR + 'num3-7_nl-0.74_BCDEFGHJdistinct-0.3_grid6_policy-cheat+jitter_lum[0.1, 0.4, 0.7, 0.3, 0.6, 0.9]_logpolar_12_4000.nc')


