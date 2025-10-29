import xarray as xr
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

ds_dir = "../minmaxdatasets"
ds_files = glob.glob(os.path.join(ds_dir, "*_40000.nc"))
ds_path = ds_files[0]
ds = xr.open_dataset(ds_path)

print(ds)

# plot the distribution of min values in the dataset
sns.histplot(ds["numerosity_min"].values)

vals, counts = np.unique(ds["numerosity_min"].values, return_counts=True)
counts/counts.sum()


# run a simulation to see if doing the inverse will give a uniform distribution
n_samples = 100000
probs = 1/ counts
probs = probs / probs.sum()
samples = np.random.choice(vals, size=n_samples, p=probs)
sns.histplot(samples)

# plot the distribution of shapes in the dataset
shapes = np.unique([np.where(x == 1)[0][0] if not np.size(np.where(x==1)[0])==0 else np.nan for x in ds["shape_hist"].values])
sns.countplot(x=shapes)