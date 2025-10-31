import xarray as xr
import os
import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import load
import pandas as pd
import json

ds_dir = "../datasets/image_sets_min_max_pairs_min7_max7"
json_pat = {
    "train": "*_pair-train*.json",
    "test":  "*_pair-test*.json"
}
ds_meta = {}
for k, v in json_pat.items():
    ds_meta[k] = json.load(open(glob.glob(os.path.join(ds_dir, v))[0], 'r'))

file_pat = {
    "train":      "*_pair-train_*_lum[[]0.1, 0.4, 0.7[]]*_100000.pkl",
    "validation": "*_pair-train_*_lum[[]0.1, 0.4, 0.7[]]*_5000.pkl",
    "OODlum":     "*_pair-train_*_lum[[]0.3, 0.6, 0.9[]]*_5000.pkl",
    "OODshape":   "*_pair-test_*_lum[[]0.1, 0.4, 0.7[]]*_5000.pkl",
    "OODboth":    "*_pair-test_*_lum[[]0.3, 0.6, 0.9[]]*_5000.pkl"
}
ds_paths = {}
for k, v in file_pat.items():
    print(f"Looking for {k} dataset files...")
    files = glob.glob(os.path.join(ds_dir, v))
    assert len(files) == 1, f"Expected exactly one file for {k}, found {len(files)}"
    print(f"{k} dataset file: {files[0]}")
    ds_paths[k] = files[0]

for dsname, ds_path in ds_paths.items():
    print(f"Checking {dsname} dataset...")
    ds = load(ds_path)
    #print(ds)

    # plot the distribution of shapes in the dataset
    pairs = []
    correct_labels = []
    correct_label_inpairs = []
    num_min = []
    for x in ds["shape_map"]:
        labels, counts = np.unique_counts(list(x.values()))
        min_label = labels[np.argmin(counts)]
        pairs.append(f"{labels[0]}_{labels[1]}")
        correct_labels.append(min_label)
        if counts[0] < counts[1]:
            correct_label_inpairs.append("l1")
        else:
            correct_label_inpairs.append("l2")
        num_min.append(counts.min())

    # check that num_min matches the dataset's numerosity_min
    assert np.array_equal(np.array(num_min), ds["numerosity_min"].values)
    # check if the actual dataset is in line with the metadata
    allowed_pairs = ds_meta["train"]["pairs"] if dsname in ["train", "validation", "OODlum"] else ds_meta["test"]["pairs"]
    allowed_pairs_str = [f"{p[0]}_{p[1]}" for p in allowed_pairs]
    assert np.array_equal(
        np.unique(pairs),
        np.array(sorted(allowed_pairs_str))
    )
    print(f"{dsname} dataset passed pairs checks.")
    

    
    df = pd.DataFrame({"pair": pairs, "min_count": num_min,
                       "correct_label": correct_labels, "correct_label_inpairs": correct_label_inpairs})
    df.sort_values("pair", inplace=True)
    plt.subplots(2,2, figsize=(12,12),gridspec_kw={'width_ratios': [1, 2]})
    plt.subplot(2,2,1)
    sns.countplot(x="min_count", data=df)
    plt.title("Distribution of min counts")

    plt.subplot(2,2,2)
    sns.countplot(x="pair", data=df)
    plt.title("Distribution of label pairs")

    plt.subplot(2,2,3)
    sns.countplot(x="correct_label", data=df)
    plt.title("Distribution of correct labels")

    plt.subplot(2,2,4)
    sns.countplot(x="pair", hue="correct_label_inpairs", data=df)
    plt.title("Correct label position in pairs")

    plt.suptitle(f"Dataset: {dsname}")
    plt.tight_layout()
    plt.show()