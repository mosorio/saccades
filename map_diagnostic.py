"""
This module provides utility functions for computing the performance of the map head
in predicting shapes at different spatial locations. 
"""
import numpy as np
from scipy.stats import norm

def softmax(x, axis=None):
    # subtract max for numerical stability
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def elementwise_sigmoid(logits: np.ndarray) -> np.ndarray:
    probs = 1 / (1 + np.exp(-logits))
    return probs

def get_shape_map_from_logits(map_logits: np.ndarray, linkfunction = lambda x: x) -> np.ndarray:
    """
    Given map logits of shape (B,S,M), predict the shapes (plus absence) that occupied each location
    B: number of samples
    S: number of all possible shape+1 to include the "no shape" class to indicate absence of shape
    M: flattened spatial dimensions
    The prediction is done by applying the provided linkfunction at each location across heads,
    Returns the index of the predicted location

    Parameters
    ----------
    map_logits : list or np.ndarray
        The map logits of shape (B,S,M)
    linkfunction : function, optional
        The link function to apply to the logits (default is identity)
        
    Returns
    -------
    np.ndarray
        pred_shape_map: np.ndarray of shape (B,S,M), binarised predicted shape map
    """
    map_logits = np.asarray(map_logits)
    B, S, M = map_logits.shape
    probs = linkfunction(map_logits)
    # For each location, find the head with the maximum probability
    pred_shapes = np.argmax(probs, axis=1)  # shape (B,M)

    # make pred_shapes into a binarised map of B,S,M
    pred_shape_map = np.zeros((B,S,M), dtype=int)
    pred_shape_map[np.arange(B)[:, None], pred_shapes, np.arange(M)] = 1
    return pred_shape_map

def get_shape_prediction_from_map(pred_shape_map) -> tuple:
    """
    Given map logits of shape (B,S,M), predict the shapes (plus absence) that occupied each location
    B: number of samples
    S: number of all possible shape+1 to include the "no shape" class to indicate absence of shape
    M: flattened spatial dimensions
    The prediction is done by applying the provided linkfunction at each location across heads,
    Returns the index of the predicted location

    Parameters
    ----------
    map_logits : list or np.ndarray
        The map logits of shape (B,S,M)
    linkfunction : function, optional
        The link function to apply to the logits (default is identity)
        
    Returns
    -------
    tuple
        pred_shapes: np.ndarray of shape (B,M), predicted shape index at each location \n
        where_shapes: list of lists, where_shapes[b][h] gives the locations where head h is predicted for sample b \n
        head_counts: list of lists, head_counts[b][h] gives the count of locations predicted as head h for sample b \n    
    """
    pred_shape_map = np.asarray(pred_shape_map)
    B,S,M = pred_shape_map.shape
    pred_shapes = np.argmax(pred_shape_map, axis=1)    # tells us the predicted shape index at each location
    assert pred_shapes.shape == (B,M), f"Expected pred_shapes shape {(B,M)}, got {pred_shapes.shape}"
    # get where each shape+absence is predicted for each sample
    where_shapes = [[np.where(pred_shapes[b]==h)[0] for h in range(S)] for b in range(B)]
    # Count occurrences of each head being the predicted head
    head_counts = [
        np.bincount(pred_shapes[b], minlength=S).tolist()
        for b in range(B)
    ]

    return pred_shapes, where_shapes, head_counts

def shapes_present_from_pred_shapes(pred_shapes):
    """
    Given predicted shapes of shape (B,M), extract the unique shapes present in each sample. We exclude the absence class (0).
    Parameters
    ----------
    pred_shapes : np.ndarray
        Predicted shapes of shape (B,M)
    Returns
    -------
    list
        A list of length B, each element is a list of unique shape indices present in that sample

    """
    shape_sets = []
    for b in range(pred_shapes.shape[0]):
        shapes = np.unique(pred_shapes[b])
        shapes = shapes[shapes != 0]  # drop absence
        shape_sets.append(shapes.tolist())
    return shape_sets


def compute_shape_map_accuracies(pred_shape_map, target_shape_map):
    """ 
    Given predicted and target shape maps of shape (B,S,M), compute various accuracy metrics.
    For target and prediction of shape B,S,M, compute the following accuracies:
    1) overall accuracy: mean accuracy across all shapes and locations (mean of the accuracy matrix B,S,M)
    2) presence accuracy: for the slots that have shapes in target, whether predicted shape matches target shape
    3) absence accuracy: whether absence of shape is correctly predicted
    4) per-shape accuracy: for each shape, proportion of locations where predicted shape matches target shape

    Parameters
    ----------
    pred_shape_map : np.ndarray
        Predicted shape map of shape (B,S,M)
    target_shape_map : np.ndarray
        Target shape map of shape (B,S,M)
    Returns
    -------
    dict
        A dictionary containing various accuracy metrics:
        - overall_accuracy: float, mean accuracy across all shapes and locations
        - presence_accuracy: float, accuracy for locations with shapes present
        - absence_accuracy: float, accuracy for locations with no shapes
        - per_shape_accuracies: dict, accuracy for each individual shape
    """
    B,S,M = np.array(pred_shape_map).shape
    assert pred_shape_map.shape == target_shape_map.shape, f"Shape mismatch: pred {pred_shape_map.shape}, target {target_shape_map.shape}"

    # correctness mask: True where prediction matches target
    correct = pred_shape_map == target_shape_map

    # target presence mask
    target_present = target_shape_map.astype(bool)

    # 1) Overall accuracy
    overall_accuracy = correct.mean() * 100

    # 2) Presence accuracy (exclude absence head 0)
    presence_mask = target_present[:, 1:, :]
    presence_correct = correct[:, 1:, :]
    presence_accuracy = presence_correct[presence_mask].mean() * 100

    # 3) Absence accuracy (head 0 only)
    absence_mask = target_present[:, 0, :]
    absence_correct = correct[:, 0, :]
    absence_accuracy = absence_correct[absence_mask].mean() * 100

    # 4) Per-shape accuracy (vectorized)
    per_shape_accuracy = correct.mean(axis=(0, 2)) * 100  # shape (S,)
    per_shape_accuracies = [per_shape_accuracy[h] for h in range(S)]

    return {
        "overall_accuracy": overall_accuracy,
        "presence_accuracy": presence_accuracy,
        "absence_accuracy": absence_accuracy,
        "per_shape_accuracies": per_shape_accuracies
    }
    
def to_iterable(x):
    """
    Convert input x to an iterable (list).
    Handles numpy scalars, numpy arrays, None, and assumes lists are already iterable.
    Parameters
    ----------
    x : any
        Input to convert to iterable
    Returns
    -------
    list
        Iterable version of x
    """
    # if x is a 0-d numpy scalar → treat as empty set
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return []
    # if x is a numpy array → flatten it to 1-D
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    # if x is None
    if x is None:
        return []
    # otherwise assume it's already iterable (list)
    return x

def jaccard(a, b):
    """
    Jaccard index between two sets/lists a and b
    Parameters
    ----------
    a : iterable
        First set/list
    b : iterable
        Second set/list
    Returns
    -------
    float
        Jaccard index between a and b
    """
    A = set(to_iterable(a))
    B = set(to_iterable(b))

    if not A and not B:
        return 1.0   # both empty → perfect similarity
    if not A or not B:
        return 0.0   # one empty → no overlap

    return len(A & B) / len(A | B)


def compute_sdt_metrics(pred_shape_map, target_shape_map):
    """
    Given predicted and target shape maps of shape (B,S,M), compute signal detection theory metrics
    Parameters
    ----------
    pred_shape_map : np.ndarray
        Predicted shape map of shape (B,S,M)
    target_shape_map : np.ndarray
        Target shape map of shape (B,S,M)
    Returns
    -------
    dict
        A dictionary containing SDT metrics:
        - hit_rate: float
        - false_positive_rate: float
        - false_negative_rate: float        
        - correct_rejection_rate: float
        - d_prime: float
    """
    pred_shape_map = np.array(pred_shape_map)
    target_shape_map = np.array(target_shape_map)

    # Basic counts
    hits = np.sum((target_shape_map == 1) & (pred_shape_map == 1))
    misses = np.sum((target_shape_map == 1) & (pred_shape_map == 0))
    false_alarms = np.sum((target_shape_map == 0) & (pred_shape_map == 1))
    correct_rejects = np.sum((target_shape_map == 0) & (pred_shape_map == 0))

    n_signal = hits + misses
    n_noise = false_alarms + correct_rejects

    # Correction for 0 or 1 rates (Macmillan & Creelman)
    def adjust(rate, n):
        if rate == 0:
            return 1 / (2*n)
        if rate == 1:
            return 1 - 1/(2*n)
        return rate

    # Rates
    hit_rate = hits / n_signal if n_signal > 0 else np.nan
    false_negative_rate = misses / n_signal if n_signal > 0 else np.nan

    false_positive_rate = false_alarms / n_noise if n_noise > 0 else np.nan
    correct_rejection_rate = correct_rejects / n_noise if n_noise > 0 else np.nan

    # Adjusted rates for d'
    if n_signal > 0:
        hit_rate_adj = adjust(hit_rate, n_signal)
    else:
        hit_rate_adj = np.nan

    if n_noise > 0:
        false_positive_rate_adj = adjust(false_positive_rate, n_noise)
    else:
        false_positive_rate_adj = np.nan

    # d'
    if np.isnan(hit_rate_adj) or np.isnan(false_positive_rate_adj):
        dprime = np.nan
    else:
        dprime = norm.ppf(hit_rate_adj) - norm.ppf(false_positive_rate_adj)

    return {
        "hit_rate": hit_rate,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "correct_rejection_rate": correct_rejection_rate,
        "d_prime": dprime
    }

    
## compute the prediction of presence/absence of shapes in the image: jaccard index between the predicted set and true set
def compute_shape_matches(pred_shapes, true_shapes):
    """
    Given predicted and true shapes for each sample, compute the Jaccard index for shape presence/absence.

    Parameters
    ----------
    pred_shapes : list of lists
        has length B (batch size), predicted shapes for each sample. Each element is a list of predicted shape indices in one sample.
    true_shapes : list of lists
        has length B (batch size), True shapes for each sample. Each element is a list of true shape indices in one sample.

    Returns
    -------
    list
        A list of Jaccard indices for each sample.
    """
    assert len(pred_shapes) == len(true_shapes), "Length mismatch between predicted and true shapes"
    return [jaccard(pred, true) for pred, true in zip(pred_shapes, true_shapes)]

# put together the metrics and return the mean values in a dictionary
def compute_map_metrics(map_logits:np.ndarray,target_map:np.ndarray) -> dict:
    """
    Compute various metrics for the predicted shape map against the target shape map.
    Parameters
    ----------
    map_logits : np.ndarray
        The map logits of shape (B,S,M)
    target_map : np.ndarray
        The target shape map of shape (B,S,M)
    Returns
    -------
    dict
        A dictionary containing various metrics:
        - overall_accuracy: float, mean accuracy across all shapes and locations
        - presence_accuracy: float, accuracy for locations with shapes present
        - absence_accuracy: float, accuracy for locations with no shapes
        - per_shape_accuracies: dict, accuracy for each individual shape
        - shape_presence_jaccard: float, mean Jaccard index for the shape sets present in the predictions vs targets
    """
    pred_shape_map = get_shape_map_from_logits(map_logits, linkfunction=softmax)
    shape_map_accuracies = compute_shape_map_accuracies(pred_shape_map, target_map)

    pred_shapes, _, _ = get_shape_prediction_from_map(pred_shape_map)
    true_shapes, _, _ = get_shape_prediction_from_map(target_map)

    shape_sets_pred = shapes_present_from_pred_shapes(pred_shapes)
    shape_sets_true = shapes_present_from_pred_shapes(true_shapes)

    shape_jaccard_indices = compute_shape_matches(shape_sets_pred, shape_sets_true)

    # take the means disregarding NaNs
    mean_shape_presence_jaccard = np.nanmean(shape_jaccard_indices)
    shape_map_accuracies = {
        k: np.nanmean(v) for k, v in shape_map_accuracies.items()
    }
    return {
        **shape_map_accuracies,
        "shape_presence_jaccard": mean_shape_presence_jaccard

    }
