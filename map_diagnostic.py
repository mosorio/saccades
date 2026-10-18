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

def shapes_present_from_pred_shapes(pred_shapes, absence_class=-1):
    """
    Given predicted shapes of shape (B,M), extract the unique shapes present in each sample.
    The absence class is excluded from each sample set.
    Parameters
    ----------
    pred_shapes : np.ndarray
        Predicted shapes of shape (B,M)
    absence_class : int
        The index of the absence class. If -1, infer as the max class index found in pred_shapes.
    Returns
    -------
    list
        A list of length B, each element is a list of unique shape indices present in that sample

    """
    pred_shapes = np.asarray(pred_shapes)
    if pred_shapes.ndim != 2:
        raise ValueError(f"pred_shapes must be 2D (B,M), got shape {pred_shapes.shape}")

    if absence_class == -1:
        absence_class = int(np.max(pred_shapes))

    shape_sets = []
    for b in range(pred_shapes.shape[0]):
        shapes = np.unique(pred_shapes[b])
        shapes = shapes[shapes != absence_class]  # drop absence
        shape_sets.append(shapes.tolist())
    return shape_sets


def compute_shape_map_accuracies(pred_shape_map: np.ndarray, target_shape_map: np.ndarray, absence_class: int) -> dict:
    """ 
    Given predicted and target shape maps of shape (B,S,M), compute various accuracy metrics.
    For target and prediction of shape B,S,M, compute the following accuracies:
    1) overall accuracy: also called slot-wise accuracy (equals to the mean per-loc accuracy), the mean accuracy across all shapes and locations (mean of the accuracy matrix B,S,M)
    2) presence accuracy: for the slots that have shapes in target, whether predicted shape matches target shape
    3) absence accuracy: whether absence of shape is correctly predicted
    4) per-shape accuracy: for each shape, proportion of locations where predicted shape matches target shape
    5) per-location accuracy: for each location, proportion of samples where predicted shape matches target shape
    6) strict accuracy: proportion of samples where the predicted shape map exactly matches the target shape map (all locations must be correct)

    Parameters
    ----------
    pred_shape_map : np.ndarray
        Predicted shape map of shape (B,S,M)
    target_shape_map : np.ndarray
        Target shape map of shape (B,S,M)
    absence_class : int
        The index of the absence class (default is 0)
    Returns
    -------
    dict
        A dictionary containing various accuracy metrics averaged over the batch:
        - overall_accuracy: float, mean accuracy across all shapes and locations
        - presence_accuracy: float, accuracy for locations with shapes present
        - absence_accuracy: float, accuracy for locations with no shapes
        - per_shape_accuracies: dict, accuracy for each individual shape
        - per_location_accuracies: dict, accuracy for each individual location
        - strict_accuracy: float, proportion of samples where the predicted shape map exactly matches the target shape map

    """
    B,C,M = np.array(pred_shape_map).shape
    assert pred_shape_map.shape == target_shape_map.shape, f"Shape mismatch: pred {pred_shape_map.shape}, target {target_shape_map.shape}"

    # Convert target (one-hot along C) to class indices
    # If target is strictly binary one-hot, argmax works. If it's already class indices, user should convert first.
    target_classes = np.argmax(target_shape_map, axis=1)  # shape (B, M)

    # Convert predictions to class indices (argmax along C)
    pred_classes = np.argmax(pred_shape_map, axis=1)      # shape (B, M)

    correct = (pred_classes == target_classes)      # (B, M)

    # -------- 1) Overall accuracy --------
    overall_acc = correct.mean().item()

    # -------- 2-3) Presence/Absence accuracy --------
    # boolean mask for presence / absence
    is_absence = (target_classes == absence_class)            # shape (B, M)
    is_presence = ~is_absence

    # Flatten masks and predictions/targets for overall presence/absence accuracies
    total_presence = is_presence.sum()
    total_absence = is_absence.sum()

    correct_flat = correct.reshape(-1) # shape (B*M,)
    target_flat = target_classes.reshape(-1)
    pred_flat = pred_classes.reshape(-1)
    is_presence_flat = is_presence.reshape(-1)
    is_absence_flat = is_absence.reshape(-1)

    # 2) mean accuracy of presence locations
    if total_presence > 0:
        presence_acc = correct_flat[is_presence_flat].mean().item()
    else:
        presence_acc = np.nan
    
    # 3) mean accuracy of absence locations
    if total_absence > 0:
        absence_acc = correct_flat[is_absence_flat].mean().item()
    else:
        absence_acc = np.nan

    # -------- 4) mean accuracy per classy --------
    per_class_acc = np.full((C,), np.nan, dtype=float)
    per_class_counts = np.zeros((C,), dtype=int)
    for c in range(C):
        mask_c = (target_flat == c)
        count_c = mask_c.sum()
        per_class_counts[c] = int(count_c)
        if count_c > 0:
            per_class_acc[c] = correct_flat[mask_c].mean().item()

    # 5) mean accuracy per location (over the B batch items) => for each m in 0..M-1
    per_location_acc = np.full((M,), np.nan, dtype=float)
    per_location_counts = np.zeros((M,), dtype=int)
    # For each location m compute across batch dim
    # pred_classes and target_classes are (B, M)
    for m in range(M):
        targ_at_m = target_classes[:, m]
        pred_at_m = pred_classes[:, m]
        # count = B (unless you want to ignore some examples, but here assume all B valid)
        correct_at_m = (pred_at_m == targ_at_m)
        per_location_counts[m] = int(len(correct_at_m))  # normally B
        if per_location_counts[m] > 0:
            per_location_acc[m] = correct_at_m.mean().item()

    # 6) strict accuracy: proportion of samples where the predicted shape map exactly matches the target shape map (all locations must be correct)
    strict_acc = float(np.all(pred_classes == target_classes, axis=1).mean())

    return {
        "overall_accuracy": overall_acc,
        "presence_accuracy": presence_acc,
        "absence_accuracy": absence_acc,
        "per_shape_accuracies": per_class_acc,
        "per_location_accuracies": per_location_acc,
        "strict_accuracy": strict_acc
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


def _safe_div(num, den, default=np.nan):
    return num / den if den > 0 else default


def _safe_nanmean(values, default=np.nan):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float(default)
    valid = ~np.isnan(arr)
    if not np.any(valid):
        return float(default)
    return float(arr[valid].mean())


def compute_shape_set_metrics_from_logits(
    map_logits: np.ndarray,
    target_map: np.ndarray,
    *,
    linkfunction,
    absence_class: int,
    topk_values=(2,),
    presence_threshold=0.5,
):
    """
    Compute set-level metrics over shape presence (excluding absence class):
    - exact set match
    - micro/macro precision, recall, F1
    - cardinality MAE
    - recall@k for configurable top-k values
    """
    probs = linkfunction(np.asarray(map_logits))  # (B,C,M)
    target_map = np.asarray(target_map)

    B, C, M = probs.shape
    shape_class_indices = [c for c in range(C) if c != absence_class]

    # Presence score per shape class for each sample (B, N_shape)
    # Max-over-slots is a simple pooling for "is shape present somewhere"
    shape_scores = probs[:, shape_class_indices, :].max(axis=2)

    # True/pred binary presence vectors (B, N_shape)
    true_presence = (target_map[:, shape_class_indices, :].max(axis=2) > 0).astype(np.int32)
    pred_presence = (shape_scores >= float(presence_threshold)).astype(np.int32)

    # Exact set match
    exact_set_match = float(np.all(pred_presence == true_presence, axis=1).mean())

    # Micro precision/recall/F1
    tp = int(((pred_presence == 1) & (true_presence == 1)).sum())
    fp = int(((pred_presence == 1) & (true_presence == 0)).sum())
    fn = int(((pred_presence == 0) & (true_presence == 1)).sum())

    precision_micro = _safe_div(tp, tp + fp, default=0.0)
    recall_micro = _safe_div(tp, tp + fn, default=0.0)
    f1_micro = _safe_div(2 * precision_micro * recall_micro, precision_micro + recall_micro, default=0.0)

    # Macro (per-sample, then mean)
    precision_samples, recall_samples, f1_samples = [], [], []
    for b in range(B):
        tp_b = int(((pred_presence[b] == 1) & (true_presence[b] == 1)).sum())
        fp_b = int(((pred_presence[b] == 1) & (true_presence[b] == 0)).sum())
        fn_b = int(((pred_presence[b] == 0) & (true_presence[b] == 1)).sum())
        p_b = _safe_div(tp_b, tp_b + fp_b, default=np.nan)
        r_b = _safe_div(tp_b, tp_b + fn_b, default=np.nan)
        f_b = _safe_div(2 * p_b * r_b, p_b + r_b, default=np.nan) if (not np.isnan(p_b) and not np.isnan(r_b)) else np.nan
        precision_samples.append(p_b)
        recall_samples.append(r_b)
        f1_samples.append(f_b)

    precision_macro = _safe_nanmean(precision_samples)
    recall_macro = _safe_nanmean(recall_samples)
    f1_macro = _safe_nanmean(f1_samples)

    # Cardinality metrics
    pred_card = pred_presence.sum(axis=1)
    true_card = true_presence.sum(axis=1)
    cardinality_mae = float(np.mean(np.abs(pred_card - true_card)))
    mean_pred_cardinality = float(np.mean(pred_card))
    mean_true_cardinality = float(np.mean(true_card))

    # Flexible recall@k (ranking by presence score)
    topk_recall = {}
    if isinstance(topk_values, int):
        topk_values = [topk_values]
    topk_values = sorted(set(int(k) for k in topk_values if int(k) > 0))

    sorted_idx = np.argsort(-shape_scores, axis=1)
    n_shape = shape_scores.shape[1]
    for k in topk_values:
        k_eff = min(k, n_shape)
        recalls = []
        for b in range(B):
            true_set = set(np.where(true_presence[b] == 1)[0].tolist())
            if len(true_set) == 0:
                recalls.append(np.nan)
                continue
            pred_topk = set(sorted_idx[b, :k_eff].tolist())
            recalls.append(len(pred_topk & true_set) / len(true_set))
        topk_recall[f"shape_set_recall_at_{k_eff}"] = _safe_nanmean(recalls)

    return {
        "shape_set_exact_match": exact_set_match,
        "shape_set_precision_micro": float(precision_micro),
        "shape_set_recall_micro": float(recall_micro),
        "shape_set_f1_micro": float(f1_micro),
        "shape_set_precision_macro": precision_macro,
        "shape_set_recall_macro": recall_macro,
        "shape_set_f1_macro": f1_macro,
        "shape_set_cardinality_mae": cardinality_mae,
        "shape_set_pred_cardinality_mean": mean_pred_cardinality,
        "shape_set_true_cardinality_mean": mean_true_cardinality,
        **topk_recall,
    }

# put together the metrics and return the mean values in a dictionary
def compute_map_metrics(
    map_logits:np.ndarray,
    target_map:np.ndarray,
    linkfunction=None,
    absence_class=-1,
    n_shapes = None,
    topk_values=(2,),
    presence_threshold=0.5,
    include_shape_set_metrics=True,
) -> dict:
    """
    Compute various metrics for the predicted shape map against the target shape map.
    Parameters
    ----------
    map_logits : np.ndarray
        The map logits of shape (B,S,M)
    target_map : np.ndarray
        The target shape map of shape (B,S,M)
    linkfunction : function, optional
        The link function to apply to the logits (default is None), can choose from None, softmax or sigmoid
    absence_class : int, optional
        The index of the absence class (default is -1, which infers as the max class index)
    topk_values : int or list of ints, optional
        The k values for computing recall@k (default is (2,)), can be a single int or a list of ints
    presence_threshold : float, optional
        The threshold for predicting presence of a shape in set-level metrics (default is 0.5) 
    include_shape_set_metrics : bool, optional
        Whether to include shape-set level metrics (default is True)
        
    Returns
    -------
    dict
        A dictionary containing various metrics:
        - overall_accuracy: float, mean accuracy across all shapes and locations
        - presence_accuracy: float, accuracy for locations with shapes present
        - absence_accuracy: float, accuracy for locations with no shapes
        - per_shape_accuracies: dict, accuracy for each individual shape
        - shape_presence_jaccard: float, mean Jaccard index for the shape sets present in the predictions vs targets
        - optional shape-set metrics (precision/recall/F1/exact/cardinality/top-k recall)
    """

    if linkfunction is None:
        linkfunction = lambda x: x
    elif linkfunction == "softmax":
        linkfunction = softmax
    elif linkfunction == "sigmoid":
        linkfunction = elementwise_sigmoid
    else:
        raise ValueError(f"Unsupported linkfunction: {linkfunction}")
    
    if absence_class == -1:
        assert n_shapes is not None, "n_shapes must be provided if absence_class is -1"
        absence_class = n_shapes  # assume last class is absence class by default

    pred_shape_map = get_shape_map_from_logits(map_logits, linkfunction=linkfunction)
    shape_map_accuracies = compute_shape_map_accuracies(pred_shape_map, target_map, absence_class=absence_class)

    pred_shapes, _, _ = get_shape_prediction_from_map(pred_shape_map)
    true_shapes, _, _ = get_shape_prediction_from_map(target_map)

    shape_sets_pred = shapes_present_from_pred_shapes(pred_shapes, absence_class=absence_class)
    shape_sets_true = shapes_present_from_pred_shapes(true_shapes, absence_class=absence_class)

    shape_jaccard_indices = compute_shape_matches(shape_sets_pred, shape_sets_true)

    # take the means disregarding NaNs
    mean_shape_presence_jaccard = np.nanmean(shape_jaccard_indices)

    overall_accuracy = float(shape_map_accuracies["overall_accuracy"])
    presence_accuracy = float(shape_map_accuracies["presence_accuracy"])
    absence_accuracy = float(shape_map_accuracies["absence_accuracy"])
    per_shape_accuracies = shape_map_accuracies["per_shape_accuracies"]
    per_location_accuracies = shape_map_accuracies["per_location_accuracies"]
    strict_accuracy = float(shape_map_accuracies["strict_accuracy"])

    metrics = {
        "overall_accuracy": overall_accuracy,
        "presence_accuracy": presence_accuracy,
        "absence_accuracy": absence_accuracy,
        "per_shape_accuracies": per_shape_accuracies,
        "per_location_accuracies": per_location_accuracies,
        "strict_accuracy": strict_accuracy,
        "mean_per_shape_accuracy": float(np.nanmean(per_shape_accuracies)),
        "mean_per_location_accuracy": float(np.nanmean(per_location_accuracies)),
        "shape_presence_jaccard": mean_shape_presence_jaccard

    }

    if include_shape_set_metrics:
        set_metrics = compute_shape_set_metrics_from_logits(
            map_logits,
            target_map,
            linkfunction=linkfunction,
            absence_class=absence_class,
            topk_values=topk_values,
            presence_threshold=presence_threshold,
        )
        metrics.update(set_metrics)

    return metrics