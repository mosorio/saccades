"""Helpers for building per-K train/test combination pools.
"""

from collections import Counter
from itertools import combinations
from math import comb

import numpy as np


def split_combinations_by_count(
    s_total,
    *,
    combo_size,
    n_train,
    labels=None,
    randomseed=None,
    n_search=1000,
    max_allowed_imbalance=1,
):
    """Split all C(S, K) combinations into train/test with balanced label use."""
    s_total = int(s_total)
    combo_size = int(combo_size)
    n_train = int(n_train)

    if labels is None:
        labels = [f"L{i}" for i in range(s_total)]
    else:
        labels = list(labels)
        if len(labels) != s_total:
            raise ValueError("labels length must equal s_total")

    if not (1 <= combo_size <= s_total):
        raise ValueError(f"combo_size must be in [1, {s_total}], got {combo_size}")

    all_combos = list(combinations(labels, combo_size))
    n_total = len(all_combos)
    if not (0 < n_train < n_total):
        raise ValueError(f"n_train must be between 1 and {n_total - 1}, got {n_train}")

    rng = np.random.RandomState(randomseed) if randomseed is not None else np.random

    def get_counts(group):
        cnt = Counter()
        for combo in group:
            for item in combo:
                cnt[item] += 1
        return cnt

    def within_set_imbalance(group):
        cnt = get_counts(group)
        vals = [cnt[l] for l in labels]
        return int(max(vals) - min(vals))

    def split_score(train_set, test_set):
        return max(within_set_imbalance(train_set), within_set_imbalance(test_set))

    best_split = None
    best_score = float("inf")

    for _ in range(int(n_search)):
        idx = rng.permutation(n_total)
        train_set = [all_combos[i] for i in idx[:n_train]]
        test_set = [all_combos[i] for i in idx[n_train:]]
        score = split_score(train_set, test_set)
        if score < best_score:
            best_score = score
            best_split = (train_set, test_set)
        if score == 0:
            break

    train_set, test_set = best_split
    if int(best_score) > int(max_allowed_imbalance):
        raise ValueError(
            f"Could not satisfy max_allowed_imbalance={max_allowed_imbalance} "
            f"after n_search={n_search}. Best score={int(best_score)}."
        )

    stats = {
        "s_total": s_total,
        "k_unique": combo_size,
        "combo_size": combo_size,
        "n_total": n_total,
        "n_train": len(train_set),
        "n_test": len(test_set),
        "max_imbalance_per_label": int(best_score),
    }
    return train_set, test_set, stats


def combinations_to_index_tuples(combos, label_order):
    """Normalize combination tuples while preserving actual shape IDs when possible."""
    normalized = []
    for combo in combos:
        try:
            normalized.append(tuple(sorted(int(x) for x in combo)))
        except (TypeError, ValueError):
            normalized.append(tuple(combo))
    return normalized


def build_combo_splits_by_k(
    n_objects,
    scene_k_values,
    labels,
    seed,
    per_k_train_ratio,
    test_only_k_values=None,
    n_search=10000,
):
    """Build train/test combination pools for every required K."""
    test_only_k_values = set(int(k) for k in (test_only_k_values or []))
    per_k_train_ratio = {int(k): float(v) for k, v in per_k_train_ratio.items()}

    train_by_k = {}
    test_by_k = {}
    stats_by_k = {}

    for k_val in sorted(int(k) for k in scene_k_values):
        if k_val in test_only_k_values:
            full_lbl = list(combinations(labels, k_val))
            train_lbl = []
            test_lbl = full_lbl
            stats_by_k[k_val] = {
                "s_total": int(n_objects),
                "k_unique": int(k_val),
                "combo_size": int(k_val),
                "n_total": comb(int(n_objects), int(k_val)),
                "n_train": 0,
                "n_test": comb(int(n_objects), int(k_val)),
                "max_imbalance_per_label": None,
                "test_only": True,
            }
        else:
            k_ratio = per_k_train_ratio.get(k_val, 0.0)
            n_total = comb(int(n_objects), int(k_val))
            if n_total < 2:
                raise ValueError(f"K={k_val} yields only {n_total} combinations; cannot split.")

            n_train_k = int(round(k_ratio * n_total))
            if n_train_k <= 0 or n_train_k >= n_total:
                raise ValueError(
                    f"train ratio {k_ratio} yields invalid split for K={k_val}: "
                    f"n_train={n_train_k}, n_test={n_total - n_train_k}"
                )

            train_lbl, test_lbl, stats = split_combinations_by_count(
                int(n_objects),
                combo_size=int(k_val),
                n_train=n_train_k,
                labels=labels,
                randomseed=int(seed) + 100 * int(k_val),
                n_search=n_search,
                max_allowed_imbalance=1,
            )
            stats["test_only"] = False
            stats_by_k[k_val] = stats

        train_by_k[k_val] = combinations_to_index_tuples(train_lbl, labels)
        test_by_k[k_val] = combinations_to_index_tuples(test_lbl, labels)

    return train_by_k, test_by_k, stats_by_k
