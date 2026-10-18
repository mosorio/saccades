"""Curriculum presets local to the image-based dataset generator."""


CURRICULUM_PRESETS = {
    "2to7": {"k_values": [2, 3, 4, 5, 6, 7], "eval_k_values": [2, 3, 4, 5, 6, 7]},
    "246": {"k_values": [2, 4, 6], "eval_k_values": [2, 3, 4, 5, 6, 7]},
    "357": {"k_values": [3, 5, 7], "eval_k_values": [2, 3, 4, 5, 6, 7]},
}


def resolve_curriculum_preset(name, valid_k_values, train_ratio=0.5):
    """Resolve a preset into explicit per-K train/test sampling config."""
    if not name:
        return None
    if name not in CURRICULUM_PRESETS:
        raise ValueError(
            f"Unknown curriculum '{name}'. Available: {sorted(CURRICULUM_PRESETS.keys())}"
        )

    valid_k_values = sorted(set(int(k) for k in valid_k_values))
    requested_k_values = [int(k) for k in CURRICULUM_PRESETS[name]["k_values"]]
    active_k_values = [k for k in requested_k_values if k in valid_k_values]
    if not active_k_values:
        raise ValueError(
            f"Curriculum '{name}' has no valid K values after intersecting with {valid_k_values}"
        )

    requested_eval_k_values = CURRICULUM_PRESETS[name].get("eval_k_values")
    if requested_eval_k_values is not None:
        # Preset declares its own eval range explicitly -- don't let it silently
        # expand to whatever else happens to be combinatorially valid (e.g. an
        # extra K from the task's min_k, or from the shape count itself).
        requested_eval_k_values = set(int(k) for k in requested_eval_k_values)
        eval_k_values = [k for k in valid_k_values if k in requested_eval_k_values]
    else:
        eval_k_values = valid_k_values.copy()
    split_ratio_by_k = {
        k: (float(train_ratio) if k in active_k_values else 0.0)
        for k in eval_k_values
    }
    test_only_k_values = [k for k in eval_k_values if k not in active_k_values]
    train_sampling_weights = {k: 1.0 for k in active_k_values}
    test_sampling_weights = {k: 1.0 for k in eval_k_values}

    return {
        "name": name,
        "train_k_values": active_k_values,
        "eval_k_values": eval_k_values,
        "split_ratio_by_k": split_ratio_by_k,
        "test_only_k_values": test_only_k_values,
        "train_sampling_weights": train_sampling_weights,
        "test_sampling_weights": test_sampling_weights,
    }
