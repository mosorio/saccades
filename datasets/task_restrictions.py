"""Task-specific scene restrictions for dataset generation."""


TASK_RESTRICTIONS = {
    "count": {
        "allow_uniform_counts": True,
        "require_non_uniform_counts": False,
        "min_k": 1,
    },
    "count_total": {
        "allow_uniform_counts": True,
        "require_non_uniform_counts": False,
        "min_k": 1,
    },
    "count_per_shape": {
        "allow_uniform_counts": True,
        "require_non_uniform_counts": False,
        "min_k": 1,
    },
    "min": {
        "allow_uniform_counts": False,
        "require_non_uniform_counts": True,
        "min_k": 2,
    },
    "max": {
        "allow_uniform_counts": False,
        "require_non_uniform_counts": True,
        "min_k": 2,
    },
    "symmetry": {
        "allow_uniform_counts": True,
        "require_non_uniform_counts": False,
        "min_k": 2,
        "implemented": False,
    },
}


def get_task_restrictions(task_name):
    task_name = str(task_name).lower()
    if task_name not in TASK_RESTRICTIONS:
        raise ValueError(
            f"Unknown task_type '{task_name}'. Available: {sorted(TASK_RESTRICTIONS.keys())}"
        )
    cfg = dict(TASK_RESTRICTIONS[task_name])
    if cfg.get("implemented", True) is False:
        raise NotImplementedError(
            f"Task '{task_name}' is registered but its scene restrictions are not implemented yet."
        )
    return cfg


def minimum_total_for_k(task_name, k_unique):
    """Minimum total count N needed for a valid scene under this task."""
    cfg = get_task_restrictions(task_name)
    k_unique = int(k_unique)
    if cfg.get("require_non_uniform_counts", False) and k_unique > 1:
        return k_unique + 1
    return k_unique


def counts_are_valid(task_name, counts):
    """Whether a per-shape count vector satisfies task-specific restrictions."""
    cfg = get_task_restrictions(task_name)
    if not cfg.get("allow_uniform_counts", True) and len(set(int(c) for c in counts)) == 1:
        return False
    return True
