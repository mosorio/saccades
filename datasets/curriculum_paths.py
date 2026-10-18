"""Shared filename scheme for curriculum-mode datasets.

Both `dataset_generator.py` (writes the files) and `loaders.py` (reads them
back at training time) need to compute the exact same filename from the same
settings. This module is the one place that naming logic lives, so the two
sides can't silently drift apart.
"""

CURRICULUM_DIRNAME = 'datasets/image_sets_curriculum'


def resolve_active_n_total_values(train_n_total_values, test_n_total_values, dataset_split, test_condition=None):
    """Which total-count (N) values feed this split/condition.

    Train uses its own N values as-is. Test conditions cross the (seen vs
    heldout combo) axis against the (in-range vs OOD N) axis:
    - 'ood_combo_inrange_n': N values also present in train_n_total_values
    - 'ood_combo_ood_n' / 'seen_combo_ood_n': N values NOT present in train_n_total_values
    """
    if dataset_split == 'train':
        return list(train_n_total_values)

    train_set = set(train_n_total_values)
    inrange = sorted(v for v in test_n_total_values if v in train_set)
    ood = sorted(v for v in test_n_total_values if v not in train_set)

    if test_condition == 'ood_combo_inrange_n':
        return inrange
    elif test_condition in ('ood_combo_ood_n', 'seen_combo_ood_n'):
        return ood
    else:
        raise ValueError(
            f"Unknown test_condition {test_condition!r} for dataset_split='test'. "
            "Choose one of: ood_combo_inrange_n, ood_combo_ood_n, seen_combo_ood_n"
        )


def build_curriculum_filename(*, curriculum, dataset_split, test_condition, active_n_total_values,
                               noise_level, shapes, distinctiveness, challenge, grid, policy,
                               luminances, transform, n_glimpses, size, trunc='', logscale='',
                               dirname=CURRICULUM_DIRNAME):
    """Reproduce dataset_generator.py's curriculum-mode filename, exactly."""
    cond_suffix = f'-{test_condition}' if test_condition else ''
    split_suffix = f'_curr-{curriculum}_split-{dataset_split}{cond_suffix}'
    num_label = f'nvals-{"-".join(str(v) for v in active_n_total_values)}'
    return (
        f'{dirname}/{num_label}_nl-{noise_level}{trunc}{logscale}_{shapes}{distinctiveness}'
        f'{challenge}{split_suffix}_grid{grid}_policy-{policy}_lum{luminances}_{transform}{n_glimpses}{size}'
    )
