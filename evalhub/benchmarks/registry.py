DATASET_MAP = {}
DATASET_HUB = {}
EVALUATE_DATASETS = set()
THIRD_PARTY_DATASETS = set()


def register_dataset(*names):
    r"""Decorator to register a dataset class."""

    def decorator(cls):
        for ds, hub, evaluable in names:
            DATASET_MAP[ds] = cls
            DATASET_HUB[ds] = hub
            if evaluable:
                EVALUATE_DATASETS.add(ds)
            else:
                THIRD_PARTY_DATASETS.add(ds)
        return cls

    return decorator
