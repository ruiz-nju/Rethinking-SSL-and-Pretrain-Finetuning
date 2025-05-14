from dassl.utils import Registry, check_availability

DATASET_REGISTRY = Registry("DATASET")


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)


def build_dataset_out(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.TEST.OOD_DATASET, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.TEST.OOD_DATASET))
    return DATASET_REGISTRY.get(cfg.TEST.OOD_DATASET)(cfg)
