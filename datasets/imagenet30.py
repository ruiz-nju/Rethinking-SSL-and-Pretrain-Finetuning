import pdb
import os
import pickle
from collections import OrderedDict
import math
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet30(DatasetBase):

    dataset_dir = "imagenet-30"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            train = self.read_data("train")
            test = self.read_data("val")

            preprocessed = {"train": train, "test": test}
            with open(self.preprocessed, "wb") as f:
                pickle.dump(preprocessed, f, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                data = {"train": train}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        train, test = OxfordPets.subsample_classes(
            train, test, subsample=subsample, base_ratio=2 / 3
        )

        super().__init__(train_x=train, val=test, test=test)

    def read_data(self, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)  # images/train
        folders = sorted(
            f.name for f in os.scandir(split_dir) if f.is_dir()
        )  # acorn, american_alligator, ....
        items = []

        for label, folder in enumerate(folders):
            path = os.path.join(
                split_dir, folder
            )  # /mnt/hdd/DATA/imagenet-30/images/val/acorn
            if os.path.basename(split_dir).lower() == "val":
                sub_folder = [f.name for f in os.scandir(path) if f.is_dir()][0]
                path = os.path.join(path, sub_folder)
            imnames = listdir_nohidden(path)
            classname = folder.replace("_", " ")  # 将 "_" 替换为空格
            for imname in imnames:
                # /mnt/hdd/DATA/imagenet-30/images/train/acorn/n12267677_6732.JPEG
                impath = os.path.join(path, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
