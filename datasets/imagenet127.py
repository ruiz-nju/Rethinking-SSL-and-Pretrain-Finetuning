import os
import pickle
from collections import OrderedDict
import pdb
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class ImageNet127(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed = os.path.join(self.dataset_dir, "preprocessed_127.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot_127")
        self.oldcls2newcls_and_newidx = self.get_1000_to_127_mapping()
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed):
            with open(self.preprocessed, "rb") as f:
                preprocessed = pickle.load(f)
                train = preprocessed["train"]
                test = preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, 'classnames_127.txt')
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            # Follow standard practice to perform evaluation on the val set
            # Also used as the val set (so evaluate the last-step model)
            test = self.read_data(classnames, "val")

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
        train, test = OxfordPets.subsample_classes(train, test, subsample=subsample)

        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(" ")
                folder = line[0]
                classname = " ".join(line[1:])
                classnames[folder] = classname
        return classnames
    
    def read_data(self, classnames, split_dir):
        split_dir = os.path.join(self.image_dir, split_dir)
        folders = sorted(f.name for f in os.scandir(split_dir) if f.is_dir())
        items = []

        for old_cls in folders:
            imnames = listdir_nohidden(os.path.join(split_dir, old_cls))
            for imname in imnames:
                impath = os.path.join(split_dir, old_cls, imname)
                # 重新将label和classname映射到新的label和classname
                new_cls, new_idx = self.oldcls2newcls_and_newidx[old_cls]
                classname = classnames[new_cls]
                item = Datum(impath=impath, label=new_idx, classname=classname)
                items.append(item)

        return items

    def get_1000_to_127_mapping(self):
        # 将旧class的dirname映射到新的class的dirname以及class_idx
        oldcls2newcls = {}
        imgname2folder_path = os.path.join(self.dataset_dir, 'imgname2folder.txt')
        with open(imgname2folder_path, 'r') as f:
            txt = f.readlines()
        for line in txt:
            img_name, folder = line.strip().split(' ')
            old_class = img_name[:-5].split('_')[0]
            if oldcls2newcls.get(old_class) is None:
                oldcls2newcls[old_class] = folder
        # print(oldcls2newcls)

        # compute newcls2idx
        newcls2idx = {}
        classnames_127 = os.path.join(self.dataset_dir, 'classnames_127.txt')
        with open(classnames_127, 'r') as f:
            txt = f.readlines()
        for idx, line in enumerate(txt):
            new_cls, *_ = line.split()
            newcls2idx[new_cls] = idx
        # print(newcls2idx)

        oldcls2newcls_and_newidx = {}
        for old_cls, new_cls in oldcls2newcls.items():
            oldcls2newcls_and_newidx[old_cls] = (new_cls, newcls2idx[new_cls])
            
        return oldcls2newcls_and_newidx

    def print_mapping(self):
        classnames_1000 = os.path.join(self.dataset_dir, 'classnames.txt')
        classnames_127 = os.path.join(self.dataset_dir, 'classnames_127.txt')
        
        with open(classnames_1000, 'r') as f:
            txt = f.readlines()
        old_classnames = {}
        for line in txt:
            old_cls, old_classname = line.strip().split(' ', 1)
            old_classnames[old_cls] = old_classname
            
        with open(classnames_127, 'r') as f:
            txt = f.readlines()
        new_classnames = {}
        for line in txt:
            new_cls, new_classname = line.strip().split(' ', 1)
            new_classnames[new_cls] = new_classname
            
        new_to_old_mapping = {}
        for old_cls, (new_cls, _) in self.oldcls2newcls_and_newidx.items():
            if new_cls not in new_to_old_mapping:
                new_to_old_mapping[new_cls] = []
            new_to_old_mapping[new_cls].append((old_cls, old_classnames[old_cls]))
            

        print("\nMapping from 127 classes to 1000 classes:")
        print("=" * 80)
        for new_cls, new_classname in new_classnames.items():
            print(f"\n{new_cls}: {new_classname}")
            print("-" * 40)
            if new_cls in new_to_old_mapping:
                for old_cls, old_classname in sorted(new_to_old_mapping[new_cls]):
                    print(f"  - {old_cls}: {old_classname}")
            else:
                print("  No corresponding old classes found")
        print("\n" + "=" * 80)

    def get_127_to_1000_name_mapping(self):
        classnames_1000 = os.path.join(self.dataset_dir, 'classnames.txt')
        classnames_127 = os.path.join(self.dataset_dir, 'classnames_127.txt')
        
        with open(classnames_1000, 'r') as f:
            txt = f.readlines()
        old_classnames = {}
        for line in txt:
            old_cls, old_classname = line.strip().split(' ', 1)
            old_classnames[old_cls] = old_classname
            
        with open(classnames_127, 'r') as f:
            txt = f.readlines()
        new_classnames = {}
        for line in txt:
            new_cls, new_classname = line.strip().split(' ', 1)
            new_classnames[new_cls] = new_classname
            
        new_to_old_mapping = {}
        for old_cls, (new_cls, _) in self.oldcls2newcls_and_newidx.items():
            if new_cls not in new_to_old_mapping:
                new_to_old_mapping[new_cls] = []
            new_to_old_mapping[new_cls].append((old_cls, old_classnames[old_cls]))
            
        new_name_to_old_name_mapping = {}
        for new_cls, new_classname in new_classnames.items():
            new_name_to_old_name_mapping[new_classname] = []
            if new_cls in new_to_old_mapping:
                for old_cls, old_classname in sorted(new_to_old_mapping[new_cls]):
                    new_name_to_old_name_mapping[new_classname].append(old_classname)
        
        return new_name_to_old_name_mapping
        
        