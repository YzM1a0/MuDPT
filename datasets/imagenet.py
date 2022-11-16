import os
import math
import pickle

from collections import OrderedDict
from dassl.data.datasets import DatasetBase, Datum, DATASET_REGISTRY
from dassl.utils import listdir_nohidden, mkdir_if_missing


@DATASET_REGISTRY.register()
class ImageNet(DatasetBase):

    dataset_dir = "imagenet"

    def __init__(self, config):
        root = os.path.abspath(os.path.expanduser(config.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, "images")
        self.preprocessed_dir = os.path.join(self.dataset_dir, "preprocessed.pkl")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        if os.path.exists(self.preprocessed_dir):
            print(f"Loading preprocessed data from {self.preprocessed_dir}")
            with open(self.preprocessed_dir, "rb") as f:
                preprocessed = pickle.load(f)
                train, test = preprocessed["train"], preprocessed["test"]
        else:
            text_file = os.path.join(self.dataset_dir, "classnames.txt")
            classnames = self.read_classnames(text_file)
            train = self.read_data(classnames, "train")
            test = self.read_data(classnames, "val")
            preprocessed = {"train": train, "test": test}
            print(f"Saving preprocessed data to {self.preprocessed_dir}")
            with open(self.preprocessed_dir, "wb") as file:
                pickle.dump(preprocessed, file, protocol=pickle.HIGHEST_PROTOCOL)

        num_shots = config.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = config.SEED
            fewshot_preprocessed_dir = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")

            if os.path.exists(fewshot_preprocessed_dir):
                print(f"Loading few-shot data from {fewshot_preprocessed_dir}")
                with open(fewshot_preprocessed_dir, "rb") as file:
                    fewshot_preprocessed = pickle.load(file)
                    train = fewshot_preprocessed["train"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                fewshot_preprocessed = {"train": train}
                print(f"Saving few-shot data to {fewshot_preprocessed_dir}")
                with open(fewshot_preprocessed_dir, "wb") as file:
                    pickle.dump(fewshot_preprocessed, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = config.DATASET.SUBSAMPLE_CLASSES
        train, test = self.subsample_classes(train, test, subsample=subsample)
        super().__init__(train_x=train, val=test, test=test)

    @staticmethod
    def subsample_classes(*args, subsample="all"):
        """Divide classes into two groups. The first group represents base classes
        while the second group represents new classes.
        Args:
            args: a list of datasets, e.g. train, val and test.
            subsample (str): what classes to subsample.
        """
        assert subsample in ["all", "base", "new"]

        if subsample == "all":
            return args

        dataset = args[0]
        labels = set()
        for item in dataset:
            labels.add(item.label)
        labels = list(labels)
        labels.sort()
        n = len(labels)
        # Divide classes into two halves
        m = math.ceil(n / 2)

        print(f"SUBSAMPLE {subsample.upper()} CLASSES!")
        if subsample == "base":
            selected = labels[:m]  # take the first half
        else:
            selected = labels[m:]  # take the second half
        relabeler = {y: y_new for y_new, y in enumerate(selected)}

        output = []
        for dataset in args:
            dataset_new = []
            for item in dataset:
                if item.label not in selected:
                    continue
                item_new = Datum(impath=item.impath, label=relabeler[item.label], classname=item.classname)
                dataset_new.append(item_new)
            output.append(dataset_new)

        return output

    @staticmethod
    def read_classnames(text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, 'r') as f:
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

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(split_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(split_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items

