import os
import torch
import argparse
import numpy as np

import sys
sys.path.append("..")

from tqdm import tqdm
from clip import clip
from torch.utils.data import DataLoader
from dassl.config import get_cfg_default
from dassl.utils import set_random_seed, setup_logger
from dassl.data.transforms import build_transform
from dassl.data import DatasetWrapper

from datasets.oxford_pets import OxfordPets
from datasets.oxford_flowers import OxfordFlowers
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.dtd import DescribableTextures
from datasets.eurosat import EuroSAT
from datasets.stanford_cars import StanfordCars
from datasets.food101 import Food101
from datasets.sun397 import SUN397
from datasets.caltech101 import Caltech101
from datasets.ucf101 import UCF101
from datasets.imagenet import ImageNet
from datasets.imagenetv2 import ImageNetV2
from datasets.imagenet_sketch import ImageNetSketch
from datasets.imagenet_a import ImageNetA
from datasets.imagenet_r import ImageNetR

from yacs.config import CfgNode as CN


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.backbone_name:
        cfg.MODEL.BACKBONE.NAME = args.backbone_name

    if args.backbone_path:
        cfg.MODEL.BACKBONE.PATH = args.backbone_path


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)
    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    # 3. From input arguments
    reset_cfg(cfg, args)
    cfg.freeze()
    return cfg


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    cfg.MODEL.BACKBONE.PATH = ""
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
    cfg.DATASET.NUM_SHOTS = 16


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print(f"Setup DataLoader for Dataset: {cfg.DATASET.NAME}")
    dataset = eval(cfg.DATASET.NAME)(cfg)

    if args.split == "train":
        dataset_input = dataset.train_x
    elif args.split == "val":
        dataset_input = dataset.val
    else:
        dataset_input = dataset.test

    tfm_train = build_transform(cfg, is_train=False)  # building transform_test
    data_loader = DataLoader(
        DatasetWrapper(cfg, dataset_input, transform=tfm_train, is_train=False),
        batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
        sampler=None,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        drop_last=False,
        pin_memory=(torch.cuda.is_available() and cfg.USE_CUDA),
    )

    print(f"Setup Network, clip backbone: {cfg.MODEL.BACKBONE.NAME}")
    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.PATH, "cuda", jit=False)
    if "vit" in args.config_file:
        clip_model.float()
    clip_model.eval()

    print(f"Start Feature Extractor")
    feature_list = []
    label_list = []
    train_dataiter = iter(data_loader)
    for i in tqdm(range(1, len(train_dataiter) + 1)):
        batch = next(train_dataiter)
        data = batch["img"].cuda()
        feature = clip_model.visual(data)
        feature = feature.cpu()
        for idx in range(len(data)):
            feature_list.append(feature[idx].tolist())
        label_list.extend(batch["label"].tolist())
    save_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.NAME)
    os.makedirs(save_dir, exist_ok=True)
    save_filename = f"{args.split}"
    np.savez(
        os.path.join(save_dir, save_filename),
        feature_list=feature_list,
        label_list=label_list,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output_dir", type=str, default="", help="output directory")
    parser.add_argument("--config_file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset_config_file", type=str, default="", help="path to config file for dataset setup")

    parser.add_argument("--split", type=str, choices=["train", "val", "test"], help="which split")

    parser.add_argument("--backbone_name", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--backbone_path", type=str, default="", help="path of CNN backbone")

    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")

    args = parser.parse_args()

    main(args)