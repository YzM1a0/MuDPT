import argparse
import torch
import sys
sys.path.append("..")

from dassl.utils import setup_logger, set_random_seed
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
from yacs.config import CfgNode

import warnings
warnings.filterwarnings("ignore")


import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.imagenet_sketch

from trainers import (
    coop,
    cocoop,
    zsclip,
    vpt,
    mpt,
    mudpt,
    umudpt,
    uumudpt,
)


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.dataset_root:
        cfg.DATASET.ROOT = args.dataset_root
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.seed:
        cfg.SEED = args.seed
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer


def extend_cfg(cfg):
    """
    Add new configs variables.
    E.g.
        from yacs.configs import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    cfg.MODEL.BACKBONE.PATH = ""
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.NUM_SHOTS = 16

    # CoOp config
    cfg.TRAINER.COOP = CfgNode()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    # CoCoOp config
    cfg.TRAINER.COCOOP = CfgNode()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COCOOP.CSC = False  # class-specific context
    cfg.TRAINER.COCOOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.VPT = CfgNode()
    cfg.TRAINER.VPT.DEEP_TEXT_N_CTX = 0
    cfg.TRAINER.VPT.DEEP_VISUAL_N_CTX = 0
    cfg.TRAINER.VPT.TEXT_PROMPT_DEPTH = 0
    cfg.TRAINER.VPT.VISUAL_PROMPT_DEPTH = 0  # if depth = 1, set shallow visual prompt
    cfg.TRAINER.VPT.TEXT_CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp

    cfg.TRAINER.MPT = CfgNode()
    cfg.TRAINER.MPT.DEEP_TEXT_N_CTX = 0
    cfg.TRAINER.MPT.DEEP_VISUAL_N_CTX = 0
    cfg.TRAINER.MPT.TEXT_PROMPT_DEPTH = 0
    cfg.TRAINER.MPT.VISUAL_PROMPT_DEPTH = 0
    cfg.TRAINER.MPT.TEXT_CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MPT.PREC = "fp16"  # fp16, fp32, amp

    # MuDPT config
    cfg.TRAINER.MUDPT = CfgNode()
    cfg.TRAINER.MUDPT.N_CTX = 2
    cfg.TRAINER.MUDPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MUDPT.DEEP_PROMPT_DEPTH = 8
    cfg.TRAINER.MUDPT.PREC = "fp16"  # fp16, fp32, amp

    # unified multi-modal deep prompt tuning
    cfg.TRAINER.UMUDPT = CfgNode()
    cfg.TRAINER.UMUDPT.N_CTX = 2
    cfg.TRAINER.UMUDPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.UMUDPT.DEEP_PROMPT_DEPTH = 8
    cfg.TRAINER.UMUDPT.PREC = "fp16"  # fp16, fp32, amp

    # unified multi-modal deep prompt tuning
    cfg.TRAINER.UUMUDPT = CfgNode()
    cfg.TRAINER.UUMUDPT.N_CTX = 2
    cfg.TRAINER.UUMUDPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.UUMUDPT.DEEP_PROMPT_DEPTH = 8
    cfg.TRAINER.UUMUDPT.PREC = "fp16"  # fp16, fp32, amp


def setup_config(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    # 1. From the dataset configs file
    if args.dataset_config:
        cfg.merge_from_file(args.dataset_config)
    # 2. From the method configs file
    if args.trainer_config:
        cfg.merge_from_file(args.trainer_config)
    # 3. From input arguments
    reset_cfg(cfg, args)
    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup_config(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)

    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=str, default="", help="path to dataset")
    parser.add_argument("--output_dir", required=True, type=str, default="", help="output directory")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")

    parser.add_argument("--trainer_config", required=True, type=str, default="", help="path to configs file")
    parser.add_argument("--dataset_config", required=True, type=str, default="", help="path to configs file for dataset setup")
    parser.add_argument("--trainer", type=str, required=True, default="", help="name of trainer")

    parser.add_argument("--eval_only", action="store_true", help="evaluation only")
    parser.add_argument("--model_dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load_epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no_train", action="store_true", help="do not call trainer.train()")

    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="modify configs options using the command-line")

    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")

    args = parser.parse_args()
    main(args)
