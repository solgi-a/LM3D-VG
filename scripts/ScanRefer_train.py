import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
import torch.nn as nn
import numpy as np
import pickle

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.config import CONF
from experiments.ablation import ablation_config, ablation_hooks  # ABLATION: revision experiments
from models.refnet import RefNet
from scripts.utils.AdamW import AdamW
from scripts.utils.script_utils import set_params_lr_dict

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

#print(sys.path, '<< sys path')

def comp_weight(our_model, weight):
    """Warm-start `our_model` from `weight`, copying only tensors that match by
    name AND shape.

    This is a deliberately permissive load -- the checkpoint and the model need not
    agree -- which is what makes it usable for a warm start across architectures. The
    danger is that permissiveness is silent: a checkpoint sharing two tensors with the
    model loads just as quietly as one sharing all of them, and the run then reports a
    number that looks like fine-tuning but is mostly training from scratch.

    So it reports. `strict=True` on the final load is safe because the dict being
    loaded is built from the model's own state_dict, so nothing can be missing; it
    guards against a key set that changed under us, not against a partial checkpoint.
    """
    our_model_state_dict = our_model.state_dict()
    our_model_state_dict_keys = list(our_model_state_dict.keys())
    weight_keys = list(weight.keys())

    loaded, shape_mismatch = [], []
    for i in weight_keys:
        if i in our_model_state_dict_keys:
            if weight[i].shape == our_model_state_dict[i].shape:
                our_model_state_dict[i] = weight[i]
                loaded.append(i)
            else:
                shape_mismatch.append(i)

    total = len(our_model_state_dict_keys)
    random_init = total - len(loaded)
    print(f"[WARM START] {len(loaded)}/{total} tensor(s) loaded from the checkpoint, "
          f"{random_init} left at random initialisation.")
    if shape_mismatch:
        print(f"[WARM START] {len(shape_mismatch)} tensor(s) matched by name but NOT by "
              f"shape and were skipped, e.g. {shape_mismatch[:3]}")
    if total and random_init / total > 0.5:
        # Past half, "fine-tuning" is not an honest description of the run.
        print(f"[WARM START] WARNING: more than half of this model is randomly "
              f"initialised. The checkpoint does not match this architecture -- check "
              f"--fusion_variant. Training will proceed, but do not describe this run "
              f"as fine-tuned from the checkpoint.")

    our_model.load_state_dict(our_model_state_dict, strict=True)


def get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, split, config, augment, shuffle=True):
    dataset = ablation_hooks.build_dataset(  # ABLATION: was ScannetReferenceDataset(
        args = args,
        scanrefer=scanrefer[split],
        scanrefer_new=scanrefer_new[split],
        scanrefer_all_scene=all_scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max,
        augment=augment,
        shuffle=shuffle
    )
    # ----------------------------------------------------------------------------------
    # Background loading.
    #
    # This used to run with num_workers=0, meaning every __getitem__ executed serially in
    # the main process while the GPU waited. __getitem__ costs ~20 ms -- almost all of it
    # point-cloud work (subsampling, unique, reductions); the GloVe embedding build is
    # ~0.06 ms, i.e. 0.3% -- so a 36 665-item epoch spent ~12 minutes loading data before
    # any training happened.
    #
    # `workers` background processes each prepare batches ahead of the main one, and each
    # holds `prefetch_factor` batches ready, so the loader stays ahead of the GPU instead
    # of blocking it. This is where the speedup is: precomputing the embeddings to disk
    # would address the 0.3%, not the 99.7%.
    #
    # Worker count is bounded by CPU count -- more processes than cores adds contention
    # and RAM for no gain. Each worker is a fork holding its own copy of the dataset, so
    # RAM grows with worker count; that is affordable only because LAZY_LANG_DATA keeps
    # the per-process language data at ~1.8 GB instead of ~29 GB.
    # ----------------------------------------------------------------------------------
    workers = getattr(args, "num_workers", None)
    if workers is None:
        workers = min(4, max(0, (os.cpu_count() or 1) - 1))

    loader_kwargs = {}
    if workers > 0:
        loader_kwargs.update(
            num_workers=workers,
            prefetch_factor=getattr(args, "prefetch_factor", 3),
            persistent_workers=True,     # don't respawn every epoch
            pin_memory=torch.cuda.is_available(),
        )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                            **loader_kwargs)

    print(f"[loader] split={split} batch={args.batch_size} workers={workers}"
          + (f" prefetch={loader_kwargs['prefetch_factor']}"
             f" pin_memory={loader_kwargs['pin_memory']}" if workers > 0 else
             "  (serial loading -- set --num_workers > 0 to overlap it with training)"))

    return dataset, dataloader

def get_model(args):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = ablation_hooks.build_model(  # ABLATION: was RefNet(
        args=args,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        dataset_config=DC
    )

    # trainable model
    if args.use_pretrained and not ablation_hooks.skip_pretrained_detector():  # ABLATION
        # load model
        if args.detector == "VN":
            
            print("\nloading pretrained VoteNet...")

            pretrained_model = RefNet(
                args=args,
                num_class=DC.num_class,
                num_heading_bin=DC.num_heading_bin,
                num_size_cluster=DC.num_size_cluster,
                mean_size_arr=DC.mean_size_arr,
                input_feature_dim=input_channels,
                num_proposal=args.num_proposals,
                use_lang_classifier=(not args.no_lang_cls),
                use_bidir=args.use_bidir,
                no_reference=args.no_reference,
                dataset_config=DC
            )

            pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_criteria_25.pth")
            pretrained_model.load_state_dict(torch.load(pretrained_path, weights_only=False), strict=False)

            # mount
            model.backbone_net = pretrained_model.backbone_net
            model.vgen = pretrained_model.vgen
            model.proposal = pretrained_model.proposal

            if args.no_detection:
                # freeze pointnet++ backbone
                for param in model.backbone_net.parameters():
                    param.requires_grad = False

                # freeze voting
                for param in model.vgen.parameters():
                    param.requires_grad = False

                # freeze detector
                for param in model.proposal.parameters():
                    param.requires_grad = False
                    
    # to CUDA
    model = model.cuda()

    return model


def get_num_params(model):
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataloader):

    model = get_model(args)

    # different lr for various modules.
    weight_dict = {
                'detr': {'lr': args.detr_lr},
                'lang': {'lr': args.lang_lr},
                'match': {'lr': args.match_lr},
            }

    # scheduler parameters for training solely the detection pipeline
    lr_decay_rate = 0.1 if args.no_reference else None
    bn_decay_step = 20 if args.no_reference else None
    bn_decay_rate = 0.5 if args.no_reference else None

    if args.detector == 'GF':
        lr_decay_step = [280, 340] if args.no_reference else None
    else:
        lr_decay_step = [80, 120, 160] if args.no_reference else None

    if args.coslr:
        lr_decay_step = {
            'type': 'cosine',
            'T_max': args.epoch,
            'eta_min': 1e-6,
        }

    params = set_params_lr_dict(model, base_lr=args.lr, weight_decay=args.wd, weight_dict=weight_dict)
    optimizer = AdamW(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)

    # lr scheduler
    if lr_decay_step:
        if isinstance(lr_decay_step, list):
            lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
        elif isinstance(lr_decay_step, dict):
            if lr_decay_step['type'] != 'cosine':
                raise NotImplementedError('lr dict type should be cosine (other not implemented)')
            print(lr_decay_step, '<< lr_decay_step dict', flush=True)  # TODO
            config = lr_decay_step
            config['optimizer'] = optimizer
            config.pop('type')
            lr_scheduler = CosineAnnealingLR(**config)
        else:
            lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
    else:
        lr_scheduler = None

    if args.use_checkpoint:
        print("\nloading checkpoint {}...".format(args.use_checkpoint))

        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

        #stamp = args.use_checkpoint
        #root = os.path.join(CONF.PATH.OUTPUT, stamp)
        #checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        #model.load_state_dict(checkpoint["model_state_dict"],strict=False)
        #optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        best_model_weights = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "model_criteria_25.pth"), weights_only=False)
        comp_weight(model,best_model_weights)
        #model.load_state_dict(best_model_weights, strict=False)

        if args.GF_path:
            print("\nloading group free weights...")
            detector_weights = torch.load(args.GF_path, weights_only=False)
            #comp_weight(model.detector, detector_weights)
            model.detector.load_state_dict(detector_weights['model'], strict=True)
            
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)


    #print('LR&BN_DECAY', lr_decay_step, lr_decay_rate, bn_decay_step, bn_decay_rate, flush=True)

    solver = Solver(args=args,
        model=model,
        config=DC,
        dataloader=dataloader,
        optimizer=optimizer,
        stamp=stamp,
        val_step=args.val_step,
        detection=not args.no_detection,
        reference=not args.no_reference,
        use_lang_classifier=not args.no_lang_cls,
        lr_decay_step=lr_decay_step,
        lr_decay_rate=lr_decay_rate,
        bn_decay_step=bn_decay_step,
        bn_decay_rate=bn_decay_rate,
        lr_scheduler=lr_scheduler
    )

    num_params = get_num_params(model)
    num_params_lang = get_num_params(model.lang)
    num_params_match = get_num_params(model.match)

    if args.detector == "GF":
        num_params_detector = get_num_params(model.detector)
    else:
        num_params_detector = num_params - num_params_match - num_params_lang


    print("\n")
    print(f"num_params: {num_params}")
    print(f"num_params_detector: {num_params_detector}")
    print(f"num_params_lang: {num_params_lang}")
    print(f"num_params_match: {num_params_match}")

    return solver, num_params, root


def save_info(args, root, num_params, train_dataset, val_dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value

    info["num_train"] = len(train_dataset)
    info["num_val"] = len(val_dataset)
    info["num_train_scenes"] = len(train_dataset.scene_list)
    info["num_val_scenes"] = len(val_dataset.scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def get_scannet_scene_list(split):
    scene_list = sorted(
        [line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list


def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, lang_num_max):
    if args.no_reference:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1:
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes

        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        scanrefer_train_new = []
        scanrefer_train_new_scene = []
        scene_id = ""
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_train_new_scene) > 0:
                        scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                if len(scanrefer_train_new_scene) >= lang_num_max:
                    scanrefer_train_new.append(scanrefer_train_new_scene)
                    scanrefer_train_new_scene = []
                scanrefer_train_new_scene.append(data)
                """
                if data["scene_id"] not in scanrefer_train_new:
                    scanrefer_train_new[data["scene_id"]] = []
                scanrefer_train_new[data["scene_id"]].append(data)
                """
        scanrefer_train_new.append(scanrefer_train_new_scene)

        new_scanrefer_val = scanrefer_val
        scanrefer_val_new = []
        scanrefer_val_new_scene = []
        scene_id = ""
        for data in scanrefer_val:
            # if data["scene_id"] not in scanrefer_val_new:
            # scanrefer_val_new[data["scene_id"]] = []
            # scanrefer_val_new[data["scene_id"]].append(data)
            if scene_id != data["scene_id"]:
                scene_id = data["scene_id"]
                if len(scanrefer_val_new_scene) > 0:
                    scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            if len(scanrefer_val_new_scene) >= lang_num_max:
                scanrefer_val_new.append(scanrefer_val_new_scene)
                scanrefer_val_new_scene = []
            scanrefer_val_new_scene.append(data)
        scanrefer_val_new.append(scanrefer_val_new_scene)

    print("\nscanrefer_train_new", len(scanrefer_train_new), len(scanrefer_val_new), len(scanrefer_train_new[0]))  # 4819 1253 8
    sum = 0
    for i in range(len(scanrefer_train_new)):
        sum += len(scanrefer_train_new[i])
    print("training sample numbers", sum)  # 36665
    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples\n".format(len(new_scanrefer_train), len(new_scanrefer_val)))  # 36665 9508

    return new_scanrefer_train, new_scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new


def train(args):
    # init training dataset
    print("\npreparing data...")
    scanrefer_train, scanrefer_val, all_scene_list, scanrefer_train_new, scanrefer_val_new = get_scanrefer(
        SCANREFER_TRAIN, SCANREFER_VAL, args.num_scenes, args.lang_num_max)
    scanrefer = {
        "train": scanrefer_train,
        "val": scanrefer_val
    }
    scanrefer_new = {
        "train": scanrefer_train_new,
        "val": scanrefer_val_new
    }

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "train", DC, augment=True)
    val_dataset, val_dataloader = get_dataloader(args, scanrefer, scanrefer_new, all_scene_list, "val", DC, augment=False)
    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader
    }

    print("\ninitializing...")
    solver, num_params, root = get_solver(args, dataloader)

    print("\nStart training...\n")
    save_info(args, root, num_params, train_dataset, val_dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, help="tag for the training, e.g. cuda_wl", default="")
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--batch_size", type=int, help="batch size", default=14)
    parser.add_argument("--epoch", type=int, help="number of epochs", default=50)
    parser.add_argument("--verbose", type=int, help="iterations of showing verbose", default=10)
    parser.add_argument("--val_step", type=int, help="iterations of validating", default=5000)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--wd", type=float, help="weight decay", default=1e-5)
    parser.add_argument("--lang_num_max", type=int, help="lang num max", default=32)
    parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
    parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
    parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--coslr", action='store_true', help="cosine learning rate")
    parser.add_argument("--amsgrad", action='store_true', help="optimizer with amsgrad")
    parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
    parser.add_argument("--no_augment", action="store_true", help="Do NOT use augment on trainingset (not used)")
    parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
    parser.add_argument("--no_detection", action="store_true", help="Do NOT train the detection module.")
    parser.add_argument("--no_reference", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--detection", action="store_true", help="Do NOT train the localization module.")
    parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
    parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
    parser.add_argument("--use_bidir", action="store_true", help="Use bi-directional GRU.")
    parser.add_argument("--use_pretrained", type=str, help="Specify the folder name containing the pretrained detection module.")
    parser.add_argument("--use_checkpoint", type=str, help="Specify the checkpoint root", default="")
    parser.add_argument("--no_warm_start", action="store_true",
                        help="Train from random initialisation: do not load fusion "
                             "weights from a checkpoint. Overrides --use_checkpoint.")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader worker processes. 0 loads serially in the main "
                             "process (the old behaviour). Default: CPU count - 1, "
                             "capped at 4.")
    parser.add_argument("--prefetch_factor", type=int, default=3,
                        help="Batches each worker keeps ready in RAM. Ignored when "
                             "--num_workers 0.")

    #----------------------------------------------------------------------------------------------------------------------------------

    parser.add_argument("--lang_input", type=str, default='glove+parse')
    parser.add_argument("--detector", type=str, default='GF', choices=["VN", "GF"])
    parser.add_argument("--GF_path", type=str)
    parser.add_argument("--lang_lr", type=float, help="lang module learning rate", default=0.0005)
    parser.add_argument("--match_lr", type=float, help="match module learning rate", default=0.0005)
    parser.add_argument("--detr_lr", type=float, help="match module learning rate", default=0.0001)

    #----------------------------------------------------------------------------------------------------------------------------------
    
    parser.add_argument('--width', default=1, type=int, help='backbone width')
    parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--sampling', default='kps', type=str, help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--nhead', default=8, type=int, help='multi-head number')
    parser.add_argument('--num_decoder_layers', default=6, type=int, help='number of decoder layers')
    parser.add_argument('--dim_feedforward', default=2048, type=int, help='dim_feedforward')
    parser.add_argument('--transformer_dropout', default=0.1, type=float, help='transformer_dropout')
    parser.add_argument('--transformer_activation', default='relu', type=str, help='transformer_activation')
    parser.add_argument('--self_position_embedding', default='loc_learned', type=str,
                        help='position_embedding in self attention (none, xyz_learned, loc_learned)')
    parser.add_argument('--cross_position_embedding', default='xyz_learned', type=str,
                        help='position embedding in cross attention (none, xyz_learned)')

    # Loss
    parser.add_argument('--query_points_generator_loss_coef', default=0.8, type=float)
    parser.add_argument('--obj_loss_coef', default=0.1, type=float, help='Loss weight for objectness loss')
    parser.add_argument('--box_loss_coef', default=1, type=float, help='Loss weight for box loss')
    parser.add_argument('--sem_cls_loss_coef', default=0.1, type=float, help='Loss weight for classification loss')
    parser.add_argument('--center_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--center_delta', default=0.04, type=float, help='delta for smoothl1 loss in center loss')
    parser.add_argument('--size_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--size_delta', default=0.111111111111, type=float, help='delta for smoothl1 loss in size loss')
    parser.add_argument('--heading_loss_type', default='smoothl1', type=str, help='(smoothl1, l1)')
    parser.add_argument('--heading_delta', default=1.0, type=float, help='delta for smoothl1 loss in heading loss')
    parser.add_argument('--query_points_obj_topk', default=4, type=int, help='query_points_obj_topk')
    parser.add_argument('--size_cls_agnostic', action='store_true', help='Use class-agnostic size prediction.')


    # Training
    parser.add_argument('--start_epoch', type=int, default=1, help='Epoch to run [default: 1]')
    #parser.add_argument('--max_epoch', type=int, default=400, help='Epoch to run [default: 180]')
    parser.add_argument('--optimizer', type=str, default='adamW', help='optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Optimization L2 weight decay [default: 0.0005]')
    parser.add_argument('--learning_rate', type=float, default=0.004, help='Initial learning rate for all except decoder [default: 0.004]')
    parser.add_argument('--decoder_learning_rate', type=float, default=0.0004, help='Initial learning rate for decoder [default: 0.0004]')
    #parser.add_argument('--lr-scheduler', type=str, default='step', choices=["step", "cosine"], help="learning rate scheduler")
    #parser.add_argument('--warmup-epoch', type=int, default=-1, help='warmup epoch')
    #parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    #parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340], nargs='+', help='for step scheduler. where to decay lr, can be a list')
    #parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1, help='Default bn momeuntum')
    parser.add_argument('--syncbn', action='store_true', help='whether to use sync bn')

    # io
    parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')

    # others
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5], nargs='+', help='A list of AP IoU thresholds [default: 0.25,0.5]')

    ablation_config.add_arguments(parser)  # ABLATION: optional CLI overrides

    args = parser.parse_args()
    args.detection = args.no_reference

    # ----------------------------------------------------------------------------------
    # Defaults for a bare `python scripts/ScanRefer_train.py` with no flags.
    #
    # This block used to run unconditionally and assign to args *after* parsing, which
    # silently discarded every value passed on the command line: an ablation runner
    # asking for --epoch 50 --batch_size 4 got 100 and 8 regardless, and every arm
    # trained under identical settings no matter what it requested. It also pointed
    # --use_checkpoint at a folder that does not exist in outputs/, so any run that
    # reached it died on a missing file.
    #
    # Now it only fills in what the user did NOT ask for. `_passed` reads sys.argv
    # rather than comparing against argparse defaults, because "the user passed the
    # default value explicitly" and "the user passed nothing" must not be conflated --
    # --batch_size 8 is a real choice, not an absence.
    # ----------------------------------------------------------------------------------
    def _passed(*flags):
        return any(a == f or a.startswith(f + "=") for a in sys.argv[1:] for f in flags)

    APPLY_DEFAULTS = True          # set False to run on argparse defaults alone

    if APPLY_DEFAULTS:
        if not _passed("--detector"):      args.detector = 'VN'
        if not _passed("--use_color"):     args.use_color = True
        if not _passed("--use_normal"):    args.use_normal = True
        if not _passed("--batch_size"):    args.batch_size = 8
        if not _passed("--lang_num_max"):  args.lang_num_max = 32
        if not _passed("--epoch"):         args.epoch = 50
        if not _passed("--lr"):            args.lr = 0.002
        if not _passed("--coslr"):         args.coslr = True
        if not _passed("--tag"):           args.tag = '3DVG-GF'
        if not _passed("--val_step"):      args.val_step = 5000
        # Warm start. The 2024-12-18 run is the only checkpoint present in outputs/;
        # the folder previously named here does not exist. Under the frozen-detector
        # protocol --use_pretrained is ignored (ablation_hooks.skip_pretrained_detector),
        # so the fusion weights come in through --use_checkpoint.
        if not _passed("--use_checkpoint", "--no_warm_start"):
            args.use_checkpoint = '2024-12-18_20-40-38_3DVG-FIXED'
        if not _passed("--use_pretrained"):
            args.use_pretrained = '2024-12-18_20-40-38_3DVG-FIXED'

    if getattr(args, "no_warm_start", False):
        args.use_checkpoint = ""

    args = ablation_config.apply(args)          # ABLATION: fold in flags, then report them
    print("\n" + ablation_config.describe() + "\n")

    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)

