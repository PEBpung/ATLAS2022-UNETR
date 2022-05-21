from monai.networks.nets import UNETR
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism
from monai.transforms import Activations, AsDiscrete, Compose
from monai.inferers import sliding_window_inference
from functools import partial
from train import run_training
from dataloader import get_loader
import gc
import os
import torch
import torch.nn as nn
import argparse
import datetime as dt
import wandb 

gc.collect()

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument("--logdir", default="logs", type=str, help="directory to save the tensorboard logs")
parser.add_argument("--data_dir", default="data/Task500_ATLAS", type=str, help="dataset directory")
parser.add_argument("--json_list", default="dataset.json", type=str, help="dataset json file")
parser.add_argument("--max_epochs", default=200, type=int, help="max number of training epochs")
parser.add_argument("--batch_size", default=8, type=int, help="number of batch size")
parser.add_argument("--optim_lr", default=2e-4, type=float, help="optimization learning rate")
parser.add_argument("--workers", default=14, type=int, help="number of workers")
parser.add_argument("--roi_x", default=128, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=128, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=128, type=int, help="roi size in z direction")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
parser.add_argument("--wandb", action="store_true", help="start wandb")
parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
parser.add_argument("--val_every", default=10, type=int, help="validation frequency")

def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    set_determinism(2022)

    device = torch.device("cuda")

    logdir_name = str(dt.datetime.now().strftime("%y-%m-%d"))
    args.logdir = os.path.join("./logs", logdir_name)
    os.makedirs(args.logdir, exist_ok=True)

    roi_size = [args.roi_x, args.roi_y, args.roi_z]

    model = UNETR(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        img_size=tuple(roi_size),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.0,
    )

    model = nn.DataParallel(model, output_device=0)
    model.to(device)

    loader = get_loader(args)

    if args.wandb:
        wandb.init(config=vars(args), project="TASK02_ATLAS")
        wandb.watch(model)

    epoch = 100
    dice_loss = DiceCELoss(to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    model_inferer = partial(
        sliding_window_inference,
        roi_size=roi_size,
        sw_batch_size=2,
        predictor=model,
        overlap=0.5,
    )

    run_training(
        model=model,
        model_inferer=model_inferer,
        loader=loader,
        optimizer=optimizer,
        loss_func=dice_loss,
        dsc=dice_metric,
        scheduler=scheduler,
        post_pred=post_pred,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
