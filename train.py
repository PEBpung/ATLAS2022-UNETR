from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import os
import wandb
import numpy as np


def save_checkpoint(args, model, mode, device):
    rand_input = torch.rand(2, args.in_channels, args.roi_x, args.roi_y, args.roi_z).to(device)
    traced_net = torch.jit.trace(model.module, rand_input)
    filename = f"{mode}_model_jit.pt"
    filename = os.path.join(args.logdir, filename)
    traced_net.save(filename)
    del traced_net
    del rand_input
    print("Saving checkpoint", filename)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def run_training(model, model_inferer, loader, optimizer, loss_func, dsc, scheduler, post_pred, device, args):
    best_metric = -1
    best_metric_epoch = -1
    scaler = GradScaler()

    torch.cuda.empty_cache()

    for epoch in range(args.max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        max_norm = 5
        train_loader = tqdm(loader[0], bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            with autocast(enabled=args.amp):
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

            post_outputs = post_pred(outputs)
            dsc(y_pred=post_outputs, y=labels)

        epoch_loss /= step
        metric = dsc.aggregate().item()
        if args.wandb:
            wandb.log({"TRAIN_LOSS": epoch_loss}, step=epoch)
            wandb.log({"TRAIN_DICE": metric}, step=epoch)
        print(f"epoch {epoch + 1} average train loss: {epoch_loss:.4f}", f"\nmean train dice: {metric:.4f}")

        dsc.reset()

        for param in model.parameters():
            param.grad = None

        if (epoch + 1) % args.val_every == 0:
            model.eval()
            with torch.no_grad():
                valid_loader = tqdm(loader[1], bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")

                for val_data in valid_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    with autocast(enabled=args.amp):
                        val_outputs = model_inferer(val_inputs)

                    # compute overall mean dice
                    val_outputs = post_pred(val_outputs)
                    dsc(y_pred=val_outputs, y=val_labels)

                val_metric = dsc.aggregate().item()
                if val_metric > best_metric:
                    mode = "best"
                    save_checkpoint(args, model, mode, device)

                    best_metric = val_metric
                    best_metric_epoch = epoch + 1

                if args.wandb:
                    wandb.log({"VALID_DICE": val_metric}, step=epoch)

                print(
                    f"current epoch: {epoch + 1} current mean val dice: {val_metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

            dsc.reset()
        # scheduler.step()

    mode = "final"
    save_checkpoint(args, model, mode, device)

    print(f"train completed, best_metric: {best_metric:.4f}" f" at epoch: {best_metric_epoch}")
