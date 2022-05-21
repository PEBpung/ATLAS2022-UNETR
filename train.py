from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torch
import os
import wandb

def save_checkpoint(args, model, mode, device):
    rand_input = torch.rand(2, args.in_channels, args.roi_x, args.roi_y, args.roi_z).to(device)
    traced_net = torch.jit.trace(model.module, rand_input)
    filename = f"{mode}_model_jit.pt"
    filename = os.path.join(args.logdir, filename)
    traced_net.save(filename)
    del traced_net
    del rand_input
    print("Saving checkpoint", filename)

def run_training(model, model_inferer, loader, optimizer, loss_func, dsc, scheduler, post_pred, device, args):
    max_epochs = 200
    best_metric = -1
    best_metric_epoch = -1
    metric_count = 0
    metric_sum = 0.0
    epoch_loss_values = []
    metric_values = []
    scaler = GradScaler()

    torch.cuda.empty_cache()

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        step = 0
        train_loader = tqdm(loader[0], bar_format="{l_bar}{bar:20}{r_bar}{bar:-10b}")
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            
            optimizer.zero_grad()
            with autocast(enabled=args.amp):
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

            if args.amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        if args.wandb:
            wandb.log({"TRAIN_LOSS": epoch_loss}, step=epoch)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

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
                    val_outputs = post_pred(val_outputs)

                    # compute overall mean dice
                    dsc(y_pred=val_outputs, y=val_labels)
                    value, not_nans = dsc.aggregate()
                    not_nans = not_nans.mean().item()
                    metric_count += not_nans
                    metric_sum += value.mean().item() * not_nans

                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    mode = "best"
                    save_checkpoint(args, model, mode, device)

                    best_metric = metric
                    best_metric_epoch = epoch + 1

                if args.wandb:
                    wandb.log({"VALID_DICE": metric}, step=epoch)

                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )
        
        scheduler.step()

    mode = "final"
    save_checkpoint(args, model, mode, device)

    print(
        f"train completed, best_metric: {best_metric:.4f}"
        f" at epoch: {best_metric_epoch}"
    )
