from ast import arg
import os
import torch
import numpy as np
from monai.transforms import Activations, AsDiscrete, Compose, KeepLargestConnectedComponent
from monai.inferers import sliding_window_inference
from dataloader import get_loader
from utils import resample_3d, dice
import nibabel as nib
import argparse

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./logs/22-05-27', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='data/Task500_ATLAS', type=str, help='dataset directory')
parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
parser.add_argument('--exp_name', default='Task500_ATLAS', type=str, help='experiment name')
parser.add_argument('--json_list', default='dataset.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='best_model_jit.pt', type=str, help='pretrained model name')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=1, type=int, help='number of output channels')
parser.add_argument('--roi_x', default=128, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=128, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=128, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--workers', default=10, type=int, help='number of workers')
parser.add_argument("--cache_num", default=100, type=int, help="seed number")

def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = './outputs/'+args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name

    roi_size = (args.roi_x, args.roi_y, args.roi_z)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = torch.jit.load(pretrained_pth)
    model.eval()
    model.to(device)
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])
    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch['label_meta_dict']['affine'][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   roi_size,
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap,
                                                   mode="gaussian")
            
            val_outputs = post_pred(val_outputs)
            val_outputs = [
                KeepLargestConnectedComponent(applied_labels=[1], connectivity=1)(i)
                for i in val_outputs
            ][0]
            val_outputs = val_outputs.cpu().numpy().astype(np.uint8)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)

            organ_Dice = dice(val_outputs, val_labels)
            print("Mean Organ Dice: {}".format(organ_Dice))
            dice_list_case.append(organ_Dice)
            nib.save(nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                     os.path.join(output_directory, img_name))

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':
    main()