#!/usr/bin/env python3
# encoding: utf-8

from monai import transforms, data
import glob
import os


def get_loader(args):
    data_dir, batch_size, workers =  args.data_dir, args.batch_size, args.workers

    roi_size = tuple([args.roi_x, args.roi_y, args.roi_z])
    pixdim = (1.0, 1.0, 1.0)

    train_transforms = transforms.Compose(
        [
            # load 4 Nifti images and stack them together
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            transforms.RandSpatialCropd(keys=["image", "label"], roi_size=roi_size, random_size=False),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=pixdim,
                mode=("bilinear", "nearest"),
            ),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.EnsureTyped(keys=["image", "label"]),
        ]
    )

    train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
    data_dicts = [
        {"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)
    ]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]

    train_ds = data.Dataset(data=train_files, transform=train_transforms)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_ds = data.Dataset(data=val_files, transform=val_transforms)

    val_loader = data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
    )

    loader = [train_loader, val_loader]
    return loader
