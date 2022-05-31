#!/usr/bin/env python3
# encoding: utf-8

from monai import transforms, data
from monai.data import load_decathlon_datalist
import glob
import os


def get_loader(args):
    data_dir, batch_size, workers = args.data_dir, args.batch_size, args.workers
    datalist_json = os.path.join(data_dir, args.json_list)

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
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
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
            transforms.Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            transforms.CenterSpatialCropd(keys=["image", "label"], roi_size=roi_size),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

        test_ds = data.Dataset(data=val_files, transform=test_transforms)

        test_loader = data.DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)

        train_ds = data.CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=args.cache_num,
            cache_rate=1.0,
            num_workers=args.workers,
        )

        train_loader = data.DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )

        val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

        val_ds = data.Dataset(data=val_files, transform=val_transforms)

        val_loader = data.DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=True,
        )

        loader = [train_loader, val_loader]
    return loader
