from dataset.mask import get_mask_func
from dataset.transform import DataTransform
from torch.utils.data import DataLoader
import os
def create_datasets(args):
    MaskFunc = get_mask_func(args.mask_style)
    train_mask = MaskFunc(args.cf, args.acc)
    dev_mask = MaskFunc(args.cf, args.acc)
    if args.dset == 'fastmri':
        print('using fastmri dataset')
        # fast mri dataset
        from dataset.fastmri_data import MRI_DATA
        train_data = MRI_DATA(
            root=os.path.join(args.data_path, 'singlecoil_train'),
            transform=DataTransform(train_mask, max_frames=args.mf, gap=args.gap),
            sample_rate=args.sample_rate,
            resolution=args.resolution,
            acquisition=args.acquisition
        )
        dev_data = MRI_DATA(
            root=os.path.join(args.data_path, 'singlecoil_val'),
            transform=DataTransform(dev_mask, use_seed=True, max_frames=-1, gap=args.gap),
            sample_rate=args.sample_rate,
            resolution=args.resolution,
            acquisition=args.acquisition
        )
    else:
        # calgary dataset
        from dataset.calgary_data import MRI_DATA
        train_data = MRI_DATA(
            root=os.path.join(args.data_path, 'train'),
            transform=DataTransform(train_mask, max_frames=args.mf, gap=args.gap),
        )
        dev_data = MRI_DATA(
            root=os.path.join(args.data_path, 'val'),
            transform=DataTransform(dev_mask, use_seed=True, max_frames=-1, gap=args.gap),
            train=False,
            gap=args.gap
        )
    return dev_data, train_data

def create_train_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, 4, 2)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
    )
    data_loaders = dict(
        train=train_loader,
        dev=dev_loader,
        display=display_loader
    )
    return data_loaders

def create_test_loader(args):
    MaskFunc = get_mask_func(args.mask_style)
    train_mask = MaskFunc(args.cf, args.acc, args.same)
    try: 
        dset = args.dset
        # suing fast mri dataset
        from dataset.fastmri_data import MRI_DATA
        print('using fastmri dataset')
        dataset = MRI_DATA(
            root=os.path.join(args.data_path, 'singlecoil_test'),
            transform=DataTransform(train_mask, max_frames=-1, gap=args.gap),
            sample_rate=1,
            resolution=args.resolution, 
            acquisition=args.acquisition
        )
    except:
        from dataset.calgary_data import MRI_DATA
        print('using calgary dataset')
        dataset = MRI_DATA(
            root=os.path.join(args.data_path, 'test'),
            transform=DataTransform(train_mask, use_seed=True, max_frames=-1, gap=args.gap),
        )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=False,
    )
    return data_loader