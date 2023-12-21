import os
import torchvision.transforms as T

from RectifiedFlow_Pytorch.datasets.ImageDataset import ImageDataset
from RectifiedFlow_Pytorch.datasets.lsun import LSUN


def get_train_test_datasets(args, config):
    if config.data.dataset == "LSUN":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        root_dir = os.path.join(args.data_dir, "lsun")
        if config.data.random_flip:
            train_tfm = T.Compose([
                T.Resize(config.data.image_size),
                T.CenterCrop(config.data.image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ])
        else:
            train_tfm = T.Compose([
                T.Resize(config.data.image_size),
                T.CenterCrop(config.data.image_size),
                T.ToTensor(),
            ])
        test_tfm = T.Compose([
            T.Resize(config.data.image_size),
            T.CenterCrop(config.data.image_size),
            T.ToTensor(),
        ])
        train_dataset = LSUN(root_dir, classes=[train_folder], transform=train_tfm)
        test_dataset  = LSUN(root_dir, classes=[val_folder], transform=test_tfm)
    elif config.data.dataset == "LSUN2":
        train_folder = "{}_train".format(config.data.category)
        val_folder = "{}_val".format(config.data.category)
        root_dir = os.path.join(args.data_dir, "lsun")
        if config.data.random_flip:
            train_tfm = T.Compose([T.RandomHorizontalFlip(p=0.5), T.ToTensor()])
        else:
            train_tfm = T.Compose([T.ToTensor()])
        train_dataset = ImageDataset(root_dir, classes=[train_folder], transform=train_tfm)
        test_dataset  = ImageDataset(root_dir, classes=[val_folder], transform=T.Compose([T.ToTensor()]))
    else:
        raise ValueError(f"Unknown config.data.dataset: {config.data.dataset}")

    return train_dataset, test_dataset
