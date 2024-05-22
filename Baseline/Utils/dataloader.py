import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, args, csv_file, root_dir, img_type = "_image_color_norm", transform=None, train=True, num_classes=5):
        """
        Args:
            csv_file (string): Path to the CSV file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): If True, dataset includes labels; if False, it does not.
            num_classes (int): Number of classes for one-hot encoding.
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transforms.Compose([
                transforms.Resize((512, 512)),  # Resize images to 128x128 pixels
                transforms.ToTensor(),  # Convert image to PyTorch tensor
                # transforms.Normalize(mean=[0.425753653049469, 0.29737451672554016, 0.21293757855892181], 
                #                      std=[0.27670302987098694, 0.20240527391433716, 0.1686241775751114])  # Normalize based on ImageNet standards
            ])
        # self.transform = transform
        self.train = train
        self.img_type = img_type
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get the image file name and path
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 1] + self.img_type +'.png')  # Assuming images are in .jpg format
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.train:
            # Get the binary label
            label = self.data.iloc[idx, 2]  # Assuming the binary label is in the forth column
            label_onehot = np.zeros(5)
            label_onehot[label] = 1
            # Convert label to one-hot encoding
            # label = F.one_hot(label, num_classes=self.num_classes).float()
            return image, label_onehot, label
        else:
            return image


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    input_size = args.input_size 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if input_size >= 384:  
            t.append(
            transforms.Resize((input_size, input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)