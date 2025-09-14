import torch
import torchvision.transforms as transforms
import os
import cv2
from sklearn.model_selection import train_test_split
from PIL import Image

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, img_root: str, split: str = "train", val_ratio: float = 0.2, random_state: int = 42):

        self.img_root = img_root
        self.split = split
        self.data_infos = []

        # 取得所有 class
        cls_folders = sorted(os.listdir(img_root))
        all_paths, all_labels = [], []

        self.class_to_idx = {cls_f: i for i, cls_f in enumerate(cls_folders)}

        for i, cls_f in enumerate(cls_folders):
            cls_path = os.path.join(img_root, cls_f)
            files = os.listdir(cls_path)
            for file in files:
                img_path = os.path.join(cls_path, file)
                all_paths.append(img_path)
                all_labels.append(i)

        # stratified split 
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            all_paths, all_labels, test_size=val_ratio, stratify=all_labels, random_state=random_state
        )

        if split == "train":
            self.data_infos = [{"path": p, "label": l} for p, l in zip(train_paths, train_labels)]
        elif split == "val":
            self.data_infos = [{"path": p, "label": l} for p, l in zip(val_paths, val_labels)]
        else:
            raise ValueError("split 必須是 'train' 或 'val'")

        # transform
        if split == "train":
            self.trans = transforms.Compose([
                transforms.Resize((256, 256)),                       
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
                transforms.RandomHorizontalFlip(p=0.5),              
                transforms.RandomRotation(degrees=15),               
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),                                                   
                transforms.RandomGrayscale(p=0.1),                   
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.num_cls = len(cls_folders)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, index: int):
        img_path = self.data_infos[index]["path"]
        label = self.data_infos[index]["label"]

        

        img = Image.open(img_path).convert("RGB")  
        data = self.trans(img)


        return data, label
