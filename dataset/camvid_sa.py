import os
import torch
import torch.utils.data as data

import torchvision.transforms as transforms

from PIL import Image

from dataset.transform_sa import *

import numpy as np
np.random.seed(233)

import json
import cv2
from utils import get_boxes_from_mask, init_point_sampling, train_transforms, TransformSam
from util.stream import segment_paths_by_prefix

class CamVidSA(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """

    def __init__(self,
                 root_dir,
                 mode='train',
                 image_size=1024,
                 ref_gap=5,
                 mask_num = 5, 
                 point_num = 1, 
                 sub_set = None):

        self.root_dir = root_dir
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        # self.transform = transform
        # self.label_transform = label_transform
        self.ref_gap = ref_gap

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
            ])
        
        self.train_aug = True
        self.trans_train_color = pairColorJitter(
            brightness = 0.5,
            contrast = 0.5,
            saturation = 0.5
        )

        if sub_set is None:
            dataset = json.load(open(os.path.join(root_dir, f'image2label_{mode}.json'), "r"))
        else:
            dataset = json.load(open(os.path.join(root_dir, f'train_sub_jsons/image2label_{mode}_{sub_set}.json'), "r"))
        
        self.image_paths = list(dataset.keys())
        self.label_paths = list(dataset.values())
        self.mask_num = mask_num
        self.point_num = point_num
        self.image_size = image_size

        try:
            ignore_name = 'Seq05VD_f00000'
            bool_list = [(ignore_name in x[0]) for x in self.label_paths]
            idx = np.where(bool_list)[0][0]
            del[self.label_paths[idx]]
            bool_list = [(ignore_name in x) for x in self.image_paths]
            idx = np.where(bool_list)[0][0]
            del[self.image_paths[idx]]
        except:
            pass

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        image_input = {}
        try:
            data_path = self.image_paths[index]
            image = cv2.imread(self.image_paths[index])
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        if self.train_aug:
            transforms = train_transforms(self.image_size, h, w)
        else:
            transforms = TransformSam(self.image_size, h, w)

        masks_list = []
        boxes_list = []
        point_coords_list, point_labels_list = [], []
        mask_path = random.choices(self.label_paths[index], k=self.mask_num)
        for m in mask_path:
            pre_mask = cv2.imread(m, 0)
            if pre_mask.max() == 255:
                pre_mask = pre_mask / 255
                if not self.train_aug:
                    pre_mask = pre_mask.astype(np.uint8)

            if self.train_aug:
                augments = transforms(image=image, mask=pre_mask)
                image_cv2, mask_cv2 = augments['image'], augments['mask']
            else:
                image_cv2 = transforms(image)
                mask_cv2 = transforms(pre_mask)

            image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
            if self.train_aug:
                image_pil, _ = self.trans_train_color(image_pil, image_pil.copy())

            image_tensor = self.to_tensor(image_pil)
            mask_tensor = torch.from_numpy(mask_cv2).to(torch.int64).squeeze(0)

            boxes = get_boxes_from_mask(mask_tensor)
            point_coords, point_label = init_point_sampling(mask_tensor, self.point_num)

            masks_list.append(mask_tensor)
            boxes_list.append(boxes)
            point_coords_list.append(point_coords)
            point_labels_list.append(point_label)

        mask = torch.stack(masks_list, dim=0)
        boxes = torch.stack(boxes_list, dim=0)
        point_coords = torch.stack(point_coords_list, dim=0)
        point_labels = torch.stack(point_labels_list, dim=0)

        image_input["image"] = image_tensor.unsqueeze(0)
        image_input["label"] = mask.unsqueeze(1)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["image_path"] = data_path

        return image_input

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.image_paths)

class CamVidSATesting(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.
    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.
    """

    def __init__(self,
                 root_dir,
                 mode='val',
                 image_size=1024,
                 ref_gap=5,
                 point_num = 1, 
                 stream = False,
                 sub_set = None
                ):

        self.root_dir = root_dir
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        # self.transform = transform
        # self.label_transform = label_transform
        self.ref_gap = ref_gap

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.39068785, 0.40521392, 0.41434407), (0.29652068, 0.30514979, 0.30080369)),
            ])
        
        if sub_set is None:
            dataset = json.load(open(os.path.join(root_dir, f'label2image_{mode}.json'), "r"))
        else:
            dataset = json.load(open(os.path.join(root_dir, "val_sub_jsons", f'label2image_{mode}_{sub_set}.json'), "r"))
        
        self.image_paths = list(dataset.values())
        self.label_paths = list(dataset.keys())
        self.point_num = point_num
        self.image_size = image_size

        if stream:
            self.image_paths, self.label_paths, self.segments = segment_paths_by_prefix(self.image_paths, self.prefix_func, self.label_paths)

    def prefix_func(self, img_path):
        return img_path.split('/')[-1][:6]
        
    
    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset
        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.
        """
        image_input = {}

        try:
            data_path = self.image_paths[index]
            image = cv2.imread(self.image_paths[index])
        except:
            print(self.image_paths[index])

        h, w, _ = image.shape
        transforms = TransformSam(self.image_size, h, w)

        mask_path = self.label_paths[index]
    
        pre_mask = cv2.imread(mask_path, 0)
        if pre_mask.max() == 255:
            pre_mask = pre_mask / 255
            pre_mask = pre_mask.astype(np.uint8)

        image_cv2 = transforms(image)
        mask_cv2 = transforms(pre_mask)
        
        image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
        image_tensor = self.to_tensor(image_pil)
        mask_tensor = torch.from_numpy(mask_cv2).to(torch.int64).squeeze(0)


        boxes = get_boxes_from_mask(mask_tensor)
        point_coords, point_labels = init_point_sampling(mask_tensor, self.point_num)

        image_input["image"] = image_tensor
        image_input["label"] = mask_tensor.unsqueeze(0)
        image_input["boxes"] = boxes
        image_input["point_coords"] = point_coords
        image_input["point_labels"] = point_labels
        image_input["original_size"] = (h, w)
        image_input["ori_label"] = torch.tensor(pre_mask).unsqueeze(0)
        image_input["image_path"] = data_path
        image_input["label_name"] = os.path.basename(mask_path)

        return image_input

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.image_paths)

if __name__ == "__main__":
    # train_dataset = CamVidSA(root_dir='./data/CamVid', image_size=1024, mode='train', point_num=1)
    # data_item = train_dataset[0]

    test_dataset = CamVidSATesting(root_dir='./data/CamVid', image_size=1024, mode='val', point_num=1)
    data_item = test_dataset[2069]

    pass
