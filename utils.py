import cv2
import albumentations as A
import torch
import numpy as np
from torch.nn import functional as F
from skimage.measure import label, regionprops
import torch.nn as nn
import logging
import os
from torchvision.transforms.functional import resize, to_pil_image

def get_total_grad_norm(parameters, norm_type=2):
    
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    return total_norm


def get_boxes_from_mask(mask, box_num=1, std = 0.1, max_pixel = 5):
    """
    Args:
        mask: Mask, can be a torch.Tensor or a numpy array of binary mask.
        box_num: Number of bounding boxes, default is 1.
        std: Standard deviation of the noise, default is 0.1.
        max_pixel: Maximum noise pixel value, default is 5.
    Returns:
        noise_boxes: Bounding boxes after noise perturbation, returned as a torch.Tensor.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
    label_img = label(mask)
    regions = regionprops(label_img)

    # Iterate through all regions and get the bounding box coordinates
    boxes = [tuple(region.bbox) for region in regions]

    # If the generated number of boxes is greater than the number of categories,
    # sort them by region area and select the top n regions
    if len(boxes) >= box_num:
        sorted_regions = sorted(regions, key=lambda x: x.area, reverse=True)[:box_num]
        boxes = [tuple(region.bbox) for region in sorted_regions]

    # If the generated number of boxes is less than the number of categories,
    # duplicate the existing boxes
    elif len(boxes) < box_num:
        num_duplicates = box_num - len(boxes)
        boxes += [boxes[i % len(boxes)] for i in range(num_duplicates)]

    # Perturb each bounding box with noise
    noise_boxes = []
    for box in boxes:
        y0, x0,  y1, x1  = box
        width, height = abs(x1 - x0), abs(y1 - y0)
        # Calculate the standard deviation and maximum noise value
        noise_std = min(width, height) * std
        max_noise = min(max_pixel, int(noise_std * 5))
         # Add random noise to each coordinate
        noise_x = np.random.randint(-max_noise, max_noise)
        noise_y = np.random.randint(-max_noise, max_noise)
        x0, y0 = x0 + noise_x, y0 + noise_y
        x1, y1 = x1 + noise_x, y1 + noise_y
        noise_boxes.append((x0, y0, x1, y1))
    return torch.as_tensor(noise_boxes, dtype=torch.float)


def select_random_points(pr, gt, point_num = 9):
    """
    Selects random points from the predicted and ground truth masks and assigns labels to them.
    Args:
        pred (torch.Tensor): Predicted mask tensor.
        gt (torch.Tensor): Ground truth mask tensor.
        point_num (int): Number of random points to select. Default is 9.
    Returns:
        batch_points (np.array): Array of selected points coordinates (x, y) for each batch.
        batch_labels (np.array): Array of corresponding labels (0 for background, 1 for foreground) for each batch.
    """
    pred, gt = pr.data.cpu().numpy(), gt.data.cpu().numpy()
    error = np.zeros_like(pred)
    error[pred != gt] = 1

    # error = np.logical_xor(pred, gt)
    batch_points = []
    batch_labels = []
    for j in range(error.shape[0]):
        one_pred = pred[j].squeeze(0)
        one_gt = gt[j].squeeze(0)
        one_erroer = error[j].squeeze(0)

        indices = np.argwhere(one_erroer == 1)
        if indices.shape[0] > 0:
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        else:
            indices = np.random.randint(0, 256, size=(point_num, 2))
            selected_indices = indices[np.random.choice(indices.shape[0], point_num, replace=True)]
        selected_indices = selected_indices.reshape(-1, 2)

        points, labels = [], []
        for i in selected_indices:
            x, y = i[0], i[1]
            if one_pred[x,y] == 0 and one_gt[x,y] == 1:
                label = 1
            elif one_pred[x,y] == 1 and one_gt[x,y] == 0:
                label = 0
            points.append((y, x))   #Negate the coordinates
            labels.append(label)

        batch_points.append(points)
        batch_labels.append(labels)
    return np.array(batch_points), np.array(batch_labels)


def init_point_sampling(mask, get_point=1):
    """
    Initialization samples points from the mask and assigns labels to them.
    Args:
        mask (torch.Tensor): Input mask tensor.
        num_points (int): Number of points to sample. Default is 1.
    Returns:
        coords (torch.Tensor): Tensor containing the sampled points' coordinates (x, y).
        labels (torch.Tensor): Tensor containing the corresponding labels (0 for background, 1 for foreground).
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()
        
     # Get coordinates of black/white pixels
    fg_coords = np.argwhere(mask == 1)[:,::-1]
    bg_coords = np.argwhere(mask == 0)[:,::-1]

    fg_size = len(fg_coords)
    bg_size = len(bg_coords)

    if get_point == 1:
        if fg_size > 0:
            index = np.random.randint(fg_size)
            fg_coord = fg_coords[index]
            label = 1
        else:
            index = np.random.randint(bg_size)
            fg_coord = bg_coords[index]
            label = 0
        return torch.as_tensor([fg_coord.tolist()], dtype=torch.float), torch.as_tensor([label], dtype=torch.int)
    else:
        num_fg = get_point // 2
        num_bg = get_point - num_fg
        fg_indices = np.random.choice(fg_size, size=num_fg, replace=True)
        bg_indices = np.random.choice(bg_size, size=num_bg, replace=True)
        fg_coords = fg_coords[fg_indices]
        bg_coords = bg_coords[bg_indices]
        coords = np.concatenate([fg_coords, bg_coords], axis=0)
        labels = np.concatenate([np.ones(num_fg), np.zeros(num_bg)]).astype(int)
        indices = np.random.permutation(get_point)
        coords, labels = torch.as_tensor(coords[indices], dtype=torch.float), torch.as_tensor(labels[indices], dtype=torch.int)
        return coords, labels
    

def train_transforms(img_size, ori_h, ori_w):
    transforms = []
    if ori_h < img_size and ori_w < img_size:
        transforms.append(A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0)))
    else:
        transforms.append(A.Resize(int(img_size), int(img_size), interpolation=cv2.INTER_NEAREST))
    # transforms.append(ToTensorV2(p=1.0))
    return A.Compose(transforms, p=1.)

class TransformSam():
    def __init__(self, img_size=1024, ori_h=720, ori_w=960) -> None:

        self.img_size = img_size
        self.ori_h = ori_h
        self.ori_w = ori_w
        
    def __call__(self, image, ):
        """
        the transform is the same as SAM, please refer to:
        segment_anything/modeling/sam.py -- preprocess()
        segment_anything/utils/transforms.py -- ResizeLongestSide.get_preprocess_shape()
        """
        scale = self.img_size * 1.0 / max(self.ori_h, self.ori_w)
        newh, neww = self.ori_h * scale, self.ori_w * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        target_size = (newh, neww)
        # resize to (768,1024)
        image = np.array(resize(to_pil_image(image), target_size))
        # padding to (1024,1024)
        padh = self.img_size - newh
        padw = self.img_size - neww
        if len(image.shape) == 3:
            image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant')    
        else:
            image = np.expand_dims(image, axis=2)
            image = np.pad(image, ((0, padh), (0, padw), (0, 0)), mode='constant')
            image = np.squeeze(image, axis=2)
        return image

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def generate_point(masks, labels, low_res_masks, batched_input, point_num):
    masks_clone = masks.clone()
    masks_sigmoid = torch.sigmoid(masks_clone)
    masks_binary = (masks_sigmoid > 0.5).float()

    low_res_masks_clone = low_res_masks.clone()
    low_res_masks_logist = torch.sigmoid(low_res_masks_clone)

    points, point_labels = select_random_points(masks_binary, labels, point_num = point_num)
    batched_input["mask_inputs"] = low_res_masks_logist
    batched_input["point_coords"] = torch.as_tensor(points)
    batched_input["point_labels"] = torch.as_tensor(point_labels)
    batched_input["boxes"] = None
    return batched_input


def setting_prompt_none(batched_input):
    batched_input["point_coords"] = None
    batched_input["point_labels"] = None
    batched_input["boxes"] = None
    return batched_input


def draw_boxes(img, boxes):
    img_copy = np.copy(img)
    for box in boxes:
        cv2.rectangle(img_copy, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    return img_copy


def save_masks(preds, save_path, mask_name, image_size, original_size, pad=None,  boxes=None, points=None, visual_prompt=False):
    ori_h, ori_w = original_size

    preds = torch.sigmoid(preds)
    preds[preds > 0.5] = int(1)
    preds[preds <= 0.5] = int(0)

    mask = preds.squeeze().cpu().numpy()
    mask = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    if visual_prompt: #visualize the prompt
        if boxes is not None:
            boxes = boxes.squeeze().cpu().numpy()

            x0, y0, x1, y1 = boxes
            if pad is not None:
                x0_ori = int((x0 - pad[1]) + 0.5)
                y0_ori = int((y0 - pad[0]) + 0.5)
                x1_ori = int((x1 - pad[1]) + 0.5)
                y1_ori = int((y1 - pad[0]) + 0.5)
            else:
                x0_ori = int(x0 * ori_w / image_size) 
                y0_ori = int(y0 * ori_h / image_size) 
                x1_ori = int(x1 * ori_w / image_size) 
                y1_ori = int(y1 * ori_h / image_size)

            boxes = [(x0_ori, y0_ori, x1_ori, y1_ori)]
            mask = draw_boxes(mask, boxes)

        if points is not None:
            point_coords, point_labels = points[0].squeeze(0).cpu().numpy(),  points[1].squeeze(0).cpu().numpy()
            point_coords = point_coords.tolist()
            if pad is not None:
                ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))]if l==0 else [x - pad[1], y - pad[0]]  for (x, y), l in zip(point_coords, point_labels)]
            else:
                ori_points = [[int((x * ori_w / image_size)) , int((y * ori_h / image_size))] for x, y in point_coords]

            for point, label in zip(ori_points, point_labels):
                x, y = map(int, point)
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                mask[y, x] = color
                cv2.drawMarker(mask, (x, y), color, markerType=cv2.MARKER_CROSS , markerSize=7, thickness=2)  
    os.makedirs(save_path, exist_ok=True)
    mask_path = os.path.join(save_path, f"{mask_name}")
    cv2.imwrite(mask_path, np.uint8(mask))


#Loss funcation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        num_pos = torch.sum(mask)
        num_neg = mask.numel() - num_pos
        w_pos = (1 - p) ** self.gamma
        w_neg = p ** self.gamma

        loss_pos = -self.alpha * mask * w_pos * torch.log(p + 1e-12)
        loss_neg = -(1 - self.alpha) * (1 - mask) * w_neg * torch.log(1 - p + 1e-12)

        loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)

        return loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        p = torch.sigmoid(pred)
        intersection = torch.sum(p * mask)
        union = torch.sum(p) + torch.sum(mask)
        dice_loss = (2.0 * intersection + self.smooth) / (union + self.smooth)

        return 1 - dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        # replace focal loss with BCE loss
        # self.focal_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        # p = torch.sigmoid(pred)
        # save_image(p[0],'t_mask.png')
        # save_image(mask[0].to(torch.float32),'t_label.png')

        focal_loss = self.focal_loss(pred, mask)
        # replace focal loss with BCE loss
        # focal_loss = self.focal_loss(p, mask.to(torch.float32))
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        # loss = loss1
        return loss
    

def stack_dict_batched_coco_style(data_item, args):
    x, y = data_item
    data_item = {}
    # data_item["point_coords"] = None
    boxes = []
    labels = []
    orig_size = []
    image_paths = []
    point_coords = []
    point_labels = []
    mask_nums = []
    if args.dataset == 'ytvis':
        for target in y:
            valid_indices = (target['valid'] == 1).nonzero(as_tuple=True)
            boxes.append(target['boxes'][valid_indices])
            labels.append(target['masks'][valid_indices])
            orig_size.append(target['orig_size'])
            image_paths.append(target['image_path'])
            point_coords.append(target['point_coords'][valid_indices])
            point_labels.append(target['point_labels'][valid_indices])
    else:
        for target in y:
            boxes.append(target['boxes'])
            labels.append(target['masks'])
            orig_size.append(target['orig_size'])
            image_paths.append(target['image_path'])
            point_coords.append(target['point_coords'])
            point_labels.append(target['point_labels'])
            mask_nums.append(len(target['boxes']))

    data_item['boxes'] = torch.cat(boxes, dim=0).unsqueeze(1)
    data_item['label'] = torch.cat(labels, dim=0).unsqueeze(1).to(torch.int)
    # shape: (bsz,2)
    data_item['orig_size'] = torch.stack(orig_size, dim=0) 
    data_item['image'] = x
    data_item['image_path'] = image_paths
    data_item['point_coords'] = torch.cat(point_coords, dim=0).unsqueeze(1)
    data_item['point_labels'] = torch.cat(point_labels, dim=0).unsqueeze(1)
    data_item['mask_nums'] = torch.tensor(mask_nums, dtype=torch.int)

    return data_item


def stack_dict_batched(data_item):
    out_dict = {}
    for k,v in data_item.items():
        if isinstance(v, list):
            out_dict[k] = v
        else:
            out_dict[k] = v.reshape(-1, *v.shape[2:])
    return out_dict

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key=='image' or key=='label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                 device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input

def unwrap(wrapped_module):
    return wrapped_module

def check_unused_parameters(model):
    
    unused_params = [name for name, param in unwrap(model).named_parameters() 
                        if param.grad is None and param.requires_grad==True]
    if unused_params:
        raise RuntimeError(f"Unused parameters: {unused_params}")
    else:
        print("All the parameters are used.")

def collate_fn(batch):
    ret = []
    for items in zip(*batch):
        if isinstance(items[0], torch.Tensor):
            items = torch.stack(items, dim=0)
            ret.append(items)
        elif isinstance(items[0], dict):
            ret.append(items)
        else:
            raise TypeError
    return tuple(ret)

def transform_prompt(coord,label):

    coord = coord.unsqueeze(1)
    label = label.unsqueeze(1)

    batch_size, max_num_queries, num_pts, _ = coord.shape
    rescaled_batched_points = coord

    decoder_max_num_input_points = 6
    if num_pts > decoder_max_num_input_points:
        rescaled_batched_points = rescaled_batched_points[
            :, :, : decoder_max_num_input_points, :
        ]
        label = label[
            :, :, : decoder_max_num_input_points
        ]
    elif num_pts < decoder_max_num_input_points:
        rescaled_batched_points = F.pad(
            rescaled_batched_points,
            (0, 0, 0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
        label = F.pad(
            label,
            (0, decoder_max_num_input_points - num_pts),
            value=-1.0,
        )
    
    rescaled_batched_points = rescaled_batched_points.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points, 2
    )
    label = label.reshape(
        batch_size * max_num_queries, decoder_max_num_input_points
    )

    return rescaled_batched_points, label

def custom_take_along_dim(input, indices, dim):
    
    all_indices = [torch.arange(s) for s in input.shape]
    mesh = torch.meshgrid(*all_indices)

    
    index_list = list(mesh)
    index_list[dim] = indices

    
    result = input[index_list]

    return result

def remove_symlinks_in_directory(directory_path):
    
    if not os.path.exists(directory_path):
        print(f"{directory_path} does not exists")
        return
    
    for root, dirs, files in os.walk(directory_path):
        for name in dirs + files:
            path = os.path.join(root, name)
            if os.path.islink(path):
                try:
                    os.unlink(path)
                    print(f"Remove symlink: {path}")
                except Exception as e:
                    print(f"Failed to remove symlink: {path}, error: {e}")

def create_symlinks_with_renaming(source_dir, target_dir, old_str, new_str):
    # Get absolute paths
    source_dir = os.path.abspath(source_dir)
    target_dir = os.path.abspath(target_dir)
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist")
        return
    
    # Check if target directory exists, create if not
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Iterate over all files and directories in the source directory
    for item in os.listdir(source_dir):
        source_path = os.path.join(source_dir, item)
        # Replace old_str with new_str in the file name
        new_name = item.replace(old_str, new_str)
        target_path = os.path.join(target_dir, new_name)
        
        # Skip if target path already exists
        if os.path.exists(target_path):
            print(f"Target path {target_path} already exists, skipping")
            continue
        
        # Create symlink
        try:
            os.symlink(source_path, target_path)
            print(f"Created symlink: {source_path} -> {target_path}")
        except Exception as e:
            print(f"Failed to create symlink: {source_path} -> {target_path}, error: {e}")

