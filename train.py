import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from segment_anything.utils.lr_scheduler import LRWarmupScheduler
import torch.nn.functional as F
import numpy as np
import random
import time
import argparse

from segment_anything import sam_model_registry
from segment_anything.modeling import TinyViTWithFuseSuperNetSampler, sample_adapter_configuration, sample_mlp_configuration
from dataset.camvid_sa import CamVidSA
from utils import FocalDiceloss_IoULoss, generate_point, setting_prompt_none, to_device, stack_dict_batched, check_unused_parameters, get_total_grad_norm, transform_prompt, custom_take_along_dim


def prompt_and_decoder(data_item, model, image_embeddings, image_size, multimask, hq=False, interm_embeddings=None, decoder_iter=False):
    if  data_item["point_coords"] is not None:
        points = (data_item["point_coords"], data_item["point_labels"])
    else:
        points = None

    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=points,
                boxes=data_item.get("boxes", None),
                masks=data_item.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.prompt_encoder(
            points=points,
            boxes=data_item.get("boxes", None),
            masks=data_item.get("mask_inputs", None),
        )

    if hq:
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
            hq_token_only=False,
            interm_embeddings=interm_embeddings
        )
    else:
        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings = image_embeddings,
            image_pe = model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask,
        )
  
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(image_size, image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions

def prompt_and_decoder_points(points, model, image_embeddings, image_size, multimask):
    
    # points [bs, 1, 2]    lable[bs, 1]
    points = (points, torch.ones(points.shape[0], device=points.device).unsqueeze(1))
    
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=points,
        boxes=None,
        masks=None,
    )  

    low_res_masks, iou_predictions = model.mask_decoder(
        image_embeddings = image_embeddings,
        image_pe = model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask,
    )
  
    if multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i+1, idx])
        low_res_masks = torch.stack(low_res, 0)

    masks = F.interpolate(low_res_masks,(image_size, image_size), mode="bilinear", align_corners=False,)
    return masks, low_res_masks, iou_predictions

# function for grid generation
def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

def batch_iterator(batch_size: int, *args):
    assert len(args) > 0 and all(
        len(a) == len(args[0]) for a in args
    ), "Batched iteration must have inputs of all the same size."
    n_batches = len(args[0]) // batch_size + int(len(args[0]) % batch_size != 0)
    for b in range(n_batches):
        yield [arg[b * batch_size : (b + 1) * batch_size] for arg in args]

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data-path', type=str, help='Path to dataset folder')
    parser.add_argument('--split', type=str, default="train")
    parser.add_argument('--models-path', type=str, default=None, help='Path for storing model snapshots')
    parser.add_argument('--backend', type=str, default='efficient_vit_h', help='Feature extractor')
    parser.add_argument('--snapshot', type=str, default='ckpt/sam_vit_h_4b8939.pth', help='Path to pretrained weights')
    parser.add_argument('--image_size', type=int, default=1024, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
    parser.add_argument('--epochs', type=float, default=20, help='Number of training epochs to run')
    parser.add_argument('--start-lr', type=float, default=0.001)
    parser.add_argument('--ref_gap', type=int, default=12, help='The length of reference GOP.')
    parser.add_argument('--bitrate', type=int, default=3, help='bitrate of dataset.')
    parser.add_argument('--dataset', type=str, default='camvid', help='dataset')
    parser.add_argument('--resume', type=bool, default=False, help='Resume from snapshot')
    parser.add_argument('--mask_num', type=int, default=5, help='number of sampling masks')
    parser.add_argument('--iterative_train', type=bool, default=False, help='iterative prompt training')
    parser.add_argument('--point_num', type=int, default=1, help='number of sampling point')
    parser.add_argument('--multimask', type=bool, default=True, help='ouput multimask')
    parser.add_argument('--no_multimask', dest='multimask', action='store_false')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--iter_point", type=int, default=8, help="iter num")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument('--frozen', type=bool, default=True, help='frozen early layers or not')
    parser.add_argument("--train_workers", type=int, default=4)
    parser.add_argument('--lr_warm_up', type=bool, default=False)
    parser.add_argument('--n_points_per_side', type=int, default=4)
    parser.add_argument('--batch_size_points', type=int, default=16)
    parser.add_argument('--adapter_type', type=int, default=-1, help="-1=No Adapter, 0=Series Adapter, 1=Parallel Adapter, 2=Mixed Adapter, 3=Supernet")
    parser.add_argument("--adapter_config", nargs='*', type=int, default=[-1,-1,-1,-1,-1,-1], help='The adapter config, \
                    len(config)==6 means only the third layer is adapted, len(config)==10 means all the layers except the first layer are adapted')
    parser.add_argument('--train_supernet', type=bool, default=False)
    parser.add_argument('--single_path', type=bool, default=False)
    parser.add_argument("--adapter_option", nargs='*', type=int, default=[-1, 2], help='adapter_option for supernet')
    parser.add_argument("--mlp_config", nargs='*', type=float, default=[0.25]*20, help='MLP ratio config')
    parser.add_argument("--subset_for_camvid", type=str, default=None, help='0001TP, 0006R0, 0016E5 or Seq05VD')
    parser.add_argument("--unfrozen_norm", type=bool, default=False)

    parser.add_argument("--directly_tuning_layers", type=int, default=None)
    
    # for retrain
    parser.add_argument("--starting_epoch", type=int, default=0)

    return parser.parse_args()


def train():
    
    args = parse_args()

    os.makedirs(args.models_path, exist_ok=True)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # sam = sam_model_registry["default"](checkpoint="ckpt/sam_vit_h_4b8939.pth")
    # sam.to(device='cuda')
    # sam.eval()
    if args.backend == 'mutable_efficient_vit_t' or args.backend == 'efficient_vit_t' or args.backend == 'vit_t':
        args.model_name = 'MobileSAM'
    elif args.backend == 'mutable_efficient_vit_b' or args.backend == 'efficient_vit_b':
        args.model_name = 'SAM_b'
    elif args.backend == 'efficient_vit_t_hq' or args.backend == 'mutable_efficient_vit_t_hq':
        args.model_name = 'HQSAM'
    else:
        raise NotImplementedError(f'No {args.backend} model name')

    sam = None
    
    if not args.resume:
        if args.backend == 'vit_t':
            efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot)
        else:
            efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot,
                                                        adapter_type=args.adapter_type,
                                                        adapter_config=args.adapter_config,
                                                        mlp_config=args.mlp_config)
        starting_epoch = 0
    else:
        checkpoint = torch.load(args.snapshot, map_location='cpu')
        if args.starting_epoch is not None:
            starting_epoch = int(args.starting_epoch)
        else:
            starting_epoch = checkpoint['epoch']

        efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot,  
                                                    adapter_type=args.adapter_type,
                                                    adapter_config=args.adapter_config,
                                                    mlp_config=args.mlp_config)
        print("Snapshot loaded from {}".format(args.snapshot))

    efficient_sam.to(device='cuda')
    efficient_sam.train()

    for param in efficient_sam.parameters():
        param.requires_grad = False

    if args.frozen:
        if args.directly_tuning_layers is None:
            try:
                for block in efficient_sam.image_encoder.blocks:
                    for adapter in block.Adapters_list:
                        for param in adapter.parameters():
                            param.requires_grad = True
            except:
                try:
                    for lid, layer in enumerate(efficient_sam.image_encoder.layers):
                        if lid == 0:
                            continue
                        else:
                            for block in layer.blocks:
                                for adapter in block.Adapters_list:
                                    for param in adapter.parameters():
                                        param.requires_grad = True
                except:
                    raise AttributeError
            
            if args.unfrozen_norm:
                norm_count = 0
                try:
                    for block in efficient_sam.image_encoder.blocks:
                        if len(block.Adapters_list) == 0:
                            continue
                        for module in block.modules():
                            if isinstance(module, nn.LayerNorm):
                                norm_count = norm_count + 1
                                for param in module.parameters():
                                    param.requires_grad = True
                except:
                    try:
                        for lid, layer in enumerate(efficient_sam.image_encoder.layers):
                            if lid == 0:
                                continue
                            else:
                                for block in layer.blocks:
                                    for module in block.modules():
                                        if isinstance(module, nn.LayerNorm):
                                            norm_count = norm_count + 1
                                            for param in module.parameters():
                                                param.requires_grad = True
                    except:
                        raise AttributeError
                print('unfrozen norm: ', norm_count)
        
        else:
            try:
                for param in efficient_sam.image_encoder.blocks[-args.directly_tuning_layers:].parameters():
                    param.requires_grad = True
            except:
                try: 
                    for param in efficient_sam.image_encoder.layers[-args.directly_tuning_layers:].parameters():
                        param.requires_grad = True
                except:
                    raise AttributeError
            
            for param in efficient_sam.image_encoder.neck.parameters():
                param.requires_grad = True
    else:
        for param in efficient_sam.image_encoder.parameters():
            param.requires_grad = True

    if args.train_supernet:
        if args.adapter_type == 3:
            args.supernet_type = 1
        elif args.adapter_type == 4:
            args.supernet_type = 2
        else:
            raise ValueError(f'No such supernet for adapter_type {args.adapter_type}')
        efficient_sam = TinyViTWithFuseSuperNetSampler(efficient_sam, num_decisions=10, supernet_type=args.supernet_type)
        efficient_sam.train()

    if args.dataset == 'camvid':
        cropsize = [args.image_size, args.image_size]
        randomscale = (0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
        train_ds_sa = CamVidSA(args.data_path, image_size=args.image_size, mode='train',
                                ref_gap=args.ref_gap,
                                mask_num=args.mask_num,
                                point_num=args.point_num,
                                sub_set=args.subset_for_camvid)

    else:
        raise NotImplementedError(f'Only camvid is supported in this pre-release version (got {args.dataset}).')

    sampler_train_sa = torch.utils.data.RandomSampler(train_ds_sa)

    batch_sampler_train_sa = torch.utils.data.BatchSampler(
        sampler_train_sa, args.batch_size, drop_last=True)

    this_collate_fn = None

    train_loader_sa = DataLoader(train_ds_sa, batch_sampler=batch_sampler_train_sa,
                num_workers = args.train_workers,
                collate_fn=this_collate_fn,
                pin_memory = False)

    if not args.train_supernet:
        efficient_sam_without_ddp = efficient_sam
    else:
        supernet = efficient_sam
        efficient_sam_without_ddp = efficient_sam.model

    optimizer = optim.Adam(efficient_sam.parameters(), lr=args.start_lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs*(len(train_ds_sa) // args.batch_size + 1))
    if args.lr_warm_up:
        warmup_scheduler = LRWarmupScheduler(scheduler)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['lr_scheduler'])

    criterion = FocalDiceloss_IoULoss()

    count4contine = 0

    if args.epochs % 1 != 0:
        exit_idx = args.epochs*(len(train_ds_sa) // args.batch_size + 1)
        args.epochs = 1
    else:
        exit_idx = 1e6
        args.epochs = int(args.epochs)

    for epoch in range(starting_epoch, args.epochs):
        epoch_losses = []

        train_iterator = train_loader_sa

        for idx, data_item in enumerate(train_iterator):

            if idx >= exit_idx:
                break
            
            if args.train_supernet:
                if not args.single_path:
                    if args.supernet_type == 1:
                        path = sample_adapter_configuration(num_decisions=10, options=args.adapter_option)
                        path = np.asarray(path)
                        supernet.sample_path(path)
                    elif args.supernet_type == 2:
                        path_adapter = sample_adapter_configuration(num_decisions=10, options=args.adapter_option)
                        path_mlp = sample_mlp_configuration(num_decisions=20)
                        path_adapter = np.asarray(path_adapter)
                        path_mlp = np.asarray(path_mlp)
                        supernet.sample_path(adapter_configuration=path_adapter, mlp_configuration=path_mlp)
                else:
                    if args.supernet_type == 1:
                        supernet.sample_path(args.adapter_config)
                    elif args.supernet_type == 2:
                        supernet.sample_path(args.adapter_config, args.mlp_config)

            optimizer.zero_grad()

            if args.dataset == 'camvid':
                data_item = stack_dict_batched(data_item)
                data_item = to_device(data_item, 'cuda')
                x = data_item["image"]
            else:
                raise NotImplementedError(f'Dataset {args.dataset} not supported in this pre-release version.')

            attn_features = efficient_sam_without_ddp.image_encoder(x)

            if random.random() > 0.5:
                data_item['point_coords'] = None
            else:
                data_item['boxes'] = None

            batch, _, _, _ = attn_features.shape
            image_embeddings_repeat = []
            for i in range(batch):
                image_embed = attn_features[i]
                image_embed = image_embed.repeat(args.mask_num, 1, 1, 1)
                image_embeddings_repeat.append(image_embed)
            image_embeddings = torch.cat(image_embeddings_repeat, dim=0)

            masks, low_res_masks, iou_predictions = prompt_and_decoder(data_item, efficient_sam_without_ddp, image_embeddings, args.image_size, args.multimask)

            labels = data_item["label"]
            if masks.shape[1] == 1:
                loss = criterion(masks, labels, iou_predictions)
            else:
                loss = torch.tensor(0.).cuda()
                for multi_mask_id in range(masks.shape[1]):
                    loss = loss + criterion(masks[:, multi_mask_id:multi_mask_id+1, ...], labels, iou_predictions)

            # iterative prompt training
            if args.iterative_train:
                point_num = random.choice(args.point_list)
                data_item = generate_point(masks, labels, low_res_masks, data_item, point_num)
                data_item = to_device(data_item, 'cuda')

                init_mask_num = np.random.randint(1, iter_point - 1)
                for iter in range(iter_point):
                    if iter == init_mask_num or iter == iter_point - 1:
                        data_item = setting_prompt_none(data_item)
                    
                    masks, low_res_masks, iou_predictions = prompt_and_decoder(data_item, efficient_sam_without_ddp, image_embeddings, args.image_size, args.multimask, decoder_iter=False)
                    
                    loss = loss + criterion(masks, labels, iou_predictions)

                    if iter != iter_point - 1:
                        point_num = random.choice(args.point_list)
                        data_item = generate_point(masks, labels, low_res_masks, data_item, point_num)
                        data_item = to_device(data_item, 'cuda')

            else:
                iter_point = 0 # without point iterative 

            loss = loss / (iter_point+1)

            if torch.isnan(loss).any():
                raise ValueError("Loss contains NaN values. Taking corrective actions...")

            loss.backward()
            optimizer.step()
            scheduler.step()

            # statistic grad
            grad_total_norm_2 = get_total_grad_norm(efficient_sam.parameters(), 2)
            # grad_total_norm_2 = 0
            
            if args.lr_warm_up:
                warmup_scheduler.iter_update()

            if idx == 0:
                check_unused_parameters(efficient_sam)

            epoch_losses.append(loss.item())

            status_dict = {
                'loss': loss.item(),
                'avg': np.mean(epoch_losses),
            }

            status = '[{0}] loss = {1:0.5f} avg = {2:0.5f} grad = {3:0.5f} LR = {4:0.7f} \n'.format(
                epoch + 1, status_dict['loss'], status_dict['avg'], grad_total_norm_2, scheduler.get_last_lr()[0])
            with open(os.path.join(args.models_path,"log_epoch.txt"), "a") as f:
                f.write(status)
            
        # if lr_warm_up:
        #     warmup_scheduler.epoch_update()
        # else:
        #     scheduler.step()

        if (epoch + 1) % 1 == 0:
            torch.save(
                    {'model': efficient_sam_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict(),
                    'epoch': epoch,},
                    os.path.join(args.models_path, '_'.join([args.backend, str(epoch + 1), '.pth']))
            )
    
if __name__ == '__main__':
    train()
