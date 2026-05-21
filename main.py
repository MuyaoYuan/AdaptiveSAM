import torch
import numpy as np
import json
import math
import cv2
import random
import os
import gc
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from segment_anything.modeling import TinyViTWithFuseSuperNetSampler, ImageEncoderViTWithFuseSuperNetSampler
from utils import FocalDiceloss_IoULoss, stack_dict_batched
from dataset.camvid_sa import CamVidSATesting, CamVidSA
from metrics import eval_seg
import argparse
from lookup.lookup import ResultLookUp
import time
from searcher.evolution_search import EvolutionSearcher
from util.stream import FIFOCache


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

def prompt_and_decoder(data_item, model, image_embeddings, image_size, multimask, hq=False, interm_embeddings=None):
    if  data_item["point_coords"] is not None:
        points = (data_item["point_coords"], data_item["point_labels"])
    else:
        points = None


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

def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
        )
    
    scale = image_size * 1.0 / max(ori_h, ori_w)
    newh, neww = ori_h * scale, ori_w * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    target_size = (newh, neww)
    masks = masks[..., : target_size[0], : target_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)

    return masks, scale

def parse_args():
    parser = argparse.ArgumentParser(description='Lookup System')
    parser.add_argument('--data-path', type=str, help='Path to dataset folder')
    parser.add_argument('--save-path', type=str, default=None, help='Path for storing model snapshots')
    parser.add_argument('--backend', type=str, default='efficient_vit_h', help='Feature extractor')
    parser.add_argument('--snapshot', type=str, default='./ckpt/mobile_sam.pt', help='Path to pretrained weights')
    parser.add_argument('--image_size', type=int, default=1024, help='Image size')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--ref_gap', type=int, default=2, help='The length of reference GOP.')
    parser.add_argument('--bitrate', type=int, default=3, help='bitrate of dataset.')
    parser.add_argument('--dataset', type=str, default='camvid', help='dataset')
    parser.add_argument('--mask_num', type=int, default=5, help='number of sampling masks')
    parser.add_argument('--point_num', type=int, default=1, help='number of sampling point')
    parser.add_argument('--multimask', type=bool, default=True, help='ouput multimask')
    parser.add_argument('--no_multimask', dest='multimask', action='store_false')
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--train_workers", type=int, default=0)
    parser.add_argument("--val_workers", type=int, default=4)
    parser.add_argument("--metrics", nargs='*', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--split', type=str, default="val")
    parser.add_argument('--early_exit', type=int, default=10**18)
    parser.add_argument('--adapter_type', type=int, default=-1, help="-1=No Adapter, 0=Series Adapter, 1=Parallel Adapter, 2=Mixed Adapter, 3=Supernet")
    parser.add_argument("--adapter_config", nargs='*', type=int, default=[-1,-1,-1,-1,-1,-1], help='The adapter config, \
                    len(config)==6 means only the third layer is adapted, len(config)==10 means all the layers except the first layer are adapted')
    parser.add_argument("--mlp_config", nargs='*', type=float, default=[0.25]*20, help='MLP ratio config')
    parser.add_argument(
        '--search_type',
        type=str,
        default='random',
        help='random, evolution, elitist_evolution, or hill_climbing',
    )
    parser.add_argument("--indicator_name", nargs='*', default=['NASWOT'], help="indicator_name")
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--population-num', type=int, default=800)
    parser.add_argument('--param-limits', type=float, default=1e10)
    parser.add_argument('--min-param-limits', type=float, default=0)
    parser.add_argument('--param-type', type=str, default='params', help='the type of param-limits')
    parser.add_argument('--contrastive_search', type=bool, default=False, help='when the indicator is SC, True')

    ## evolution search
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=25)
    parser.add_argument('--mutation-num', type=int, default=25)
    
    # lookup parameters
    parser.add_argument('--num_M', type=int, default=16, help='M for PQ')
    parser.add_argument('--nbits', type=int, default=4, help='nbits')
    parser.add_argument('--num_entry', type=int, default=100, help='num_entry for init_lookup')
    parser.add_argument('--bs_lookup_init', type=int, default=4, help='bs_lookup_init')
    parser.add_argument("--adapter_option", nargs='*', type=int, default=[-1, 2], help='adapter_option for supernet')
    parser.add_argument("--mlp_option", nargs='*', type=float, default=[0.25, 0.5, 0.75], help='adapter_option for supernet')
    parser.add_argument('--sc_lamda', type=float, default=2, help='parameter for zero-cost NAS: sample contrastive score')
    parser.add_argument('--num_pos_samples', type=int, default=10, help='parameter for zero-cost NAS: sample contrastive score')
    parser.add_argument('--num_neg_samples', type=int, default=10, help='parameter for zero-cost NAS: sample contrastive score')

    parser.add_argument('--lookup_ckpt', type=str, default=None, help='The dir of saved lookup')
    parser.add_argument('--load_lookup_updated', type=bool, default=False, help='Load lookup_updated.csv instead of lookup.csv')
    parser.add_argument('--only_pq', type=bool, default=False, help='Only load PQ without lookup table')
    parser.add_argument('--subnet_weight', type=str, default='inherited', help='inherited , retrain or fine_tune')
    parser.add_argument("--retrain_gpus", nargs='*', type=int, default=[1,2,3,4,5,6], help='adapter_option for supernet')
    parser.add_argument('--retrain_epochs', type=int, default=4, help='Number of training epochs to run')
    parser.add_argument('--retrain_root', type=str, default='./exp', help='Root directory for retrained mutable checkpoints')
    parser.add_argument('--look_up_init_from', type=str, default='sequence', help='neg sample for search come from sequence or neg_loader during lookup init')

    parser.add_argument('--update', type=bool, default=False, help='lookup update or not')
    parser.add_argument('--early_feat', type=bool, default=False, help='Using early feature as the indicator of lookup index')
    parser.add_argument('--tolerance_ratio', type=float, default=0.6, help='if error > num_M*tolerance_ratio, re-search')
    parser.add_argument('--stream', type=bool, default=False, help='if error > num_M*tolerance_ratio, re-search')
    parser.add_argument('--cache_capacity', type=int, default=20, help='if error > num_M*tolerance_ratio, re-search')
    parser.add_argument('--neg_loader', type=str, default=None, help='A dataloader to produce aux neg samples')
    parser.add_argument('--infer_gap', type=int, default=1, help='The gap of frames inferred')
    parser.add_argument('--init_population', type=bool, default=False, help='init population with lookup ')
    parser.add_argument("--subset_for_camvid", type=str, default=None, help='0001TP, 0006R0, 0016E5 or Seq05VD')

    return parser.parse_args()

def validation():
    
    ######### 1-init

    args = parse_args()

    device = "cuda"
    os.makedirs(args.save_path, exist_ok=True)

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # init backbone
    if args.backend == 'efficient_vit_t':
        args.model_name = 'MobileSAM'
    elif args.backend == 'efficient_vit_b':
        args.model_name = 'SAM_b'
    elif args.backend == 'efficient_vit_t_hq':
        args.model_name = 'HQSAM'
    else:
        raise NotImplementedError(f'No {args.backend} model name')

    if args.subnet_weight == 'inherited':
        efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot, adapter_type=args.adapter_type)

    efficient_sam_for_search = sam_model_registry[args.backend](checkpoint=args.snapshot, adapter_type=args.adapter_type)

    if args.adapter_type == 3:
        args.supernet_type = 1
    elif args.adapter_type == 4:
        args.supernet_type = 2
    else:
        raise ValueError(f'No such supernet for adapter_type {args.adapter_type}')

    if args.subnet_weight == 'inherited':
        if args.backend == 'efficient_vit_t' or args.backend == 'efficient_vit_t_hq':
            supernet = TinyViTWithFuseSuperNetSampler(efficient_sam, num_decisions=10, supernet_type=args.supernet_type)
        elif args.backend == 'efficient_vit_b':
            supernet = ImageEncoderViTWithFuseSuperNetSampler(efficient_sam, num_decisions=10, supernet_type=args.supernet_type)
        else:
            raise NotImplementedError(f'No {args.backend} supernet')
        supernet.to(device=device)
        supernet.eval()
        efficient_sam = supernet.model

    if args.backend == 'efficient_vit_t' or args.backend == 'efficient_vit_t_hq':
        supernet_for_search = TinyViTWithFuseSuperNetSampler(efficient_sam_for_search, num_decisions=10, supernet_type=args.supernet_type)
    elif args.backend == 'efficient_vit_b':
        supernet_for_search = ImageEncoderViTWithFuseSuperNetSampler(efficient_sam_for_search, num_decisions=10, supernet_type=args.supernet_type)
    else:
        raise NotImplementedError(f'No {args.backend} supernet')
    supernet_for_search.to(device=device)
    supernet_for_search.eval()
    efficient_sam_for_search = supernet_for_search.model

    # init criterion
    criterion = FocalDiceloss_IoULoss()

    # init dataset
    if args.dataset == 'camvid':
        args.dataset_save_name = 'CamVid'
        train_ds_sa = CamVidSA(args.data_path, image_size=args.image_size, mode='train',
                                ref_gap=args.ref_gap,
                                mask_num=args.mask_num,
                                point_num=args.point_num,
                                )
        
        # disable train_aug
        train_ds_sa.train_aug = False
        
        test_ds_sa = CamVidSATesting(args.data_path, image_size=args.image_size, mode=args.split,
                                    ref_gap=args.ref_gap,
                                    point_num=args.point_num,
                                    stream=args.stream,
                                    sub_set=args.subset_for_camvid)

    else:
        raise NotImplementedError(f'Only camvid is supported in this pre-release version (got {args.dataset}).')

    if args.neg_loader == 'camvid':
        neg_ds_sa = CamVidSA('data/CamVid', image_size=args.image_size, mode='train',
                                    ref_gap=args.ref_gap,
                                    mask_num=args.mask_num,
                                    point_num=args.point_num)
        
        neg_loader_sa = DataLoader(neg_ds_sa, batch_size=args.num_neg_samples,
                                    num_workers=0,
                                    shuffle=True,
                                    pin_memory=False)
        
        neg_loader_iter = neg_loader_sa.__iter__()

    else:
        print('no neg_loader dataset')
        neg_loader_iter = None

    this_collate_fn = None

    sampler_train_sa = torch.utils.data.RandomSampler(train_ds_sa)
    # bs for init_set is 32
    batch_sampler_train_sa = torch.utils.data.BatchSampler(
        sampler_train_sa, 32, drop_last=True)
    
    train_loader_sa = DataLoader(train_ds_sa, batch_sampler=batch_sampler_train_sa,
                num_workers = args.train_workers,
                collate_fn=this_collate_fn,
                pin_memory = False)

    test_loader = DataLoader(dataset=test_ds_sa, batch_size=args.batch_size, shuffle=False, num_workers=args.val_workers, collate_fn=this_collate_fn)
    print('Test data:', len(test_loader))
    print('Train data:', len(train_loader_sa))

    samples = None

    # init zero-cost NAS
    if args.search_type == 'evolution':
        searcher = EvolutionSearcher(args, supernet_for_search, efficient_sam_for_search, train_loader_sa, samples, None)
        if 'SC' in args.indicator_name:
            args.contrastive_search = True
    else:
        raise ValueError(
            f"Unknown search type: {args.search_type}. Only 'evolution' is supported in this pre-release version."
        )

    # init supernet
    if args.supernet_type == 1:
        supernet_for_search.sample_path(tuple((np.ones(10)*-1).astype(np.int8)))
    elif args.supernet_type == 2:
        supernet_for_search.sample_path(tuple((np.ones(10)*-1).astype(np.int8)), tuple(np.ones(20)*0.25))
    else:
        raise ValueError(f'No such supernet {args.supernet_type}')
    
    # init lookup
    lookup_sys = ResultLookUp(
        efficient_sam_for_search,
        train_loader_sa,
        M=args.num_M,
        nbits=args.nbits,
        num_entry=args.num_entry,
        bs=args.bs_lookup_init,
        args=args,
        update=args.update,
        searcher=searcher,
        ckpt=args.lookup_ckpt,
    )

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    test_loss = []
    test_res = [0] * len(args.metrics)
    threshold = (0.1, 0.3, 0.5, 0.7, 0.9)
    config_last = None
    
    # init data cache for search
    if args.stream:
        if args.neg_loader is None:
            assert args.cache_capacity >= args.num_pos_samples + args.num_neg_samples
        else:
            assert args.cache_capacity >= args.num_pos_samples
        samples_cache = FIFOCache(args.cache_capacity, args.mask_num)
    
    ################## 2-search & retrain
    for idx, data_item in enumerate(test_pbar):
        with open(os.path.join(args.save_path,"config.txt"), "a") as f:
            f.write('==================================================================================================================================\n')
        if idx > args.early_exit:
            print('early exit for showing')
            return
        
        if args.dataset == 'camvid':
            data_item = to_device(data_item, device)
            x, ori_labels, original_size = data_item["image"], data_item["ori_label"], data_item["original_size"]
            # noticed that the original_size in camvid is the same:
            original_size = [original_size[0][0], original_size[1][0]]
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not supported in this pre-release version.')
        
        if args.stream:
            image_path = data_item['image_path']
            if samples_cache.get_length() ==0 or image_path != samples_cache.get_last_n(1)['image_path']:
                samples_cache.put(data_item)
            
        s_time = time.time()
        # config supernet with zero-cost NAS or lookup
        # acquire attn feature with no-adapter
        if args.supernet_type == 1:
            supernet_for_search.sample_path(tuple((np.ones(10)*-1).astype(np.int8)))
        elif args.supernet_type == 2:
            supernet_for_search.sample_path(tuple((np.ones(10)*-1).astype(np.int8)), tuple(np.ones(20)*0.25))
        else:
            raise ValueError(f'No such supernet {args.supernet_type}')

        with torch.no_grad():
            if args.early_feat:
                early_features_orig = efficient_sam_for_search.image_encoder.get_early_features(x)
                early_features = early_features_orig
                if args.backend == 'efficient_vit_t' or args.backend == 'efficient_vit_t_hq':
                    B, HW, C = early_features.shape
                    Hp = Wp = int(np.sqrt(HW))
                    early_features = early_features.transpose(1, 2).view(B, C, Hp, Wp)
                elif args.backend == 'efficient_vit_b':
                    B, Hp, Wp, C = early_features.shape
                    early_features = early_features.permute(0,3,1,2)
                else:
                    NotImplementedError(f'No early_features for {args.backend}')

                window_size = int(Hp // np.sqrt(args.num_M))
                attn_features = early_features.permute(0,2,3,1).view(B, Hp // window_size, window_size, Wp // window_size, window_size, C).permute(0,2,4,1,3,5).flatten(1,-1).cpu().numpy().astype('float32')
            else:
                attn_features = efficient_sam_for_search.image_encoder(x)
                B, C, Hp, Wp = attn_features.shape
                window_size = int(Hp // np.sqrt(args.num_M))
                attn_features = attn_features.permute(0,2,3,1).view(B, Hp // window_size, window_size, Wp // window_size, window_size, C).permute(0,2,4,1,3,5).flatten(1,-1).cpu().numpy().astype('float32')

        config, row, error, code = lookup_sys.match(attn_features)
        if args.stream:
            trigger_condition = error is not None and error > args.tolerance_ratio * args.num_M
            if trigger_condition and samples_cache.is_full():
                print('error: ', error)
                target_samples = samples_cache.get_last_n(args.num_pos_samples)
                if args.contrastive_search:
                    if neg_loader_iter is None:
                        neg_samples = samples_cache.get_first_n(args.num_neg_samples)
                    else:
                        try:
                            neg_samples = next(neg_loader_iter)
                            neg_samples = stack_dict_batched(neg_samples)
                            neg_samples = to_device(neg_samples,'cuda')
                        except StopIteration:
                            neg_loader_iter = neg_loader_sa.__iter__()
                            neg_samples = next(neg_loader_iter)
                            neg_samples = stack_dict_batched(neg_samples)
                            neg_samples = to_device(neg_samples,'cuda')
                else:
                    neg_samples = None

                try:
                    efficient_sam.cpu()
                    if args.early_feat:
                        early_features_orig.cpu()
                    del masks, low_res_masks, iou_predictions, image_embeddings, attn_features
                    if args.model_name == 'HQSAM':
                        del interm_embeddings
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    print('no efficient_sam')
                config, _ = lookup_sys.re_search(target_samples, neg_samples, code)
                try:
                    efficient_sam.cuda()
                    if args.early_feat:
                        early_features_orig.cuda()
                except:
                    print('no efficient_sam')
                if args.init_population:
                    lookup_sys.searcher.init_population.append(config)
                with open(os.path.join(args.save_path,"config.txt"), "a") as f:
                    f.write(f're-search, config: {config} \n')
                    f.write(lookup_sys.lookup.to_string())
                    f.write('\n')
            with open(os.path.join(args.save_path,"config.txt"), "a") as f:
                f.write(f'config: {config}, row: {row} \n')
                f.write(f'image path: {data_item["image_path"]} \n')

        # go to the best path
        if config != config_last:
            if args.subnet_weight == 'inherited':
                if args.supernet_type == 1:
                    supernet.sample_path(config)
                elif args.supernet_type == 2:
                    supernet.sample_path(config[:10], config[10:])
                else:
                    raise ValueError(f'No such supernet type {args.supernet_type}')
            elif args.subnet_weight == 'retrain':
                adapter_config_str = "_".join(map(str, config))
                if args.supernet_type == 1:
                    adapter_config = config
                    mlp_config = [0.25] * 20
                elif args.supernet_type == 2:
                    adapter_config = config[:10]
                    mlp_config = config[10:]
                else:
                    raise ValueError(f'No such supernet type {args.supernet_type}')
                
                if args.backend == 'efficient_vit_t' or args.backend == 'efficient_vit_t_hq':
                    pass
                elif args.backend == 'efficient_vit_b':
                    adapter_config = (-1, -1) + adapter_config
                    mlp_config = (0.25, 0.25, 0.25, 0.25) + mlp_config
                else:
                    raise NotImplementedError(f'No {args.backend} model name')
                try:
                    efficient_sam = sam_model_registry['mutable_'+args.backend](checkpoint=f"{args.retrain_root}/{args.dataset_save_name}/retrain/{args.model_name}_{args.dataset_save_name}_{adapter_config_str}_eps{args.retrain_epochs}/mutable_{args.backend}_{args.retrain_epochs}_.pth", 
                                                                                adapter_config=adapter_config, mlp_config=mlp_config)
                except:
                    print('Retrain do not finished')
                    if args.backend == 'efficient_vit_t' or args.backend == 'efficient_vit_t_hq':
                        adapter_config_zero = (-1,) * 10
                        mlp_config_zero = (0.25,) * 20
                    elif args.backend == 'efficient_vit_b':
                        adapter_config_zero = (-1,) * 12
                        mlp_config_zero = (0.25,) * 24
                    efficient_sam = sam_model_registry['mutable_'+args.backend](checkpoint=args.snapshot,
                                                                                adapter_config=adapter_config_zero, mlp_config=mlp_config_zero)
                    print('Use model without adapters')

            efficient_sam.cuda()
            efficient_sam.eval()
            e_time = time.time()
            print(f"Time of match: {e_time-s_time} sec")
        else:
            print('Config remain unchanged')
        config_last = config

        with torch.no_grad():
            if args.model_name=='HQSAM':
                if args.early_feat:
                    attn_features, interm_embeddings = efficient_sam.image_encoder.forward_with_early_features(early_features_orig)
                else:
                    attn_features, interm_embeddings = efficient_sam.image_encoder(x)
            else:
                if args.early_feat:
                    attn_features = efficient_sam.image_encoder.forward_with_early_features(early_features_orig)
                else:
                    attn_features = efficient_sam.image_encoder(x)

            image_embeddings = attn_features
            data_item["point_coords"], data_item["point_labels"] = None, None
            if args.model_name == 'HQSAM':
                masks, low_res_masks, iou_predictions = prompt_and_decoder(data_item, efficient_sam, image_embeddings, args.image_size, args.multimask, True, interm_embeddings)
            else:
                masks, low_res_masks, iou_predictions = prompt_and_decoder(data_item, efficient_sam, image_embeddings, args.image_size, args.multimask)
        if args.dataset == 'camvid':
            masks, scale_post = postprocess_masks(low_res_masks, args.image_size, original_size)

            loss = criterion(masks, ori_labels, iou_predictions)
            test_loss.append(loss.item())

            temp = eval_seg(masks, ori_labels, threshold)
            test_res = [sum(a) for a in zip(test_res, temp)]
        else:
            raise NotImplementedError(f'Dataset {args.dataset} not supported in this pre-release version.')
            
        
    if args.dataset == 'camvid':
        test_iter_metrics = [a/l for a in test_res]
        test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
        average_loss = np.mean(test_loss)
        status = f"Test loss: {average_loss:.4f}, metrics: {test_metrics} \n"
        with open(os.path.join(args.save_path,"eval.txt"), "a") as f:
                        f.write(status)

    else:
        test_iter_metrics = [a/l for a in test_res]
        test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}
        status = f"metrics: {test_metrics} \n"
        with open(os.path.join(args.save_path,"eval.txt"), "a") as f:
                        f.write(status)
    
    lookup_sys.lookup.to_csv(os.path.join(args.save_path,'lookup_updated.csv'))


if __name__ == '__main__':
    validation()
