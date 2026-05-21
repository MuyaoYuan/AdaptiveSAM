import os
import gc
import json
import numpy as np
import pandas as pd
import faiss
import random
import torch
from tqdm import tqdm
import ast
from collections import defaultdict
from itertools import chain

from utils import to_device, stack_dict_batched, stack_dict_batched_coco_style
from retrain import run_training
from concurrent.futures import ThreadPoolExecutor

import time
import math

import uuid

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

def split_uint8_to_uint4(array):
    
    bs = array.shape[0]
    M_uint8 = array.shape[1]
    high = (array >> 4) & 0b1111
    low = array & 0b1111
    result = np.empty((bs, M_uint8*2), dtype=array.dtype)
    result[:, ::2] = high
    result[:, 1::2] = low
    return result

def split_uint8_to_uint2(array):
    bs = array.shape[0]
    M_uint8 = array.shape[1]

    bits1 = (array >> 6) & 0b11
    bits2 = (array >> 4) & 0b11
    bits3 = (array >> 2) & 0b11
    bits4 = array & 0b11

    result = np.empty((bs, M_uint8 * 4), dtype=array.dtype)

    result[:, ::4] = bits1
    result[:, 1::4] = bits2
    result[:, 2::4] = bits3
    result[:, 3::4] = bits4
    
    return result

def concatenate_dicts(dict_list):
    keys = dict_list[0].keys()
    result_dict = {}
    for key in keys:
        if key in ['image', 'label', 'boxes', 'point_coords', 'point_labels', 'orig_size', 'mask_nums']:
            result_dict[key] = torch.cat([d[key] for d in dict_list], dim=0)
        elif key in ['image_path']:
            result_list = []
            for d in dict_list:
                result_list = result_list + d[key]
            result_dict[key] = result_list
    return result_dict

def list_multiple_index(orig_list, indices):
    return [orig_list[idx] for idx in indices]

def slice_dict_by_indices(data_dict, indices, num_mask, masks_num_set):
    sliced_dict = {}
    for key, value in data_dict.items():
        if key in ['image', 'orig_size','mask_nums']:
            sliced_dict[key] = value[indices]
        elif key in ['label', 'boxes', 'point_coords', 'point_labels']:
            segments = []
            for idx in indices:
                start = 0 if idx == 0 else sum(masks_num_set[:idx])
                end = sum(masks_num_set[:idx+1])
                segments.append(value[start:end])
            sliced_dict[key] = torch.cat(segments, dim=0)
        elif key in ['image_path']:
            sliced_dict[key] = list_multiple_index(value, indices)
        else:
            raise KeyError(f'No such key {key}')
    return sliced_dict

def get_seq_name(path, args):
    if args.dataset == 'camvid':
        seq_name = os.path.basename(path).split('_')[0]
    elif args.dataset == 'cityscapes' or args.dataset == 'ytvis' or args.dataset == 'cityscapes_panoptic':
        seq_name = path.split('/')[-2]
    else:
        raise NotImplementedError(f'No {args.dataset} in look-up')
        
    return seq_name


class ResultLookUp:
    def __init__(self, model, dataloader, M, nbits, num_entry, bs, args, update=True, searcher=None, ckpt=None):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        self.searcher = searcher
        self.M = M
        self.nbits = nbits
        self.update = update  
        if ckpt is None:
            init_set, samples_set, data_items_set, masks_num_set = self.generate_init_set()
            d = init_set.shape[1] # data dimension
            self.pq = faiss.ProductQuantizer(d, M, nbits)
            s_time = time.time()
            self.pq.train(init_set)
            e_time = time.time()
            print(f"Time of pq.train: {e_time-s_time} sec")
            self.lookup = pd.DataFrame(columns=['Index', 'Config', 'Error', 'Source'])
            s_time = time.time()
            if args.look_up_init_from == 'sequence': 
                self.lookup_init_from_seq(init_set, samples_set, data_items_set, masks_num_set, num_entry, bs)
            else:
                self.lookup_init_from_neg_loader(init_set, samples_set, data_items_set, masks_num_set, num_entry, bs) 
            e_time = time.time()
            print(f"Time of lookup_init: {e_time-s_time} sec")
            self.save(args.save_path)
            if args.init_population:
                self.searcher.init_population = self.lookup['Config'].tolist()
                print(f'Init population from initialized lookup.')
        else:
            loaded_pq_state = np.load(os.path.join(ckpt, "product_quantizer.npy"), allow_pickle=True).item()
            self.pq = faiss.ProductQuantizer(loaded_pq_state["d"], loaded_pq_state["M"], loaded_pq_state["nbits"])
            faiss.copy_array_to_vector(loaded_pq_state["centroids"], self.pq.centroids)
            if not args.only_pq:
                if args.load_lookup_updated:
                    self.lookup = pd.read_csv(os.path.join(ckpt, "lookup_updated.csv"))
                else:
                    self.lookup = pd.read_csv(os.path.join(ckpt, "lookup.csv"))
                if self.M == 1024:
                    self.lookup['Index'] = self.lookup['Index'].apply(lambda x: np.array(json.loads(x)))
                else:
                    self.lookup['Index'] = self.lookup['Index'].apply(lambda x: np.array([np.uint8(i) for i in x.strip('[]').split()]))
                self.lookup['Config'] = self.lookup['Config'].apply(lambda x: ast.literal_eval(x))
                self.lookup = self.lookup.drop(columns=['Unnamed: 0'])
                print(f'Load PQ and lookup from {ckpt}.')
                if args.init_population:
                    self.searcher.init_population = self.lookup['Config'].tolist()
                    print(f'Init population from {ckpt}.')
            else:
                self.lookup = pd.DataFrame(columns=['Index', 'Config', 'Error', 'Source'])
                print(f'Only load PQ from {ckpt}.')
                if args.init_population:
                    lookup = pd.read_csv(os.path.join(ckpt, "lookup.csv"))
                    self.searcher.init_population = lookup['Config'].apply(lambda x: ast.literal_eval(x)).tolist()
                    print(f'Init population from {ckpt}.')
                return # no retrain_check
        
        if self.args.subnet_weight == 'retrain':
            self.searcher.supernet.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            while True:
                self.model_retrain()
                filtered_configs = self.retrain_check()
                if len(filtered_configs) == 0:
                    break
            self.searcher.supernet.cuda()
    
    def reset_searcher(self, ):
        if self.args.search_type == 'random':
                self.searcher.memory = []
                self.searcher.vis_dict = {}
                self.searcher.top = {}
                self.searcher.epoch = 0
                self.searcher.candidates = []
                self.searcher.top_accuracies = []
                self.searcher.cand_params = []
                self.searcher.all_res = []
        elif self.args.search_type in ('evolution', 'contrastive', 'elitist_evolution', 'hill_climbing'):
            self.searcher.memory = []
            self.searcher.vis_dict = {}
            self.searcher.keep_top_k = {self.searcher.select_num: [], 50: []}
            self.searcher.epoch = 0
            self.searcher.candidates = []
            self.searcher.top_accuracies = []
            self.searcher.cand_params = []
        else:
            raise ValueError(
                f"Unknown search type: {self.args.search_type}. Please choose from "
                "'random', 'evolution', 'elitist_evolution', 'hill_climbing', or 'contrastive'."
            )
            
    def generate_init_set(self):
        
        with torch.no_grad():
            # placeholder
            gt_features_p = torch.zeros([self.args.batch_size, 256, 64, 64]).cuda()

            samples_set = []
            init_set = []
            data_items_set = []
            masks_num_set = []

            dataloader_pbar = tqdm(self.dataloader)
            for idx, data_item in enumerate(dataloader_pbar):
                if self.args.dataset == 'camvid':
                    data_item = stack_dict_batched(data_item)
                    data_item = to_device(data_item, 'cuda')
                    x = data_item["image"]
                else:
                    _, y = data_item
                    data_item = stack_dict_batched_coco_style(data_item, self.args)
                    data_item = to_device(data_item, 'cuda')
                    x = data_item['image']

                if self.args.early_feat:
                    early_features = self.model.image_encoder.get_early_features(x)
                    if self.args.backend == 'efficient_vit_t' or self.args.backend == 'efficient_vit_t_hq':
                        B, HW, C = early_features.shape
                        Hp = Wp = int(np.sqrt(HW))
                        early_features = early_features.transpose(1, 2).view(B, C, Hp, Wp)
                    elif self.args.backend == 'efficient_vit_b':
                        B, Hp, Wp, C = early_features.shape
                        early_features = early_features.permute(0,3,1,2)
                    else:
                        NotImplementedError(f'No early_features for {self.args.backend}')
                    window_size = int(Hp // np.sqrt(self.M))
                    attn_features = early_features.permute(0,2,3,1).view(B, Hp // window_size, window_size, Wp // window_size, window_size, C).permute(0,2,4,1,3,5).flatten(1,-1).cpu().numpy().astype('float32')
                else:
                    attn_features = self.model.image_encoder(x)
                    
                    B, C, Hp, Wp = attn_features.shape
                    window_size = int(Hp // np.sqrt(self.M))
                    # print("window_size:", window_size)
                    attn_features = attn_features.permute(0,2,3,1).view(B, Hp // window_size, window_size, Wp // window_size, window_size, C).permute(0,2,4,1,3,5).flatten(1,-1).cpu().numpy().astype('float32')
                
                init_set.append(attn_features)
                samples_set.append(x.cpu())
                data_items_set.append(to_device(data_item, 'cpu'))
                if self.args.dataset == 'camvid':
                    masks_num_set.append(torch.tensor([self.args.mask_num]*x.shape[0], device='cuda', dtype=torch.int))
                else:
                    masks_num_set.append(data_item['mask_nums'])

                # for debug
                # if idx >= 3:
                #     break

            init_set = np.concatenate(init_set, axis=0).astype('float32')
            samples_set = torch.cat(samples_set, dim=0)
            data_items_set = concatenate_dicts(data_items_set)
            masks_num_set = torch.cat(masks_num_set, dim=0)

            return init_set, samples_set, data_items_set, masks_num_set
    
    def lookup_init_from_seq(self, init_set, samples_set, data_items_set, masks_num_set, num_entry, bs):
        n, _ = init_set.shape
        seq_name_to_indices = defaultdict(list)
        for idx, path in enumerate(data_items_set['image_path']):
            seq_name = get_seq_name(path, self.args)
            seq_name_to_indices[seq_name].append(idx)

        for _ in range(num_entry):
            valid = False
            while not valid: # find indices of images in the same video
                if not self.args.contrastive_search:
                    selected_seq_name = np.random.choice(list(seq_name_to_indices.keys()))
                    available_indices = seq_name_to_indices[selected_seq_name]
                    if len(available_indices) >= bs:
                        indices = np.random.choice(available_indices, bs, replace=False)
                        valid = True
                else:
                    selected_seq_name = np.random.choice(list(seq_name_to_indices.keys()))
                    contrastive_seq_name = [key for key in seq_name_to_indices.keys() if key != selected_seq_name]
                    available_indices = seq_name_to_indices[selected_seq_name]
                    available_contrastive_indices = list(chain.from_iterable(seq_name_to_indices[key] for key in contrastive_seq_name))
                    if len(available_indices) >= self.args.num_pos_samples and len(available_contrastive_indices) >= self.args.num_neg_samples:
                        indices = np.random.choice(available_indices, self.args.num_pos_samples, replace=False)
                        contrastive_indices = np.random.choice(available_contrastive_indices, self.args.num_neg_samples, replace=False)
                        valid = True

            features = init_set[indices]
            features_mean = np.mean(features, axis=0, keepdims=True)
            codes = self.compute_codes(features_mean)[0]
            
            # zero-cost NAS
            if self.args.contrastive_search and 'SC' in self.args.indicator_name:
                indices = np.concatenate([indices, contrastive_indices])
                
            samples = samples_set[indices]
            data_item = slice_dict_by_indices(data_items_set, indices, self.args.mask_num, masks_num_set)
            print(data_item['image_path'])
            self.searcher.samples = (samples.cuda(), to_device(data_item, 'cuda'))
            # self.searcher.pos_num = int(len(indices) / 2)
            self.searcher.pos_num = self.args.num_pos_samples
            
            config = self.searcher.search()
            self.reset_searcher()

            new_row = {'Index': codes, 'Config': config, 'Error': 0, 'Source':-1}
            if not self.lookup['Index'].isin([new_row['Index']]).any():    
                self.lookup = pd.concat([self.lookup, pd.DataFrame([new_row])], ignore_index=True)
    
    def lookup_init_from_neg_loader(self, init_set, samples_set, data_items_set, masks_num_set, num_entry, bs):
        seq_name_to_indices = defaultdict(list)
        for idx, path in enumerate(data_items_set['image_path']):
            seq_name = get_seq_name(path, self.args)
            seq_name_to_indices[seq_name].append(idx)
        
        for _ in range(num_entry):
            valid = False
            while not valid: # find indices of images in the same video
                if not self.args.contrastive_search:
                    selected_seq_name = np.random.choice(list(seq_name_to_indices.keys()))
                    available_indices = seq_name_to_indices[selected_seq_name]
                    if len(available_indices) >= bs:
                        indices = np.random.choice(available_indices, bs, replace=False)
                        valid = True
                else:
                    raise NotImplementedError(
                        'lookup_init_from_neg_loader requires datasets.coco_img_load.load_and_transform_images, '
                        'which has been removed in this pre-release version. Use --look_up_init_from sequence instead.'
                    )
                    
            # pos sample & feature
            pos_samples = samples_set[indices]
            features = init_set[indices]
            features_mean = np.mean(features, axis=0, keepdims=True)
            codes = self.compute_codes(features_mean)[0]
            
            # zero-cost NAS
            samples = torch.cat((pos_samples, neg_samples), dim=0)
            
            if self.args.contrastive_search and 'SC' in self.args.indicator_name:
                indices = np.concatenate([indices, contrastive_indices])
            data_item = slice_dict_by_indices(data_items_set, indices, self.args.mask_num, masks_num_set)
            print(data_item['image_path'])
            self.searcher.samples = (samples.cuda(), to_device(data_item, 'cuda'))
            # self.searcher.pos_num = int(len(indices) / 2)
            self.searcher.pos_num = self.args.num_pos_samples
            
            config = self.searcher.search()

            # reset searcher
            self.reset_searcher()

            new_row = {'Index': codes, 'Config': config, 'Error': 0, 'Source':-1}
            if not self.lookup['Index'].isin([new_row['Index']]).any():    
                self.lookup = pd.concat([self.lookup, pd.DataFrame([new_row])], ignore_index=True)
            
    
    def retrain_check(self):
        adapter_configs = self.lookup['Config'].tolist()
        # remove duplicates
        adapter_configs = [tuple(x) for x in set(tuple(x) for x in adapter_configs)]
        # filter trained config
        filtered_configs = []
        for config in adapter_configs:
            adapter_config_str = "_".join(map(str, config))
            file_path = f"{self.args.retrain_root}/{self.args.dataset_save_name}/retrain/{self.args.model_name}_{self.args.dataset_save_name}_{adapter_config_str}_eps{self.args.retrain_epochs}/mutable_{self.args.backend}_{self.args.retrain_epochs}_.pth"
            if not os.path.exists(file_path):
                filtered_configs.append(config)
        return filtered_configs

    def model_retrain(self):
    
        adapter_configs = self.retrain_check()

        num_configs = len(adapter_configs)
        max_workers = len(self.args.retrain_gpus)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in range(math.ceil(num_configs/max_workers)):
                start_idx = i * max_workers
                end_idx = min((i + 1) * max_workers, len(adapter_configs))
                config_batch = adapter_configs[start_idx:end_idx]
                retrain_models = []
                for adapter_config, gpu_id in zip(config_batch, self.args.retrain_gpus):
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    retrain_models.append(executor.submit(run_training, adapter_config, self.args))
                    time.sleep(60) # make sure run_training is on different gpu
                for retrain_model in retrain_models:
                    retrain_model.result()
    
    def compute_codes(self, x):
        if self.nbits == 8:
            return self.pq.compute_codes(x)
        elif self.nbits == 4 and self.M >= 2:
            return split_uint8_to_uint4(self.pq.compute_codes(x))
        elif self.nbits == 2 and self.M >= 2:
            return split_uint8_to_uint2(self.pq.compute_codes(x))
        else:
            return self.pq.compute_codes(x)

    def decode(self, x):
        return self.pq.decode(self.compute_codes(x))
    
    def match(self, x):
        # s_time = time.time()
        indices  = np.array(self.lookup['Index'].tolist()).astype('uint8')
        sources = self.lookup['Source'].tolist()
        x = np.mean(x, axis=0, keepdims=True)
        # print(x.size)
        codes = self.compute_codes(x)
        code = codes[0]
        if indices.size == 0:
            # return config, row, error, code
            return None, None, self.M, code
        max_matching_bits = 0
        # result_index = None
        row = None
        for enum_i, index in enumerate(indices):
            if sources[enum_i] == -1:
                matching_bits = np.sum(index == code)
                if matching_bits == len(code):
                    # result_index = index
                    row = enum_i
                    max_matching_bits = len(code)
                    break
                elif matching_bits > max_matching_bits:
                    max_matching_bits = matching_bits
                    # result_index = index
                    row = enum_i
            else:
                continue
        if row is None:
            print('no match')
            row = random.randint(0, len(indices)-1)
            # result_index = indices[row]
            config = self.lookup['Config'].tolist()[row]
            error = self.M
        else:
            config = self.lookup['Config'].tolist()[row]
            error = self.M - max_matching_bits
            if error != 0 and self.update:
                new_row = {'Index': code, 'Config': config, 'Error': error, 'Source':row}
                self.lookup = pd.concat([self.lookup, pd.DataFrame([new_row])], ignore_index=True)
        
        # print(result_index)
        # e_time = time.time()
        # print(f"Time of match: {e_time-s_time} sec")

        return config, row, error, code
    
    def match_new(self, x):
        x = np.mean(x, axis=0, keepdims=True)
        codes = self.compute_codes(x)
        code = codes[0]

        if self.lookup.empty:
            return None, None, self.M, code

        latest_row = self.lookup.iloc[-1]
        latest_config = latest_row['Config']
        row = len(self.lookup) - 1
        error = self.M

        return latest_config, row, error, code
    
    def re_search(self, samples, neg_samples, codes):
        if neg_samples is not None:
            all_samples = concatenate_dicts([samples, neg_samples])
        else:
            all_samples = samples
        self.searcher.samples = (all_samples["image"], all_samples)
        self.searcher.pos_num = self.args.num_pos_samples
        config = self.searcher.search()

        re_search_time = None
        # config = [0] * 10 + [0.25] * 20

        # reset searcher
        if self.args.search_type == 'random':
            self.searcher.memory = []
            self.searcher.vis_dict = {}
            self.searcher.top = {}
            self.searcher.epoch = 0
            self.searcher.candidates = []
            self.searcher.top_accuracies = []
            self.searcher.cand_params = []
            self.searcher.all_res = []
        elif self.args.search_type in ('evolution', 'contrastive', 'elitist_evolution', 'hill_climbing'):
            self.searcher.memory = []
            self.searcher.vis_dict = {}
            self.searcher.keep_top_k = {self.searcher.select_num: [], 50: []}
            self.searcher.epoch = 0
            self.searcher.candidates = []
            self.searcher.top_accuracies = []
            self.searcher.cand_params = []
        else:
            raise ValueError(
                f"Unknown search type: {self.args.search_type}. Please choose from "
                "'random', 'evolution', 'elitist_evolution', 'hill_climbing', or 'contrastive'."
            )
        
        new_row = {'Index': codes, 'Config': config, 'Error': 0, 'Source':-1}
        if not self.lookup['Index'].isin([new_row['Index']]).any():
            self.lookup = pd.concat([self.lookup, pd.DataFrame([new_row])], ignore_index=True)

        # retrain
        if self.args.subnet_weight == 'retrain':
            self.searcher.supernet.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            while True:
                self.model_retrain()
                filtered_configs = self.retrain_check()
                if len(filtered_configs) == 0:
                    break
            self.searcher.supernet.cuda()
        return config, re_search_time

    def save(self, save_path, lookup=True):
        pq_state = {
            "d": self.pq.d,
            "M": self.pq.M,
            "nbits": self.pq.nbits,
            "centroids": faiss.vector_to_array(self.pq.centroids)
        }
        np.save(os.path.join(save_path,"product_quantizer.npy"), pq_state)
        if lookup:
            if self.M == 1024:
                self.lookup['Index'] = self.lookup['Index'].apply(lambda x: json.dumps(x.tolist()))
            self.lookup.to_csv(os.path.join(save_path,'lookup.csv'))
            if self.M == 1024:
                self.lookup['Index'] = self.lookup['Index'].apply(lambda x: np.array(json.loads(x)))
    
    def load(self, save_path):
        # load pq
        loaded_pq_state = np.load(os.path.join(save_path,"product_quantizer.npy"), allow_pickle=True).item()
        self.pq = faiss.ProductQuantizer(loaded_pq_state["d"], loaded_pq_state["M"], loaded_pq_state["nbits"])
        faiss.copy_array_to_vector(loaded_pq_state["centroids"], self.pq.centroids)
        # load lookup
        self.lookup = pd.read_csv(os.path.join(save_path,'lookup.csv'))


if __name__ == '__main__':
    d = 256
    M = 4
    nbits = 2
    num_entry = 100
    bs = 4
    x_t = np.random.random([1000, 256]).astype('float32')
    pq = faiss.ProductQuantizer(d, M, nbits)
    pq.train(x_t)

    pq_state = {
        "d": pq.d,
        "M": pq.M,
        "nbits": pq.nbits,
        "centroids": faiss.vector_to_array(pq.centroids)
    }
    np.save("product_quantizer_pq.npy", pq_state)

    loaded_pq_state = np.load("product_quantizer_pq.npy", allow_pickle=True).item()
    loaded_pq = faiss.ProductQuantizer(loaded_pq_state["d"], loaded_pq_state["M"], loaded_pq_state["nbits"])
    faiss.copy_array_to_vector(loaded_pq_state["centroids"], loaded_pq.centroids)

    x = np.random.random([4, 256]).astype('float32')

    codes = pq.compute_codes(x)
    loaded_codes = loaded_pq.compute_codes(x)

    print(codes)
    print(split_uint8_to_uint2(codes))
    print(loaded_codes)
    print(split_uint8_to_uint2(loaded_codes))
    
