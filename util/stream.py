
from typing import List, Tuple, Dict, Optional, Callable
import torch
from collections import deque
from utils import to_device
import random
import copy

def segment_paths_by_prefix(
    image_paths: List[str], 
    prefix_func: Callable[[str], str], 
    label_paths: Optional[List[str]] = None
) -> Tuple[List[str], Optional[List[str]], Dict[str, List[int]]]:
   
    
    sorted_paths = sorted(image_paths)
    if label_paths is not None:
        label_paths = sorted(label_paths)
    
    segments = {}
    current_prefix = None
    start_index = 0
    
    for i, path in enumerate(sorted_paths):
        prefix = prefix_func(path)
        if prefix != current_prefix:
            if current_prefix is not None:
                segments[current_prefix] = [start_index, i - 1]
            current_prefix = prefix
            start_index = i
    
    if current_prefix is not None:
        segments[current_prefix] = [start_index, len(sorted_paths) - 1]
    
    return sorted_paths, label_paths, segments

def concatenate_dicts(dict_list):
    keys = dict_list[0].keys()
    result_dict = {}
    for key in keys:
        if key in ['image', 'label', 'boxes', 'point_coords', 'point_labels', 'mask_nums']:
            result_dict[key] = torch.cat([d[key] for d in dict_list], dim=0)
        elif key in ['image_path']:
            result_list = []
            for d in dict_list:
                result_list = result_list + d[key]
            result_dict[key] = result_list
    return result_dict

class FIFOCache:
    def __init__(self, capacity, mask_num):
        self.cache = deque()
        self.capacity = capacity
        self.mask_num = mask_num

    def put(self, value):
        
        # too many masks result in memory issue
        if len(value['label']) > self.mask_num:
            value_copy = copy.deepcopy(value)
            value = value_copy
            selected_indices = random.sample(range(value['mask_nums']), self.mask_num)
            value['label'] = value['label'][selected_indices]
            value['boxes'] = value['boxes'][selected_indices]
            value['point_coords'] = value['point_coords'][selected_indices]
            value['point_labels'] = value['point_labels'][selected_indices]
            value['mask_nums'] = torch.tensor([self.mask_num], device=value['mask_nums'].device, dtype=value['mask_nums'].dtype)

        value = to_device(value, 'cpu')
        if len(self.cache) >= self.capacity:
            self.cache.popleft()
        self.cache.append(value)

    def get_first_n(self, n):
        
        return to_device(concatenate_dicts(list(self.cache)[:n]), 'cuda')

    def get_last_n(self, n):
        
        return to_device(concatenate_dicts(list(self.cache)[-n:]), 'cuda')
    
    def is_full(self):
        return len(self.cache) >= self.capacity
    
    def get_length(self):
        return len(self.cache)
