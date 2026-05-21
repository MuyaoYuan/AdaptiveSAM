import torch
import torch.nn as nn
from utils import stack_dict_batched


def has_gradient(module):
    return any(param.requires_grad for param in module.parameters())

def merge_dicts_with_tensors(data_list, device):
    merged_dict = {}
    for d in data_list:
        for key, value in d.items():
            if key in merged_dict:
                merged_dict[key].append(value)
            else:
                merged_dict[key] = [value]
    
    for key, value_list in merged_dict.items():
        merged_dict[key] = torch.cat(value_list, dim=0).to(device)
    
    return merged_dict

def get_some_data(train_dataloader, num_batches, device):     
    
    data_items = []
    dataloader_iter = iter(train_dataloader)
    for _ in range(num_batches):
        data_item = next(dataloader_iter)
        data_items.append(data_item)
    
    # merge data_items into one dict
    data_items_dict = merge_dicts_with_tensors(data_items, device)
    data_items_dict = stack_dict_batched(data_items_dict)
    
    # size:  (bsz, 3, h, w), (bsz * num_prompt, 1, h, w)
    inputs, targets = data_items_dict['image'], data_items_dict['label']

    return inputs, targets, data_items_dict

def get_some_data_grasp(train_dataloader, num_classes, samples_per_class, device):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(train_dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    x = torch.cat([torch.cat(_, 0) for _ in datas]).to(device) 
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1).to(device)
    return x, y

def get_layer_metric_array(net, metric, mode, device):
    metric_array = []
    
    for module in net.modules():
        if hasattr(module, 'Adapters_list') and module.Adapters_list:
            for adapter in module.Adapters_list:
                if has_gradient(adapter):
                    for layer in adapter.modules():
                        if isinstance(layer, nn.Linear):
                            metric_array.append(metric(layer))
        # else:
        #     metric_array.append(torch.tensor(0).to(device))  

    return metric_array

def get_layer_metric_array_dss(net, metric, mode, device):
    metric_array = []
    
    for module in net.modules():
        if hasattr(module, 'Adapters_list') and module.Adapters_list:
            for adapter in module.Adapters_list:
                if has_gradient(adapter):
                    for layer in adapter.modules():
                        if isinstance(layer, nn.Linear):
                            metric_array.append(metric(layer))
        else:
            metric_array.append(torch.tensor(0).to(device))  

    return metric_array

def reshape_elements(elements, shapes, device):
    def broadcast_val(elements, shapes):
        ret_grads = []
        for e,sh in zip(elements, shapes):
            ret_grads.append(torch.stack([torch.Tensor(sh).fill_(v) for v in e], dim=0).to(device))
        return ret_grads
    if type(elements[0]) == list:
        outer = []
        for e,sh in zip(elements, shapes):
            outer.append(broadcast_val(e,sh))
        return outer
    else:
        return broadcast_val(elements, shapes)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def register_hook_if_not_exists(module, hook):
    if not hasattr(module, "_hooks"):
        module._hooks = []
    for h in module._hooks:
        if h == hook:
            return
    module._hooks.append(hook)
    module.register_forward_hook(hook)