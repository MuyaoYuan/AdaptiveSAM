import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
from . import indicator
from ..p_utils import has_gradient, register_hook_if_not_exists

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for module in net.modules():
            if hasattr(module, 'Adapters_list') and module.Adapters_list:
                for adapter in module.Adapters_list:
                    for m in adapter.modules():
                        if isinstance(m, nn.Conv2d):
                            nn.init.normal_(m.weight)
                            if hasattr(m, 'bias') and m.bias is not None:
                                nn.init.zeros_(m.bias)
                        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                            nn.init.ones_(m.weight)
                            nn.init.zeros_(m.bias)
                        elif isinstance(m, nn.Linear):
                            nn.init.normal_(m.weight)
                            if hasattr(m, 'bias') and m.bias is not None:
                                nn.init.zeros_(m.bias)
                        else:
                            continue

    return net

def sc_score(K,n,lamda=2):
    Na = K[0,0]
    m = K.shape[0] - n    
    K_neg = K[n:, n:]
    K_pos_sum = np.sum(K) - lamda*np.sum(K_neg)
    if lamda < 1:
        return np.log(Na * (n**2 + 2*m*n - (lamda-1) * m**2) - K_pos_sum)
    else:
        return np.log(Na * (n**2 + 2*m*n - (lamda-1) * m) - K_pos_sum)


def normalized_cut(KH, k):
    n = KH.shape[0]
    
    # Calculate Intra-Class Similarity for positive samples
    intra_class_positive = np.sum(KH[:k, :k]) - np.sum(np.diag(KH[:k, :k]))
    intra_class_positive /= k * (k - 1)
    
    # Calculate Intra-Class Similarity for negative samples
    intra_class_negative = np.sum(KH[k:, k:]) - np.sum(np.diag(KH[k:, k:]))
    intra_class_negative /= (n - k) * (n - k - 1)
    
    # Calculate Inter-Class Similarity
    inter_class_similarity = np.sum(KH[:k, k:])
    inter_class_similarity /= k * (n - k)
    
    # Calculate Normalized Cut
    normalized_cut_value = intra_class_negative / (inter_class_similarity+ 2* intra_class_positive)
    
    return normalized_cut_value
    
def get_batch_jacobian(net, x):
    net.zero_grad()
    x.requires_grad_(True)
    # y = net(x)
    y_attn_features = net.image_encoder.forward(x)
    y_attn_features.backward(torch.ones_like(y_attn_features))
    jacob = x.grad.detach()
    # return jacob, target.detach(), y.detach()
    return jacob, y_attn_features.detach()

@indicator('SC', bn=False, mode='param', copy_net=True)
def compute_nas_score(model, input, mode, split_data=1, sam_net=None, feat_criterion=None, data_item=None, pos_num=1, args=None):
    
    batch_size = input.shape[0]
    
    if sam_net:
        sam_net.eval()
    model.train()
    model.zero_grad()
    
    network_weight_gaussian_init(model)
    model.K = np.zeros((batch_size, batch_size))

    def counting_forward_hook(module, inp, out):
        if isinstance(inp, tuple):
            inp = inp[0]
        inp = inp.reshape(inp.size(0), -1)
        x = (inp > 0).float()
        K = x @ x.t()
        K2 = (1. - x) @ (1. - x.t())
        model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    for module in model.modules():
        if hasattr(module, 'Adapters_list') and module.Adapters_list:
            for adapter in module.Adapters_list:
                if has_gradient(adapter):
                    for layer in adapter.modules():
                    # if 'GeLU' in str(type(module)):
                        if isinstance(layer, torch.nn.GELU):
                            # hooks[name] = module.register_forward_hook(counting_hook)
                            module.visited_backwards = True
                            # module.register_forward_hook(counting_forward_hook) 
                            register_hook_if_not_exists(module, counting_forward_hook)
                            # module.register_backward_hook(counting_backward_hook)
            
    x = input
    
    model.eval()
    with torch.no_grad():
        attn_features = model.image_encoder.forward(x)
    
    # plot_normalized_K(model.K, save_path='normalized_K.png')

    score = sc_score(model.K, pos_num, args.sc_lamda)
    
    # score_cut = normalized_cut(model.K, pos_num)
    del model.K

    return float(score)
