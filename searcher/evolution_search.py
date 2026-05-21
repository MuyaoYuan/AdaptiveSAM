import random
import numpy as np
import time
import torch

import argparse
import os

from training_free import *

from segment_anything.modeling import TinyViTWithFuseSuperNetSampler, sample_adapter_configuration, sample_mlp_configuration
from segment_anything import sam_model_registry

from dataset.camvid_sa import CamVidSA
from utils import to_device, stack_dict_batched

from torch.utils.data import DataLoader

def has_gradient(module):
    return any(param.requires_grad for param in module.parameters())

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def calculate_adapter_params(model):
    total_params = 0
    for module in model.modules():
        if hasattr(module, 'Adapters_list'):
            if module.adapter_type != -1:
                adapters_list = module.Adapters_list
                for adapter in adapters_list:
                    if has_gradient(adapter):
                        total_params += count_parameters(adapter)
    return total_params

def parse_int_list(value):
    try:
        return [int(x) for x in value.split(',')]
    except:
        raise ValueError('Numbers must be integers separated by commas.')
    

class EvolutionSearcher(object):

    def __init__(self, args, supernet, efficient_sam_without_ddp, train_loader, samples, sam, output_dir=None, pos_num=1):

        self.args = args
        self.indicator_name=args.indicator_name
        self.supernet = supernet
        self.supernet_type = args.supernet_type
        self.efficient_sam_without_ddp = efficient_sam_without_ddp
        self.train_loader = train_loader
        self.samples = samples
        self.sam = sam
        
        self.adapter_option = args.adapter_option
        self.mlp_option = args.mlp_option
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.s_prob =args.s_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        
        self.parameters_limits = args.param_limits
        self.min_parameters_limits = args.min_param_limits

        self.output_dir = output_dir
        self.pos_num = pos_num
        
        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []
        self.top_accuracies = []
        self.cand_params = []
        self.init_population = None

    def is_legal(self, cand):
        assert isinstance(cand, tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand] = {}
        info = self.vis_dict[cand]
        
        if 'visited' in info:
            return False
        
        if all(x == -1 for x in cand):
            return False
        if self.supernet_type == 1:
            self.supernet.sample_path(cand)
        elif self.supernet_type == 2:
            self.supernet.sample_path(cand[:10], cand[10:])
        else:
            raise ValueError(f'No such supernet type {self.supernet_type}')

        n_parameters = calculate_adapter_params(self.supernet)
        info['params'] = n_parameters / 10. ** 6

        if self.args.param_type == 'params':
            if info['params'] > self.parameters_limits:
                print('parameters limit exceed')
                return False

            if info['params'] < self.min_parameters_limits:
                print('under minimum parameters limit')
                return False
                
        elif self.args.param_type == 'adapters':
            if cand.count(1) > self.parameters_limits:
                print('parameters limit exceed')
                return False

            if cand.count(1) < self.min_parameters_limits:
                print('under minimum parameters limit')
                return False
        else:
            raise NotImplementedError(f'{self.args.param_type} is not yet implemented')

        indicators = compute_indicators.find_indicators(self.efficient_sam_without_ddp,
                                                        ('in_lookup', 1, 1000),
                                                        device='cuda',
                                                        sam_net=self.sam,
                                                        indicator_names=self.indicator_name,
                                                        dataloader=self.train_loader,
                                                        samples=self.samples,
                                                        args=self.args,
                                                        pos_num=self.pos_num)

        indicator_name = self.indicator_name[0]
        indicators = indicators[indicator_name]

        info['indicator'] = indicators
        info['visited'] = True

        return True

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k] # keep the top-k

        # current_top_k = self.keep_top_k[k]
        # combined_candidates = current_top_k + candidates
        # unique_candidates = list(set(combined_candidates))
        # unique_candidates.sort(key=key, reverse=reverse)
        # self.keep_top_k[k] = unique_candidates[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):
        """
        iterate the sampled config
        """
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
                info = self.vis_dict[cand]
            for cand in cands:
                yield cand

    def get_random_cand(self):
        """
        get one random config
        """
        config = sample_adapter_configuration(num_decisions=10, options=self.adapter_option)
        if self.supernet_type == 2:
            mlp_comfig = sample_mlp_configuration(num_decisions=10*2, options=self.mlp_option)
            return tuple(config + mlp_comfig)
        return tuple(config)

    def get_random(self, num):
        """
        search, check parameter of sampled configs
        """
        print('random select ........')
        cand_iter = self.stack_random_cand(self.get_random_cand)
        while len(self.candidates) < num:
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            self.candidates.append(cand)
        #     print('random {}/{}'.format(len(self.candidates), num))
        # print('random_num = {}'.format(len(self.candidates)))

    def get_init_population(self, population):
        """
        init population for search
        population is a list or tuple of configs
        """
        print('init population ........')
        for p in population:
            if not self.is_legal(p):
                continue
            self.candidates.append(p)

    def get_mutation(self, k, mutation_num, m_prob, s_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(random.choice(self.keep_top_k[k]))
            random_s = random.random()

            # Decide whether to mutate extensively based on s_prob
            if random_s < s_prob:
                # Extensive mutation: each element in the list has a probability of m_prob to mutate
                if self.supernet_type == 1:
                    for i in range(len(cand)):
                        if random.random() < m_prob:
                            cand[i] = random.choice(self.adapter_option)
                elif self.supernet_type == 2:
                    for i in range(10):
                        if random.random() < m_prob:
                            cand[i] = random.choice(self.adapter_option)
                    for i in range(10, 10+2*10):
                        if random.random() < m_prob: 
                            cand[i] = random.choice(self.mlp_option) # mlp option
                else:
                    raise ValueError(f'No such supernet type {self.supernet_type}')
            else:
                # Small-scale mutation: mutate only one random position
                if self.supernet_type == 1:
                    index_to_mutate = random.randint(0, len(cand) - 1)
                    cand[index_to_mutate] = random.choice(self.adapter_option)
                elif self.supernet_type == 2:
                    index_to_mutate = random.randint(0, len(cand) - 1)
                    if index_to_mutate <= 9:
                        cand[index_to_mutate] = random.choice(self.adapter_option)
                    else:
                        cand[index_to_mutate] = random.choice(self.mlp_option) # mlp option
                else:
                    raise ValueError(f'No such supernet type {self.supernet_type}')
                
            return tuple(cand)

        # generate mutated cand and start mutation
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{}'.format(len(res), mutation_num))
        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():

            p1 = random.choice(self.keep_top_k[k])
            p2 = random.choice(self.keep_top_k[k])
            max_iters_tmp = 50
            while len(p1) != len(p2) and max_iters_tmp > 0:
                max_iters_tmp -= 1
                p1 = random.choice(self.keep_top_k[k])
                p2 = random.choice(self.keep_top_k[k])
            return tuple(random.choice([i, j]) for i, j in zip(p1, p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand = next(cand_iter)
            if not self.is_legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{}'.format(len(res), crossover_num))
        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        
        # init population 
        if self.init_population is not None:
            self.get_init_population(self.init_population)

        # initialize
        self.get_random(self.population_num)

        if self.max_epochs == 0:
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['indicator'])
            
        # search & iterate
        while self.epoch < self.max_epochs:
            print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)

            #  Update top k population based on zero-cost metric
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['indicator'])
            
            # Update top 50 population based on zero-cost metric
            # self.update_top_k(
            #     self.candidates, k=50, key=lambda x: self.vis_dict[x]['indicator'])

            print('epoch = {} : top {} result'.format(
                self.epoch, len(self.keep_top_k[self.select_num])))
            tmp_accuracy = []
            for i, cand in enumerate(self.keep_top_k[self.select_num]):
                print('No.{} {} Top-1 indicator = {},  params = {}'.format(
                    i + 1, cand, self.vis_dict[cand]['indicator'], self.vis_dict[cand]['params']))
                tmp_accuracy.append(self.vis_dict[cand]['indicator'])
            self.top_accuracies.append(tmp_accuracy)

            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob, self.s_prob)
            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1
   
        # Find and return the candidate with the highest indicator value
        # best_candidate_50 = max(self.keep_top_k[50], key=lambda x: self.vis_dict[x]['indicator'])
        # indicator_50 = self.vis_dict[best_candidate_50]['indicator']
        
        best_candidate_k = max(self.keep_top_k[self.select_num], key=lambda x: self.vis_dict[x]['indicator'])
        # indicator_k = self.vis_dict[best_candidate_k]['indicator']
        
        # return best_candidate_50
        return best_candidate_k


def parse_args():
    parser = argparse.ArgumentParser(description='Deep learning model training script')
    parser.add_argument('--data-path', type=str, help='Path to dataset folder')
    parser.add_argument('--save-path', type=str, default=None, help='Path for storing model snapshots')
    parser.add_argument('--backend', type=str, default='efficient_vit_h', help='Feature extractor')
    parser.add_argument('--snapshot', type=str, default='ckpt/sam_vit_h_4b8939.pth', help='Path to pretrained weights')
    parser.add_argument('--image_size', type=int, default=1024, help='Image size')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1.0, help='Coefficient for classification loss term')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs to run')
    parser.add_argument('--start-lr', type=float, default=0.001)
    parser.add_argument('--scale', type=float, default=1.0, help='scale param for augmentation')
    parser.add_argument('--atten_type', type=str, default='local', help='type of feature loss')
    parser.add_argument('--atten_k', type=int, default=7, help='type of feature loss')
    parser.add_argument('--ref_gap', type=int, default=2, help='The length of reference GOP.')
    parser.add_argument('--bitrate', type=int, default=3, help='bitrate of dataset.')
    parser.add_argument('--model_type', type=str, default='pspnet', help='model that we apply')
    parser.add_argument('--dataset', type=str, default='camvid', help='dataset')
    parser.add_argument('--fuse_version', type=int, default=1, help='Fusion version with different CReFF locations')
    parser.add_argument('--resume', type=bool, default=False, help='Resume from snapshot')
    parser.add_argument('--mask_num', type=int, default=5, help='number of sampling masks')
    parser.add_argument('--iterative_train', type=bool, default=False, help='iterative prompt training')
    parser.add_argument('--point_num', type=int, default=1, help='number of sampling point')
    parser.add_argument('--multimask', type=bool, default=True, help='ouput multimask')
    parser.add_argument('--no_multimask', dest='multimask', action='store_false')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--iter_point", type=int, default=8, help="iter num")
    parser.add_argument("--num_video", type=int, default=3000, help="video num of kinetics")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument('--frozen', type=bool, default=True, help='frozen early layers or not')
    parser.add_argument("--training_layers", type=int, default=3)
    parser.add_argument("--train_workers", type=int, default=8)
    parser.add_argument('--lr_warm_up', type=bool, default=False)
    parser.add_argument('--beta1', type=float, default=0.5, help='Coefficient for refine_i loss term')
    parser.add_argument('--beta2', type=float, default=0.5, help='Coefficient for refine_p loss term')
    parser.add_argument('--n_points_per_side', type=int, default=4)
    parser.add_argument('--batch_size_points', type=int, default=16)
    # BA
    parser.add_argument('--if_global', type=bool, default=False)
    # learning rate scheduler
    parser.add_argument('--lr_drop', type=int, default=2, help="lr decay epoch")
    parser.add_argument('--adapter_type', type=int, default=-1, help="-1=No Adapter, 0=Series Adapter, 1=Parallel Adapter, 2=Mixed Adapter, 3=LoRA")
    parser.add_argument('--decoder_ft', type=bool, default=False)
    parser.add_argument('--edge_token', type=bool, default=False)
    # parser.add_argument("--adapter_config", type=str, default="-1,-1,-1,-1,-1,-1", help="Adapter configuration")
    parser.add_argument("--sim_config", type=str, default='relation')
    parser.add_argument('--pretrained', type=bool, default=False)
    
    parser.add_argument("--adapter_config", type=str, default="-1,-1,-1,-1,-1,-1", help='The adapter config, \
                 len(config)==6 means only the third layer is adapted, len(config)==10 means all the layers except the first layer are adapted')
    
    # search parameters
    parser.add_argument('--indicator_name', default='dss', type=str)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--select-num', type=int, default=5)
    parser.add_argument('--population-num', type=int, default=20)
    parser.add_argument('--m_prob', type=float, default=0.2)
    parser.add_argument('--s_prob', type=float, default=0.4)
    parser.add_argument('--crossover-num', type=int, default=5)
    parser.add_argument('--mutation-num', type=int, default=5)
    parser.add_argument('--param-limits', type=float, default=1e10)
    parser.add_argument('--min-param-limits', type=float, default=0)
    parser.add_argument("--adapter_option", nargs='*', type=int, default=[-1, 2], help='adapter_option for supernet')
    parser.add_argument('--param-type', type=str, default='params', help='the type of param-limits')
    
    return parser.parse_args()


def search():

    args = parse_args()
    
    args.adapter_config = parse_int_list(args.adapter_config)

    os.makedirs(args.save_path, exist_ok=True)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if args.pretrained:
        sam = sam_model_registry["default"](checkpoint='ckpt/sam_vit_h_4b8939.pth')
        efficient_sam = sam_model_registry[args.backend](checkpoint=args.snapshot,
                                                    adapter_type=args.adapter_type,
                                                    adapter_config=args.adapter_config)
    else:
        sam = sam_model_registry["default"](checkpoint=None)
        efficient_sam = sam_model_registry[args.backend](checkpoint=None,
                                                    adapter_type=args.adapter_type,
                                                    adapter_config=args.adapter_config)
    # supernet, efficient sam
    efficient_sam = TinyViTWithFuseSuperNetSampler(efficient_sam, num_decisions=10)
    
    sam.to(device='cuda')
           
    starting_epoch = 0

    efficient_sam.to(device='cuda')
    efficient_sam.train()
    sam.eval()

    
    if args.dataset == 'camvid':
        cropsize = [args.image_size, args.image_size]
        randomscale = (0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
        train_ds_sa = CamVidSA(args.data_path, image_size=args.image_size, mode='train',
                                ref_gap=args.ref_gap,
                                mask_num=args.mask_num,
                                point_num=args.point_num,
                                )

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

    supernet = efficient_sam
    efficient_sam_without_ddp = efficient_sam.model

    # proxy metric
    if not isinstance(args.indicator_name, list):
        args.indicator_name = [args.indicator_name]

    for epoch in range(starting_epoch, args.epochs):
        train_iterator = train_loader_sa

        gt_feat_list = []
        stu_feat_list = []
        for idx, data_item in enumerate(train_iterator):
            
            if idx == 0:
                continue
            
            if args.dataset == 'camvid':
                data_item = stack_dict_batched(data_item)
                data_item = to_device(data_item, 'cuda')
                x = data_item["image"]
            else:
                raise NotImplementedError(f'Dataset {args.dataset} not supported in this pre-release version.')

            # get samples
            samples = (x, data_item['label'], data_item)
            
            t = time.time()
            searcher = EvolutionSearcher(args, supernet, efficient_sam_without_ddp, train_loader, samples, sam)

            searcher.search()

            print('total searching time = {:.2f} seconds'.format(
                (time.time() - t)))

            if idx >= 1:
                break
        
        break
        

    


if __name__ == '__main__':
    search()
