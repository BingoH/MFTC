import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import os
import copy
import random
import argparse

from utils import *

def eval(data_loader, model, top=(1,), device=None, print_freq=10):
    total = 0
    maxk = max(topk)
    tp = torch.zeros(maxk)

    model = model.to(device)
    if device is not None:
        model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        for i, (imgs, labels) in enumerate(data_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            prob = model(imgs)
            _, pred = prob.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()
            corr = pred.eq(labels.view(1, -1).expand_as(pred))

            total += imgs.size(0)
            tp += corr.cpu().sum(dim=1)

            if (i+1) % print_freq == 0:
                print('Processed batch: {}/{}, images: {}, tp: {}'
                        .format(i+1, len(data_loader), total, tp))

    acc = tp.float() / float(total)
    ret = []
    for k in topk:
        ret.append(acc[k-1].item())
    return ret


def random_drop_features(model, ratio=0.5, layers=None):
    """
    modify network by random select feature channels
    currently we only support sequential model
    params:
        ratio: drop ratio
        layres: layres to modify, if not given, modify all conv layers
    """
    # TODO: DAGs like ResNet model
    # TODO: squeeze model after modify

    def random_choice(ratio, n):
        selected_n = int(ratio * n)
        selected_idx = torch.randperm(n)[:selected_n]
        selected_idx, _ = torch.sort(selected_idx)
        return selected_idx

    if isinstance(model, models.ResNet):
        print('Not supported yet')
        return None

    scale = 1. / (1. - ratio)

    modified_model = copy.deepcopy(model)
    if layers is None:
        layers = [ name for name, _ in modified_model.named_modules() ]
    layers = set(layers)

    for name, module in modified_model.named_modules():
        if isinstance(module, nn.Conv2d) and name in layers:
            print('\tupdate layer: ', name)
            idx = random_choice(ratio, module.out_channels)

            # TODO: replace usage of .data
            module.weight.data[idx] = 0
            module.weight.data = module.weight.data * scale

            if module.bias is not None:
                module.bias.data[idx] = 0
                module.bias.data = module.bias.data * scale

    return modified_model

def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('-a', '--arch', type=str, default='vgg16',
                        help='network architecture, default vgg16')
    parser.add_argument('--ckp', type=str,
                        help='checkpoint, if not given, use pytorch official pretrained model')

    # dataset
    parser.add_argument('-d', '--dataset', type=str,
                        help='dataset to evaluate')
    parser.add_argument('--data_root', type=str, default='./data/',
                        help='data root, default ./data')
    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help='batch size, default 256')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loader workers, default 4')
    parser.add_argument('--val', action='store_true',
                        help='if enabled, use validation(or test) dataset')

    # drop features
    parser.add_argument('-r', '--ratio', type=float, default=0.5,
                        help='ratio of feature channels to drop, default 0.5')
    parser.add_argument('-l', '--layers', type=str,
                        help='layers to modify, split by comma')

    # common
    parser.add_argument('--seed', type=int, default=12345,
                        help='random seed, default 12345')
    parser.add_argument('--no_cuda', action='store_true',
                        help='no cuda if enabled')

    parser.add_argument('-k','--topk', type=str, default='1',
                        help='topk accuracy to evaluate, split by comma; default 1')
    return parser.parse_args()

def main(args):
    #====  log ====#
    print('#'*20, 'configure', '#'*20)
    for k, v in vars(args).items():
        print('  {} : {}'.format(k, v))
    print('#'*20, 'configure', '#'*20)

    #====  setting ====#
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device('cpu' if args.no_cuda else 'cuda')
    topk = list(map(int, args.topk.split(',')))

    #====  data ====#
    is_train = False if args.val else True
    dataset = create_dataset(args.dataset, args.data_root, is_train=is_train)
    data_loader = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, drop_last=False)

    #====  model ====#
    model = create_model(args.arch)
    if args.ckp is not None:
        state_dict = torch.load(args.ckp)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)

    #====  evaluate on original and modified model ====#

    # random drop features
    layers = None
    if args.layers is not None:
        layers = args.layers.split(',')

    model_m = random_drop_features(model, args.ratio, layers)

    # evaluate
    model = model.to(device)
    pre_acc = eval(data_loader, model, device=device, topk=topk)
    model_m = model_m.to(device)
    post_acc = eval(data_loader, model_m, device=device, topk=topk)

    print('Accuracy of original {} on {}: {}'.format(args.arch, args.dataset, pre_acc))
    print('Accuracy of modified {} on {}: {}'.format(args.arch, args.dataset, post_acc))


if __name__ == '__main__':
    args = parse_args()
    main(args)
