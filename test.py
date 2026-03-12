import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import vgg
import thop


def ratio_or_inf(numerator, denominator):
    if denominator == 0:
        return float('inf')
    return numerator / denominator


# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--dir_data', default='', type=str, metavar='PATH',
                    help='refine from prune model')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=11,
                    help='depth of the vgg')
parser.add_argument('--baseline', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--pruned', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
parser.add_argument('--finetune', default='', type=str, metavar='PATH',
                    help='path to the model (default: none)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

def test(model):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'cifar10':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(args.dir_data, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar100':
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(args.dir_data, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise ValueError("No valid dataset is given.")
    model.eval()
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct / float(len(test_loader.dataset))

# 加载剪枝前VGG11的Baseline
model = vgg.vgg(dataset=args.dataset, depth=args.depth)
print("=> loading checkpoint '{}'".format(args.baseline))
checkpoint = torch.load(args.baseline, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'], strict=False)

if args.cuda:
    model.cuda()
# 测试剪枝前模型精度
baseline_acc = test(model)
# 初始化一个数据输入模型，计算剪枝前模型的参数量与计算量
x = torch.randn(1, 3, 32, 32)
if args.cuda:
    x = x.cuda()
baseline_flops, baseline_params = thop.profile(model, inputs=(x,))
print("before prune:")
print("params:", baseline_params)
print("FLOPs:", baseline_flops)


# 加载剪枝后微调前的模型
cfg = [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256]
newmodel_pruned = vgg.vgg(dataset=args.dataset, depth=args.depth, cfg=cfg)
print("=> loading checkpoint '{}'".format(args.pruned))
checkpoint = torch.load(args.pruned, map_location='cpu')
newmodel_pruned.load_state_dict(checkpoint['state_dict'], strict=False)

if args.cuda:
    newmodel_pruned.cuda()

# 测试剪枝后微调前的模型精度
pruned_acc = test(newmodel_pruned)

# 同样初始化一个数据输入模型，计算剪枝后微调前模型的参数量与计算量
x = torch.randn(1, 3, 32, 32)
if args.cuda:
    x = x.cuda()
pruned_flops, pruned_params = thop.profile(newmodel_pruned, inputs=(x,))
print("after prune, before finetune:")
print("params:", pruned_params)
print("FLOPs:", pruned_flops)


# 加载微调后的模型
cfg = [64, 'M', 128, 'M', 256, 256, 'M', 256, 256, 'M', 256, 256]
newmodel_finetune = vgg.vgg(dataset=args.dataset, depth=args.depth, cfg=cfg)
print("=> loading checkpoint '{}'".format(args.finetune))
checkpoint = torch.load(args.finetune, map_location='cpu')
newmodel_finetune.load_state_dict(checkpoint['state_dict'], strict=False)

if args.cuda:
    newmodel_finetune.cuda()

# 测试微调后模型精度
finetune_acc = test(newmodel_finetune)

# 同样初始化一个数据输入模型，计算微调后模型的参数量与计算量
x = torch.randn(1, 3, 32, 32)
if args.cuda:
    x = x.cuda()
finetune_flops, finetune_params = thop.profile(newmodel_finetune, inputs=(x,))
print("after finetune:")
print("params:", finetune_params)
print("FLOPs:", finetune_flops)

print("\n==================== Summary ====================")
print("{:<28} {:>10} {:>15} {:>15}".format("Model", "Acc(%)", "Params", "FLOPs"))
print("{:<28} {:>10.2f} {:>15,.0f} {:>15,.0f}".format(
    "Baseline", baseline_acc * 100.0, baseline_params, baseline_flops))
print("{:<28} {:>10.2f} {:>15,.0f} {:>15,.0f}".format(
    "Pruned (before finetune)", pruned_acc * 100.0, pruned_params, pruned_flops))
print("{:<28} {:>10.2f} {:>15,.0f} {:>15,.0f}".format(
    "Pruned (after finetune)", finetune_acc * 100.0, finetune_params, finetune_flops))

param_reduction = (1.0 - ratio_or_inf(pruned_params, baseline_params)) * 100.0
flops_reduction = (1.0 - ratio_or_inf(pruned_flops, baseline_flops)) * 100.0
param_compression = ratio_or_inf(baseline_params, pruned_params)
flops_speedup = ratio_or_inf(baseline_flops, pruned_flops)
acc_drop_prune = (baseline_acc - pruned_acc) * 100.0
acc_drop_finetune = (baseline_acc - finetune_acc) * 100.0
recovery_ratio = ratio_or_inf((finetune_acc - pruned_acc), (baseline_acc - pruned_acc)) * 100.0

print("-------------------------------------------------")
print("Accuracy drop after prune: {:.2f} pp".format(acc_drop_prune))
print("Accuracy drop after finetune vs baseline: {:.2f} pp".format(acc_drop_finetune))
print("Accuracy recovery by finetune: {:.2f}%".format(recovery_ratio))
print("Parameter reduction: {:.2f}% ({:.2f}x smaller)".format(param_reduction, param_compression))
print("FLOPs reduction: {:.2f}% ({:.2f}x theoretical speedup)".format(flops_reduction, flops_speedup))
print("=================================================\n")
