import sys
sys.path.insert(0,  '.')

from pytorch_simple_classification_baselines.nets.mnist_lenet import LeNet
from pytorch_simple_classification_baselines.utils.preprocessing import *

import torchvision
import torchvision.datasets as datasets
import torchvision.models as models


def create_dataset(dataset, data_root, is_train):
    """
    create datasets
    """
    if dataset == 'imagenet':
        transform = imgnet_transform(is_training=is_train)
        split = 'train' if is_train else 'val'
        ret = datasets.ImageNet(root=data_root, split=split, download=True, transform=transform)
    elif dataset == 'mnist':
        transform =  minst_transform(is_training=is_train)
        print(data_root)
        ret = datasets.MNIST(root=data_root, train=is_train, download=True, transform=transform)
    elif dataset == 'cifar10':
        transform = cifar_transform(is_training=is_train)
        ret = datasets.CIFAR10(root=data_root, train=is_train, download=True, transform=transform)
    elif dataset == 'cifar100':
        transform = cifar_transform(is_training=is_train)
        ret = datasets.CIFAR100(root=data_root, train=is_train, download=True, transform=transform)
    else:
        print('Got dataset: {}, unfortunately not supported yet'.format(dataset))
        return None

    return ret


def create_model(arch):
    __factory = {
            'resnet18' : models.resnet18,
            'resnet50' : models.resnet50,
            'vgg16' : models.vgg16,
            'vgg19' : models.vgg19,
            'alexnet' : models.alexnet,
            'lenet' : LeNet,
            }

    if arch in __factory:
        if arch == 'lenet':
            kwargs = {}
        else:
            kwargs = { 'pretrained': True }
        model = __factory[arch](**kwargs)
    else:
        print('Got network arch: {}, unfortunately not supported yet'.format(arch))
        return None
    return model
