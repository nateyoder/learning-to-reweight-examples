import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.autograd import Variable
import itertools


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def set_named_parameter(module, name, val, zero_grads=True):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in module.named_children():
            if module_name == name:
                set_named_parameter(mod, rest, val)
                break
    else:
        if zero_grads and val.grad is not None:
            val.grad.detach_()
            val.grad.zero_()
        module._parameters[name] = val

def update_params_ny(model, lr_inner, source_params=None):
    for tgt, src in zip(model.named_parameters(), source_params):
        name_t, param_t = tgt
        grad = src
        tmp = param_t - lr_inner * grad
        set_named_parameter(model, name_t, tmp)


class LeNetNy(nn.Module):
    def __init__(self, n_out):
        super(LeNetNy, self).__init__()

        layers = list()
        layers.append(nn.Conv2d(1, 6, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(6, 16, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        layers.append(nn.Conv2d(16, 120, kernel_size=5))
        layers.append(nn.ReLU(inplace=True))

        self.main = nn.Sequential(*layers)

        layers = list()
        layers.append(nn.Linear(120, 84))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(84, n_out))

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.main(x)
        x = x.view(-1, 120)
        return self.fc_layers(x).squeeze()


def build_model(hyperparameters):
    net = LeNetNy(n_out=1)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark = True

    opt = torch.optim.SGD(net.parameters(), lr=hyperparameters["lr"])

    return net, opt
