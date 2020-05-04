import torch
import math
import torch.nn as nn
from abc import ABC, abstractmethod
import numpy as np


class BufferPPO():
    def __init__(
            self,
            img_stack=4,
            action_vec=0,
            buffer_capacity=5000
    ):
        self.buffer_capacity = buffer_capacity
        self.counter = 0
        # TODO ACTION VEC FOR 1 DONT NEED a_v, you can take just action
        if action_vec > 0:
            self._transition = np.dtype(
                [('s', np.float64, (img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                 ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96)),
                 ('a_v', np.float64, (3 * (action_vec + 1),))])
        else:
            self._transition = np.dtype(
                [('s', np.float64, (img_stack, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                 ('r', np.float64), ('s_', np.float64, (img_stack, 96, 96))])

        self.buffer = np.empty(self.buffer_capacity, dtype=self._transition)

    def store(self, transition):
        """
        Checks if buffer is full and save data
        """
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False


class Writer(ABC):
    log_dir = "runs"

    @abstractmethod
    def add_loss(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_evaluation(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_scalar(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_schedule(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_summary(self, name, mean, std, step="frame"):
        pass

class DummyWriter(Writer):
    def add_loss(self, name, value, step="frame"):
        pass

    def add_evaluation(self, name, value, step="frame"):
        pass

    def add_scalar(self, name, value, step="frame"):
        pass

    def add_schedule(self, name, value, step="frame"):
        pass

    def add_summary(self, name, mean, std, step="frame"):
        pass



def check_tuple(x):
    x_vec = None
    if isinstance(x, tuple):
        x, x_vec = x[0], x[1]
    return x, x_vec


def get_torch_cat(x, x_vec):
    if x_vec:
        x = torch.cat((x, x_vec), dim=1)
    return x


def weights_init_kaiming(m):
    # see also https://github.com/pytorch/pytorch/issues/18182
    for m in m.modules():
        if type(m) in {
            torch.nn.Conv2d,
            torch.nn.Conv3d,
            torch.nn.ConvTranspose2d,
            torch.nn.ConvTranspose3d,
            torch.nn.Linear,
        }:
            torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / math.sqrt(fan_out)
                torch.nn.init.normal_(m.bias, -bound, bound)


def weights_init_xavier(m):
    """
     Weights initialization using xavier uniform
    """
    # TODO: What if not ReLu?
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(m.bias, 0)
        # nn.init.xavier_uniform(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.constant_(m.bias, 0)
        # nn.init.xavier_normal(m.bias)

def weights_init_xavier_tanh(m):
    """
     Weights initialization using xavier uniform for tanh in linear case
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        nn.init.constant_(m.bias, 0)

"""
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':

"""


def freeze_weights(model):
    for i, chld in enumerate(model.children()):  # Freeze weights
        for params in chld.parameters():
            params.requires_grad = False


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, x.size(1))
