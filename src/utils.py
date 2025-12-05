import math
from torch import Tensor


def glorot(tensor):
    if tensor != None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor != None:
        tensor.data.fill_(0)