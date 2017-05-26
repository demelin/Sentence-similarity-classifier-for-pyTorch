""" Borrowed from https://github.com/alykhantejani/nninit/blob/master/nninit.py """

from torch.autograd import Variable
import numpy as np


def _calculate_fan_in_and_fan_out(tensor):
    if tensor.ndimension() < 2:
        raise ValueError("fan in and fan out can not be computed for tensor of size ", tensor.size())

    if tensor.ndimension() == 2:  # Linear
        fan_in = tensor.size(1)
        fan_out = tensor.size(0)
    else:
        num_input_fmaps = tensor.size(1)
        num_output_fmaps = tensor.size(0)
        receptive_field_size = np.prod(tensor.numpy().shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def xavier_uniform(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a uniform distribution.
       The resulting tensor will have values sampled from U(-a, a) where a = gain * sqrt(2/(fan_in + fan_out))
    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied
    Examples:
        w = torch.Tensor(3, 5)
        xavier_uniform(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        xavier_uniform(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        a = np.sqrt(3.0) * std
        return tensor.uniform_(-a, a)


def xavier_normal(tensor, gain=1):
    """Fills the input Tensor or Variable with values according to the method described in "Understanding the difficulty of training
       deep feedforward neural networks" - Glorot, X. and Bengio, Y., using a normal distribution.
       The resulting tensor will have values sampled from normal distribution with mean=0 and
       std = gain * sqrt(2/(fan_in + fan_out))
    Args:
        tensor: a n-dimension torch.Tensor
        gain: an optional scaling factor to be applied
    Examples:
        w = torch.Tensor(3, 5)
        xavier_normal(w, gain=np.sqrt(2.0))
    """
    if isinstance(tensor, Variable):
        xavier_normal(tensor.data, gain=gain)
        return tensor
    else:
        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return tensor.normal_(0, std)
