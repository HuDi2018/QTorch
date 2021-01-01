from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn import Module
from abc import ABCMeta, abstractmethod
from functools import partial
from torch.quantization import QConfig

def _with_args(cls_or_self, **kwargs):
    r"""Wrapper that allows creation of class factories.

    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.

    Example::

        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r

class baseFQ(Module,metaclass=ABCMeta):
    with_args = classmethod(_with_args)
    def __init__(self):
        self.params = {}

    def register_soft_buffer(self,name_str,tensor_soft_buffer):
        self.params[name_str] = tensor_soft_buffer
        self.__setattr__(name_str,tensor_soft_buffer)

    @abstractmethod
    def calculate_qparams(self):
        pass

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super(example_FQ, self)._save_to_state_dict(destination, prefix, keep_vars)
        for name_str in self.params:
            destination[prefix + name_str] = self.__getattr__(name_str)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        for name_str in self.params:
            tensor_soft_buffer = state_dict.pop(prefix + name_str)
            self.__setattr__(name_str,tensor_soft_buffer)
            self.params[name_str] = tensor_soft_buffer
        super(example_FQ, self)._load_from_state_dict(state_dict, prefix, local_metadata, False,
                                                        missing_keys, unexpected_keys, error_msgs)

class example_FQ(baseFQ):
    with_args = classmethod(_with_args)
    def __init__(self, quant_min=0, quant_max=255,ch_axis=0,averaging_constant=0.01):
        super(example_FQ, self).__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.register_params('float_min',torch.tensor([-1.0]))
        self.register_params('float_max', torch.tensor([1.0]))
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.ch_axis = ch_axis
        self.averaging_constant= averaging_constant

    def calculate_qparams(self):
        scale = (self.float_max - self.float_min) / (self.quant_max-self.quant_min)
        zp = self.quant_min
        return scale,zp

    def forward(self, X):
        # observe feature and calculate scale and zero point
        detached_X = X.detach()
        detached_X = detached_X.to(self.float_min.dtype)
        float_min = self.float_min + self.averaging_constant * (torch.min(detached_X) - self.float_min)
        float_max = self.float_max + self.averaging_constant * (torch.max(detached_X) - self.float_max)
        #self.float_min.resize_(float_min.shape)
        #self.float_max.resize_(float_max.shape)
        self.float_min.copy_(float_min)
        self.float_max.copy_(float_max)
        scale,zp = self.calculate_qparams()

        # fake quantize feature
        X = torch.fake_quantize_per_channel_affine(X, scale, zp,
                                                           self.ch_axis, self.quant_min, self.quant_max)
        return X


Example_Qconifg =QConfig(weight = example_FQ.with_args(quant_min=-128,quant_max=127),activation=example_FQ.with_args(quant_min=0,quant_max=255))