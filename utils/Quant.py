import torch
import copy
from torch.quantization.quantize import add_observer_

_RELU_BRANCH = {'son':None,'can_be_fused':True}
_BN_BRANCH = {'son': {torch.nn.ReLU:_RELU_BRANCH},'can_be_fused':True}
_NN_BRANCH = {'son': {torch.nn.ReLU:_RELU_BRANCH},'can_be_fused':False}
_CONV_BRANCH = {'son': {torch.nn.BatchNorm2d:_BN_BRANCH,torch.nn.ReLU:_RELU_BRANCH},'can_be_fused':False}
_FUSETREE = {'son':{torch.nn.Conv2d:_CONV_BRANCH,torch.nn.Linear:_NN_BRANCH},'can_be_fused':False}
# FuseTree = {torch.nn.Conv2d:{torch.nn.ReLU:None,torch.nn.BatchNorm2d:{torch.nn.ReLU:None}},torch.nn.Linear:{torch.nn.ReLU:None}}

def fuse_module(module, inplace = False):
    if not inplace:
        module = copy.deepcopy(module)
    _fuse_module_helper(module)
    return module

def _fuse_module_helper(module):
    names = []
    tmpTree = _FUSETREE
    for name,child in module.named_children():
        if type(child) in tmpTree['son']:
            tmpTree = tmpTree['son'][type(child)]
            names.append(name)
        else:
            _fuse_module_helper(child)
            if tmpTree['can_be_fused']:
                torch.quantization.fuse_modules(module,names,inplace=True)
                names = []
                tmpTree = _FUSETREE
    if tmpTree['can_be_fused']:
        torch.quantization.fuse_modules(module,names,inplace=True)

# QCONFIGS = {} #use class method
# def propagate_qconfig(module,qconfig=None,inplace=False):
#     if not inplace:
#         module = copy.deepcopy(module)
#     module.qconfig = QCONFIGS[getattr(module,'qconfig',qconfig)]
#     if module.config is None:
#         raise Exception('not qconfig passed in or set in module')
#     for name, child in module.named_children():
#         propagate_qconfig(child,qconfig)
#
# def prepare(model,inplace=False):
#     assert hasattr(model,'qconfig')
#     propagate_qconfig(model,qconfig=model.qconfig,inplace=inplace)
#     add_observer_(model)
#     return model
