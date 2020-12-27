import torch

_RELU_BRANCH = {'son':None,'can_be_fused':True}
_BN_BRANCH = {'son': {torch.nn.ReLU:_RELU_BRANCH},'can_be_fused':True}
_NN_BRANCH = {'son': {torch.nn.ReLU:_RELU_BRANCH},'can_be_fused':False}
_CONV_BRANCH = {'son': {torch.nn.BatchNorm2d:_BN_BRANCH,torch.nn.ReLU:_RELU_BRANCH},'can_be_fused':False}
_FUSETREE = {'son':{torch.nn.Conv2d:_CONV_BRANCH,torch.nn.Linear:_NN_BRANCH},'can_be_fused':False}
# FuseTree = {torch.nn.Conv2d:{torch.nn.ReLU:None,torch.nn.BatchNorm2d:{torch.nn.ReLU:None}},torch.nn.Linear:{torch.nn.ReLU:None}}

def fuse_module(module):
    names = []
    tmpTree = _FUSETREE
    for name,child in module.named_children():
        if type(child) in tmpTree['son']:
            tmpTree = tmpTree['son'][type(child)]
            names.append(name)
        else:
            fuse_module(child)
            if tmpTree['can_be_fused']:
                torch.quantization.fuse_modules(module,names,inplace=True)
                names = []
                tmpTree = _FUSETREE
    if tmpTree['can_be_fused']:
        torch.quantization.fuse_modules(module,names,inplace=True)