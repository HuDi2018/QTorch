import os
import torch.nn as nn
import torch.quantization
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from utils import Quant
from example_util import prepare_data_loaders,print_size_of_model,evaluate,train_one_epoch,MobileNetV2

# # Setup warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.quantization'
)

# Specify random seed for repeatable results
torch.manual_seed(191009)

# params
TRAIN_FLOAT_MODEL = False
EVAL_FLOAT_MODEL = False
data_path = '../data/imagenet_1k'
saved_model_dir = '../weights/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'
train_batch_size = 30
eval_batch_size = 30
num_calibration_batches = 10

# data
data_loader, data_loader_test = prepare_data_loaders(data_path,train_batch_size,eval_batch_size)

# model and loss
float_model = MobileNetV2()
criterion = nn.CrossEntropyLoss()

# train float model
if TRAIN_FLOAT_MODEL:
    optimizer = optim.SGD(float_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    for nepoch in range(8):
        train_one_epoch(float_model, criterion, optimizer, data_loader, torch.device('cuda'), train_batch_size)
        break
    torch.save(float_model.state_dict(),os.path.join(saved_model_dir,float_model_file))
else:
    float_model.load_state_dict(torch.load(os.path.join(saved_model_dir,float_model_file)))

print('\n Inverted Residual Block: Before fusion \n\n', float_model.features[1].conv)

# Note fusion of Conv+BN+Relu and Conv+Relu
print('\n Inverted Residual Block: After fusion\n\n',float_model.features[1].conv)

print("Size of baseline model")
print_size_of_model(float_model)

if EVAL_FLOAT_MODEL:
    float_model.eval()
    top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=eval_batch_size)
    print('\n Evaluation accuracy on %d images, %2.2f'%(eval_batch_size * eval_batch_size, top1.avg))
#torch.jit.save(torch.jit.script(float_model), saved_model_dir + scripted_float_model_file)

# post training
float_model.eval()
Quant.fuse_module(float_model) # Fuse Conv, bn and relu
float_model.qconfig = torch.quantization.default_qconfig # Specify quantization configuration
print(float_model.qconfig)
torch.quantization.prepare(float_model, inplace=True)

# Calibrate first
print('\n Post Training Quantization Prepare: Inserting Observers')
print('\n Inverted Residual Block:After observer insertion \n\n', float_model.features[1].conv)

# Calibrate with the training set
evaluate(float_model, criterion, data_loader, neval_batches=num_calibration_batches)
print('\n Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(float_model, inplace=True)
print('Post Training Quantization: Convert done')
print('\n Inverted Residual Block: After fusion and quantization, note fused modules: \n\n',float_model.features[1].conv)

print("Size of model after quantization")
print_size_of_model(float_model)

top1, top5 = evaluate(float_model, criterion, data_loader_test, neval_batches=eval_batch_size)
print('\n Evaluation accuracy on %d images, %2.2f'%(eval_batch_size * eval_batch_size, top1.avg))

