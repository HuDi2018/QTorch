import os
import torch.nn as nn
import torch.quantization
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from example.example_util import prepare_data_loaders,print_size_of_model,evaluate,train_one_epoch,MobileNetV2
from FakeQuantizer.base import Example_Qconifg
from utils.Quant import propagate_qconfig,prepare

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
data_path = './data/imagenet_1k'
saved_model_dir = './weights/'
float_model_file = 'mobilenet_pretrained_float.pth'
scripted_float_model_file = 'mobilenet_quantization_scripted.pth'
scripted_quantized_model_file = 'mobilenet_quantization_scripted_quantized.pth'

train_batch_size = 30
eval_batch_size = 30

# data
data_loader, data_loader_test = prepare_data_loaders(data_path,train_batch_size,eval_batch_size)

# mdoel setting
float_model = MobileNetV2().cuda()
weights = torch.load(os.path.join(saved_model_dir,float_model_file))
float_model.load_state_dict(weights)
float_model.fuse_model()
propagate_qconfig(float_model,Example_Qconifg,inplace=True)
fake_quantized_model = prepare(float_model)
# float_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
# fake_quantized_model = torch.quantization.prepare_qat(float_model, inplace=False)
print('Inverted Residual Block: After preparation for QAT, note fake-quantization modules \n',fake_quantized_model.features[1].conv)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(float_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.1)
exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

num_train_batches = 1

# Train and check accuracy after each epoch
for nepoch in range(8):
    train_one_epoch(fake_quantized_model, criterion, optimizer, data_loader, torch.device('cuda'), num_train_batches)
    if nepoch > 3:
        # Freeze quantizer parameters
        fake_quantized_model.apply(torch.quantization.disable_observer)
    if nepoch > 2:
        # Freeze batch norm mean and variance estimates
        fake_quantized_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)

    # Check the accuracy after each epoch
    quantized_model = torch.quantization.convert(fake_quantized_model.cpu().eval(), inplace=False)
    quantized_model.eval()
    top1, top5 = evaluate(quantized_model,criterion, data_loader_test)
    print('Epoch %d :Evaluation accuracy on eval images, %2.2f'%(nepoch,  top1.avg))
    break

# two ways of saving model
torch.jit.save(torch.jit.script(quantized_model), saved_model_dir + scripted_quantized_model_file)
torch.jit.save(torch.jit.trace(quantized_model,(torch.randn((1,3,224,224)))), saved_model_dir + scripted_quantized_model_file)