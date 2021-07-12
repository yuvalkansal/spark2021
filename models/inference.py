import torch
import torch.nn as nn
import torchvision
import os
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from quantization import *
from quan_resnet_imagenet import *

model = resnet18_quan(pretrained = True)

for m in model.named_modules():
    if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
        m.__reset_weight__()

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

test_dir = os.path.join('/home/candle/datasets/ILSVRC2012','val')
test_data = dset.ImageFolder(test_dir, transform=test_transform)

test_batch_size = 256
workers = 0
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=test_batch_size,
                                          shuffle=True,
                                          num_workers=workers,
                                          pin_memory=False)

criterion = torch.nn.CrossEntropyLoss()

for i, (input, target) in enumerate(test_loader):
    output = model(input)
    loss = criterion(output, target)
    print(loss)


