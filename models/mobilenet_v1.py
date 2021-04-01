import time
import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )


def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class MobileNetV1_Small(nn.Module):
    def __init__(self,num_classes=1000):
        super(MobileNetV1_Small, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 16, 2,leaky=0.1),   # 7
            conv_dw(16, 32, 1),  # 11
            conv_dw(32, 64, 2),  # 19
            conv_dw(64, 64, 1),  # 27
            conv_dw(64, 128, 2),  # 43
            conv_dw(128, 128,1),  # 43 + 16 = 59
        )
        self.stage2 = nn.Sequential(
            conv_dw(128, 256, 2), # 59 + 32 = 91
            conv_dw(256, 256, 1), # 91 + 32 = 123
            conv_dw(256, 256, 1), # 123 + 32 = 155
            conv_dw(256, 256, 1), # 155 + 32 = 187
            conv_dw(256, 256, 1), # 187 + 32 = 219
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x


class MobileNetV1_Small_Small(nn.Module):
    def __init__(self,num_classes=1000):
        super(MobileNetV1_Small_Small, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
            conv_dw(64, 64, 1), # 91 + 32 = 123
            conv_dw(64, 64, 1), # 123 + 32 = 155
            conv_dw(64, 64, 1), # 155 + 32 = 187
            conv_dw(64, 64, 1), # 187 + 32 = 219
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x



class MobileNetV1_Big(nn.Module):
    '''Official release　of  MobileNet_V1
    '''
    def __init__(self,num_classes=1000):
        super(MobileNetv1_Big, self).__init__()
 
        self.model = nn.Sequential(
            conv_bn(  3,  32, 2,leaky=0.1), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            #conv_dw(512, 1024, 2),
            #conv_dw(1024, 1024, 1),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Linear(512, num_classes)
 
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
 


def mobilenetv1(num_classes =1000, **kwargs):
    #model = MobileNetV1_Small(num_classes, **kwargs)
    model =MobileNetV1_Small_Small(num_classes, **kwargs)
    '''
    if not use_se:
        model_dict = model.state_dict()
        pretrained_dict = torch.load("/home/hzq/face_hzq/face_recognition_prj/arcface-pytorch/checkpoints/resnet_face18_20.pth")
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("fineturn model loads successfully!")
    '''
    return model
