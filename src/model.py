# -*- encoding: utf-8 -*-
# @Time    :   2024/10/03 12:02:28
# @File    :   model.py
# @Author  :   ciaoyizhen
# @Contact :   yizhen.ciao@gmail.com
# @Function:   模型文件
import os
import torch
import numpy as np
from PIL import Image
from torch import nn
from transformers import ResNetForImageClassification, ResNetConfig
from transformers.modeling_outputs import ImageClassifierOutputWithNoAttention


class MyConv2d(nn.Module):
    """因为resnet模型
    在每一个卷积层后面要跟一个BN层，然后跟一个激活函数
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.ReLU()
        
    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))

class ShortCut(nn.Module):
    """残差连接在50层以上，维度不一致，需要将维度变回去，实际上resnet50不止50层，50层不包括shortcut连接的卷积层。
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        return self.bn(self.conv(x))

class ResNet(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        self._createNet(class_num)
        self.loss_function = nn.CrossEntropyLoss()
    
    def _createNet(self, class_num):
        # 以resnet50为例
        self.conv_1 = nn.Sequential(
            MyConv2d(3, 64, kernel_size=7, stride=2, padding=3),  # reduce 2
            nn.MaxPool2d(3, 2, 1),  # reduce 2
        )
        self.short_cut_2 = ShortCut(64, 256, 1, 1)
        
        self.conv_2_1 = nn.Sequential(
            MyConv2d(64, 64, 1, 1),
            MyConv2d(64, 64, 3, 1, 1),
            MyConv2d(64, 256, 1, 1)
        )

        self.conv_2_2 = nn.Sequential(
            MyConv2d(256, 64, 1, 1),
            MyConv2d(64, 64, 3, 1, 1),
            MyConv2d(64, 256, 1, 1)
        )
        
        self.conv_2_3 = nn.Sequential(
            MyConv2d(256, 64, 1, 1),
            MyConv2d(64, 64, 3, 1, 1),
            MyConv2d(64, 256, 1, 1)
        )
        self.short_cut_3 = ShortCut(256, 512, 1, 2)  # reduce 2
        
        self.conv_3_1 = nn.Sequential(
            MyConv2d(256, 128, 1, 2),
            MyConv2d(128, 128, 3, 1, 1),
            MyConv2d(128, 512, 1, 1)
        )

        self.conv_3_2 = nn.Sequential(
            MyConv2d(512, 128, 1, 1),
            MyConv2d(128, 128, 3, 1, 1),
            MyConv2d(128, 512, 1, 1)
        )

        self.conv_3_3 = nn.Sequential(
            MyConv2d(512, 128, 1, 1),
            MyConv2d(128, 128, 3, 1, 1),
            MyConv2d(128, 512, 1, 1)
        )

        self.conv_3_4 = nn.Sequential(
            MyConv2d(512, 128, 1, 1),
            MyConv2d(128, 128, 3, 1, 1),
            MyConv2d(128, 512, 1, 1)
        )
        self.short_cut_4 = ShortCut(512, 1024, 1, 2)  # reduce 2


        self.conv_4_1 = nn.Sequential(
            MyConv2d(512, 256, 1, 2),
            MyConv2d(256, 256, 3, 1, 1),
            MyConv2d(256, 1024, 1, 1)
        )
        self.conv_4_2 = nn.Sequential(
            MyConv2d(1024, 256, 1, 1),
            MyConv2d(256, 256, 3, 1, 1),
            MyConv2d(256, 1024, 1, 1)
        )
        self.conv_4_3 = nn.Sequential(
            MyConv2d(1024, 256, 1, 1),
            MyConv2d(256, 256, 3, 1, 1),
            MyConv2d(256, 1024, 1, 1)
        )
        self.conv_4_4 = nn.Sequential(
            MyConv2d(1024, 256, 1, 1),
            MyConv2d(256, 256, 3, 1, 1),
            MyConv2d(256, 1024, 1, 1)
        )
        self.conv_4_5 = nn.Sequential(
            MyConv2d(1024, 256, 1, 1),
            MyConv2d(256, 256, 3, 1, 1),
            MyConv2d(256, 1024, 1, 1)
        )
        self.conv_4_6 = nn.Sequential(
            MyConv2d(1024, 256, 1, 1),
            MyConv2d(256, 256, 3, 1, 1),
            MyConv2d(256, 1024, 1, 1)
        )
        self.short_cut_5 = ShortCut(1024, 2048, 1, 2)  # reduce 2

        self.conv_5_1 = nn.Sequential(
            MyConv2d(1024, 512, 1, 2),
            MyConv2d(512, 512, 3, 1, 1),
            MyConv2d(512, 2048, 1, 1)
        )
        self.conv_5_2 = nn.Sequential(
            MyConv2d(2048, 512, 1, 1),
            MyConv2d(512, 512, 3, 1, 1),
            MyConv2d(512, 2048, 1, 1)
        )
        self.conv_5_3 = nn.Sequential(
            MyConv2d(2048, 512, 1, 1),
            MyConv2d(512, 512, 3, 1, 1),
            MyConv2d(512, 2048, 1, 1)
        )
        
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, class_num)
        )

        
    def forward(self, pixel_values, labels=None):
        out_1 = self.conv_1(pixel_values)
        
        out_2_1 = self.conv_2_1(out_1) + self.short_cut_2(out_1)
        out_2_2 = self.conv_2_2(out_2_1) + out_2_1
        out_2_3 = self.conv_2_3(out_2_2) + out_2_2
        
        out_3_1 = self.conv_3_1(out_2_3) + self.short_cut_3(out_2_3)
        out_3_2 = self.conv_3_2(out_3_1) + out_3_1
        out_3_3 = self.conv_3_3(out_3_2) + out_3_2
        out_3_4 = self.conv_3_4(out_3_3) + out_3_3
        
        out_4_1 = self.conv_4_1(out_3_4) + self.short_cut_4(out_3_4)
        out_4_2 = self.conv_4_2(out_4_1) + out_4_1
        out_4_3 = self.conv_4_3(out_4_2) + out_4_2
        out_4_4 = self.conv_4_4(out_4_3) + out_4_3
        out_4_5 = self.conv_4_4(out_4_4) + out_4_4
        out_4_6 = self.conv_4_6(out_4_5) + out_4_5
        
        out_5_1 = self.conv_5_1(out_4_6) + self.short_cut_5(out_4_6)
        out_5_2 = self.conv_5_2(out_5_1) + out_5_1
        out_5_3 = self.conv_5_3(out_5_2) + out_5_2
        
        output = self.output(out_5_3)
        
        if labels is not None:
            loss = self.loss_function(output, labels)
            return ImageClassifierOutputWithNoAttention(
                loss=loss,
                logits=output,
                hidden_states=None
            )
        return ImageClassifierOutputWithNoAttention(
            loss=None,
            logits=output,
            hidden_states=None
        )


class Model():
    def __init__(self, cfg, id2label, label2id) -> None:
        """init, model in yaml

        Args:
            cfg (dict): model in yaml
        """
        self.cfg = cfg
        num_labels = len(id2label)
        self.model = ResNet(num_labels)
        # self.model = ResNetForImageClassification(ResNetConfig(num_labels=num_labels))
    
if __name__ == "__main__":
    import torch
    
    inputs = torch.randn((3, 224, 224))
    inputs = inputs.unsqueeze(0)
    model = ResNet(10)
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        print(output)