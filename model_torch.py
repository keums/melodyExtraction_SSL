import torch
import torch.nn as nn
import math


class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=(kernel_size-1) // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01)
        self.activation = nn.LeakyReLU(0.01)
    
    def forward(self, x, return_shortcut=False, skip_activation=False):
        shortcut = self.bn(self.conv(x))
        if return_shortcut:
            return self.activation(shortcut), shortcut
        elif skip_activation:
            return shortcut
        else:
            return self.activation(shortcut)
      

class ResNet_Block(nn.Module):
    def __init__(self, num_input_ch, num_channels):
        super(ResNet_Block, self).__init__()
        self.conv1 = ConvNorm(num_input_ch, num_channels, 1)
        self.conv2 = ConvNorm(num_channels, num_channels, 3)
        self.conv3 = ConvNorm(num_channels, num_channels, 3)
        self.conv4 = ConvNorm(num_channels, num_channels, 1)
        
    def cal_conv(self,x):
        return self.conv4(self.conv3(self.conv2(self.conv1(x))))

    def forward(self, x):
        x, shortcut = self.conv1(x, return_shortcut=True)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x, skip_activation=True)

        x += shortcut
        x = self.conv4.activation(x)
        x = torch.max_pool2d(x, (1,4))

        return x

class Melody_ResNet(nn.Module):
    def __init__(self):
        super(Melody_ResNet,self).__init__()
        self.block = nn.Sequential(
            ResNet_Block(1, 64),
            ResNet_Block(64, 128),
            ResNet_Block(128, 192),
            ResNet_Block(192, 256),
        )

        # Keras uses a hard_sigmoid for default activation, but the test showed that using a plain sigmoid in PyTorch 
        # showed the most similar result with the pre-trained Keras Model
        # Also, PyTorch LSTM does not provides recurernt_dropout.
        self.lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True,  dropout=0.3)
        num_output = int(55 * 2 ** (math.log(8, 2)) + 2)
        self.final = nn.Linear(512,num_output)

    def forward(self, input):
        block = self.block(input) # channel first for torch
        numOutput_P = block.shape[1] * block.shape[3]
        reshape_out = block.permute(0,2,3,1).reshape(block.shape[0], 31, numOutput_P)

        lstm_out, _ = self.lstm(reshape_out)
        out = self.final(lstm_out)
        out = torch.softmax(out, dim=-1)
        return out