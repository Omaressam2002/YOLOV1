import torch
import torch.nn as nn
from utils import ConvBlock,GlobalAvgPool2d

class darkNet(nn.Module): #Architectures
    def __init__(self,classes=1000,init_weight=False):
        super(darkNet, self).__init__()
        self.darkNet = nn.Sequential(
             ConvBlock(7,3,64,2,3),
             nn.MaxPool2d(kernel_size=2,stride=2),
             ConvBlock(3,64,192),
             nn.MaxPool2d(kernel_size=2,stride=2),
             ConvBlock(1,192,128),
             ConvBlock(3,128,256),
             ConvBlock(1,256,256),
             ConvBlock(3,256,512),
             nn.MaxPool2d(kernel_size=2,stride=2),
             
            ConvBlock(1,512,256),
            ConvBlock(3,256,512),
            ConvBlock(1,512,256),
            ConvBlock(3,256,512),
            ConvBlock(1,512,256),
            ConvBlock(3,256,512),
            ConvBlock(1,512,256),
            ConvBlock(3,256,512),

            ConvBlock(1,512,512),
            ConvBlock(3,512,1024),
            nn.MaxPool2d(kernel_size=2,stride=2),

            ConvBlock(1,1024,512),
            ConvBlock(3,512,1024),
            ConvBlock(1,1024,512),
            ConvBlock(3,512,1024)
        )
        self.classifier = nn.Sequential(
            *self.darkNet,
            GlobalAvgPool2d(),
            nn.Linear(1024, classes)
        )
        
        
        
        #if init_weight:
        #    self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class YOLOv1(nn.Module): # Architectures
    def __init__(self,grids=7,boxes=2,classes=20,pretrained_backbone=False):
        super(YOLOv1,self).__init__()
        self.G = grids
        self.B = boxes
        self.C = classes
        self.DN = darkNet()
        if pretrained_backbone:
            self.DN = darkNet()
            src_state_dict = torch.load('DN_best.pth')['state_dict']
            dst_state_dict = self.DN.state_dict()
            print('Loading weights')
            for k in dst_state_dict.keys():
                dst_state_dict[k] = src_state_dict[k]
                
            self.DN.load_state_dict(dst_state_dict)
            #self.DN = darknet.modules.features
        self.fullyConnected = nn.Sequential(
            ConvBlock(3,1024,1024),
            ConvBlock(3,1024,1024,2,1),
            ConvBlock(3,1024,1024),
            ConvBlock(3,1024,1024),

            nn.Flatten(),
            nn.Linear(grids*grids*1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, grids*grids*(classes+ boxes*5))
        )
    def forward(self,x):
        x = self.DN.darkNet(x) # are you sure the weights are the trained one??
        out = self.fullyConnected(x)
        out = out.reshape(-1,self.G,self.G,self.C+(5*self.B))
        return out  