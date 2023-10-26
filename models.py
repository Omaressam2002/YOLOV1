import torch
import torch.nn as nn
from utils import non_max_suppression,parse_predictions
from operators import ConvBlock,GlobalAvgPool2d


class darkNet(nn.Module): #Architectures
    def __init__(self,classes=1000,init_weight=False):
        super(darkNet, self).__init__()
        self.darkNet = nn.Sequential(
             ConvBlock(7,3,64,2,3),
             nn.MaxPool2d(kernel_size=2,stride=2),
             ConvBlock(3,64,192,padding=1),
             nn.MaxPool2d(kernel_size=2,stride=2),
             ConvBlock(1,192,128),
             ConvBlock(3,128,256,padding=1),
             ConvBlock(1,256,256),
             ConvBlock(3,256,512,padding=1),
             nn.MaxPool2d(kernel_size=2,stride=2),
             
            ConvBlock(1,512,256),
            ConvBlock(3,256,512,padding=1),
            ConvBlock(1,512,256),
            ConvBlock(3,256,512,padding=1),
            ConvBlock(1,512,256),
            ConvBlock(3,256,512,padding=1),
            ConvBlock(1,512,256),
            ConvBlock(3,256,512,padding=1),

            ConvBlock(1,512,512),
            ConvBlock(3,512,1024,padding=1),
            nn.MaxPool2d(kernel_size=2,stride=2),

            ConvBlock(1,1024,512),
            ConvBlock(3,512,1024,padding=1),
            ConvBlock(1,1024,512),
            ConvBlock(3,512,1024,padding=1)
        )
        self.classifier = nn.Sequential(
            *self.darkNet,
            GlobalAvgPool2d(),
            nn.Linear(1024, classes)
        )
        
        
        
        if init_weight:
            self._initialize_weights()

    def forward(self, x):
        return self.classifier(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ConvBlock):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.batchNorm.weight, 1)
                nn.init.constant_(m.batchNorm.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class YOLOv1(nn.Module): # Architectures
    def __init__(self,grids=7,boxes=2,classes=3,pretrained_backbone = False,init_weight = False):
        super(YOLOv1,self).__init__()
        self.G = grids
        self.B = boxes
        self.C = classes
        self.DN = darkNet().darkNet
        if pretrained_backbone:
            checkpoint = torch.load('DN_best.pth')
            DN = darkNet()
            DN.load_state_dict(checkpoint["state_dict"])
            self.DN = DN.darkNet 
        self.fullyConnected = nn.Sequential(
            ConvBlock(3,1024,1024,padding=1),
            ConvBlock(3,1024,1024,stride=2,padding=1),
            
            ConvBlock(3,1024,1024,padding=1),
            ConvBlock(3,1024,1024,padding=1),

            nn.Flatten(),
            nn.Linear(grids*grids*1024, 1024),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
            nn.Linear(1024, grids*grids*(classes+ boxes*5))
        )
        if init_weight:
            self._initialize_weights()
    def _initialize_weights(self):
        for m in self.fullyConnected.modules():
            if isinstance(m, ConvBlock):
                nn.init.kaiming_normal_(m.conv.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.batchNorm.weight, 1)
                nn.init.constant_(m.batchNorm.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def forward(self,x):
        x = self.DN(x)
        out = self.fullyConnected(x)
        out = out.reshape(-1,self.G,self.G,self.C+(5*self.B))
        return out


    def predict(self,x):
        # to make sure x is one pic not a batch
        assert len(x.shape) == 3 
        with torch.no_grad():
            pred = self.forward(x)
            #for each grid [0->3 for class prob, obj, x,y,w,h, obj, x,y,w,h]
        
        pred = parse_predictions(pred)# shape 49,6
        # for each grid [class_idx,best_conf,x,y,w,h] dims for the box of the best confidence
        pred = non_max_suppression(pred,iou_threshold=0.2,prob_threshold= 0.3)
        # pred is final output boxes
        return pred