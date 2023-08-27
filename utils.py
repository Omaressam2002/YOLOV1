import torch
import torch.nn as nn




class ConvBlock(nn.Module):
    def __init__(self,size,ch_in,ch_out,stride=1,padding='same'):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(ch_in,ch_out,size,stride,padding,bias=False) #batch norm so bias = false
        self.batchNorm = nn.BatchNorm2d(ch_out)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self,x):
        return self.lrelu(self.batchNorm(self.conv(x)))





class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(x):
        return torch.mean(torch.flatten(x, start_dim=2), dim=2)



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes





def iou(box1, box2): 
    box1_x = box1[...,0:1]
    box1_y = box1[...,1:2]
    box1_w = box1[...,2:3]
    box1_h = box1[...,3:4]
    box2_x = box2[...,0:1]
    box2_y = box2[...,1:2]
    box2_w = box2[...,2:3]
    box2_h = box2[...,3:4]

    box1_x1 = torch.sub(box1_x,box1_w,alpha=0.5) #box1_x-(box1_w/2)
    box1_y1 = torch.sub(box1_y,box1_h,alpha=0.5) #box1_y-(box1_h/2)
    box1_x2 = torch.add(box1_x,box1_w,alpha=0.5) #box1_x+(box1_w/2)
    box1_y2 = torch.add(box1_y,box1_h,alpha=0.5) #box1_y+(box1_h/2)
    box2_x1 = torch.sub(box2_x,box2_w,alpha=0.5) #box2_x-(box2_w/2)
    box2_y1 = torch.sub(box2_y,box2_h,alpha=0.5) #box2_y-(box2_h/2)
    box2_x2 = torch.add(box2_x,box2_w,alpha=0.5) #box2_x+(box2_w/2)
    box2_y2 = torch.add(box2_y,box2_h,alpha=0.5) #box2_y+(box2_h/2)

    
    xi1 = torch.max(box1_x1,box2_x1)
    yi1 = torch.max(box1_y1,box2_y1)
    xi2 = torch.min(box1_x2,box2_x2)
    yi2 = torch.min(box1_y2,box2_y2)


    
    inter_width = torch.clamp(yi2-yi1 , min=0) 
    inter_height = torch.clamp(xi2-xi1 , min=0)
    inter_area = inter_width*inter_height

    box1_area = torch.mul(torch.sub(box1_x2,box1_x1),torch.sub(box1_y2,box1_y1))
    box2_area = torch.mul(torch.sub(box2_x2,box2_x1),torch.sub(box2_y2,box2_y1))
    union_area = box1_area + box2_area - inter_area

    iou = inter_area/union_area
    return iou



def squared_err(x1,x2): #done
    sum = torch.sum((x1-x2)**2)
    return sum