import torch
import torch.nn as nn
from utils import iou,squared_err

#y = [[0->19] prob classes, [prob obj , x,y,w,h],[[prob obj , x,y,w,h]]

class loss(nn.Module):
    def __init__(self,grids=7,boxes=2,classes=3,lambdas : tuple = (0.5,5)):
        super(loss,self).__init__()
        self.G = grids
        self.B = boxes
        self.C = classes

        # lambdas
        assert len(lambdas) == 2 , "lambdas should be of length 2"
        self.noobj_pen = lambdas[0]
        self.coord_pen = lambdas[1]
       
    def forward(self, prediction,target):
        
        iou1 = iou(prediction[...,self.C+1:self.C+5],target[...,self.C+1:self.C+5])
        iou2 = iou(prediction[...,self.C+6:self.C+10],target[...,self.C+1:self.C+5])
        
        ious = torch.cat([iou1.unsqueeze(0),iou2.unsqueeze(0)],dim=0) # squeeze to add the dimension you will concatenate on

        # to get the boxes that overlap the most with true box for each grid and for each example
        max_iou,box_idx = torch.max(ious,dim=0)
        
        #_:_ to keep it 4 dimensional
        obj_exists = target[...,self.C:self.C+1] 
        
        # Center Loss
        boxes = obj_exists*(
            (1-box_idx)*prediction[...,self.C+1:self.C+5] + 
            (box_idx)*prediction[...,self.C+6:self.C+10]
        )
        
        target_boxes = obj_exists * target[...,self.C+1:self.C+5]
        # take the square root for the  ws and hs 
        boxes[...,2:4] = torch.sign(boxes[...,2:4]) * torch.sqrt(torch.abs(boxes[...,2:4] + 1e-06)) 
        target_boxes[...,2:4] = torch.sign(target_boxes[...,2:4]) * torch.sqrt(torch.abs(target_boxes[...,2:4] + 1e-06))
        box_loss = self.coord_pen * squared_err(boxes , target_boxes)

        
        #Object Loss
        # penalty when pred conf is low and there exists an object
        pred_box = obj_exists*(
            (1-box_idx)*prediction[...,self.C:self.C+1] + 
            (box_idx)*prediction[...,self.C+5:self.C+6])
        
        target_obj = obj_exists*target[...,self.C:self.C+1]
        obj_loss = squared_err(pred_box,target_obj)
        
        
        #No Object Loss
        # penalty when pred conf is high and there is no object
        # can vary the penalty strength with noobj_pen
        nonpred_box1 = (1-obj_exists)*(prediction[...,self.C:self.C+1]) 
        nonpred_box2 = (1-obj_exists)*(prediction[...,self.C+5:self.C+6])
        noobj_loss = self.noobj_pen*(
            squared_err(nonpred_box1,(1-obj_exists)*obj_exists) + 
            squared_err(nonpred_box2,(1-obj_exists)*obj_exists))
        

        
        #Class Probs Loss
        # penalty when class conf is high and it is predicted low and vice verse
        class_loss = squared_err(obj_exists*(prediction[...,0:self.C]),obj_exists*(target[...,0:self.C]))
        
        total_loss = box_loss + obj_loss + noobj_loss + class_loss
        return total_loss
