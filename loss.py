import torch
import torch.nn as nn
from utils import iou,squared_err

#y = [[0->19] prob classes, [prob obj , x,y,w,h],[[prob obj , x,y,w,h]]

class loss(nn.Module):
    def __init__(self,grids=7,boxes=2,classes=20):
        super(loss,self).__init__()
        self.G= grids
        self.B= boxes
        self.C= classes
        self.noobj_pen = 0.5
        self.coord_pen = 5
       
    def forward(self, prediction,target):
        
        #prediction = prediction.reshape(-1,self.G,self.G,self.C+(5*self.B))
        
        iou1 = iou(prediction[...,21:25],target[...,21:25])
        iou2 = iou(prediction[...,26:30],target[...,21:25])
        
        ious = torch.cat([iou1.unsqueeze(0),iou2.unsqueeze(0)],dim=0) # squeeze to add the dimension you will concatenate on
        
        max_iou,box_idx = torch.max(ious,dim=0)
        
        
        obj_exists = target[...,20:21] #20:21 to keep it 4 dimensional
        
        #Center Loss
        # if obj_exists then calc else put = 0
        # box_idx either 0 or 1 and if 0 calc 1*box1 + 0*box2 if 1 calc 0*box1+ 1*box2
        boxes = obj_exists*(
            (1-box_idx)*prediction[...,21:25] + 
            (box_idx)*prediction[...,26:30]
        )
        
        target_boxes = obj_exists*target[...,21:25]
        boxes[...,2:4] = torch.sign(boxes[...,2:4]) * torch.sqrt(torch.abs(boxes[...,2:4] +1e-6))
        target_boxes[...,2:4] = torch.sign(target_boxes[...,2:4]) * torch.sqrt(torch.abs(target_boxes[...,2:4]))
        box_loss = self.coord_pen*squared_err(boxes , target_boxes)

        
        #Object Loss
        pred_box = obj_exists*(
            (1-box_idx)*prediction[...,20:21] + 
            (box_idx)*prediction[...,25:26])
        
        target_obj = obj_exists*target[...,20:21]
        obj_loss = squared_err(pred_box,target_obj)
        
        
        #No Object Loss
        #(1-obj_exists)*obj_exists will always be zero then if the object exists pred will be zero due to (1-obj_exists)*(prediction[...,20:21])
        #if object exists and pred is not zero here is the penalty
        nonpred_box1 = (1-obj_exists)*(prediction[...,20:21]) 
        nonpred_box2 = (1-obj_exists)*(prediction[...,25:26])
        noobj_loss = self.noobj_pen*(
            squared_err(nonpred_box1,(1-obj_exists)*obj_exists) + 
            squared_err(nonpred_box2,(1-obj_exists)*obj_exists))
        

        
        #Class Probs Loss
        class_loss = squared_err(obj_exists*(prediction[...,0:20]),obj_exists*(target[...,0:20]))
        
        total_loss = box_loss + obj_loss + noobj_loss + class_loss
        return total_loss
#done