import torch
from tqdm import tqdm
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# to change number of classes:
# change in loss and Yolo class
# change in parsepred,parselabels,meanavgprecision

def train_fn(train_loader, model, optimizer, loss_fn,loss_info : list ,cost_info : list, device='cpu'): #train
    loop = tqdm(train_loader, leave=True)
    total_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        model = model.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        total_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item()/8)
    loss_info.append(total_loss)
    cost = sum(total_loss)/len(total_loss)
    print(f"Mean loss was {cost}")
    cost_info.append(cost)


def get_bboxes(loader,model,iou_threshold,threshold,device="cpu",verbose = False,pics=0): #Utils
    all_pred_boxes = []
    all_true_boxes = []
    index = 0
    
    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        predictions = parse_predictions(predictions)# shape batch_size,(g*g),6
        labels = parse_labels(labels)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                predictions[idx],
                iou_threshold=iou_threshold,
                prob_threshold=threshold
            )


            if idx <= pics and batch_idx == 0 and verbose :
                print("Truth")
                plot_image(x[idx].permute(1,2,0).to("cpu"), labels[idx])
                print("Prediction")
                plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)

            #recheck this format thats the only thing missing her
            for nms_box in nms_boxes:
                all_pred_boxes.append([index] + nms_box)


            for box in labels[idx]:
                all_true_boxes.append([index] + box )
                                      
            index += 1



    return all_pred_boxes, all_true_boxes



def parse_predictions(predictions): #Utils
    #[0,1,0,0,0,..,obj,x_cell,y_cell,w_cell,h_cell] -> [class label,obj_conf,x_global,y_global,w_global,h_global]
    #for each grid
    # repeat for each example in batch
    c = 3
    g = 7
    #g=13
    predictions = predictions.to("cpu")
    batch = predictions.shape[0]
    
    bboxes1 = predictions[..., c+1:c+5]
    bboxes2 = predictions[..., c+6:c+10]
    #confidence in each anchor box
    scores = torch.cat(
        (predictions[..., c].unsqueeze(0), predictions[..., c+5].unsqueeze(0)), dim=0
    )

    best_confidence,best_box= torch.max(scores,dim=0)

    best_box = best_box.unsqueeze(-1)
    bboxes = (1-best_box) * bboxes1 + best_box * bboxes2

    #globalizing the coordinates
    #
    cell_no = torch.arange(g).repeat(batch,g,1).unsqueeze(-1)
    x = 1/g * (bboxes[...,0:1]+cell_no)
    y = 1/g * (bboxes[...,1:2]+cell_no.permute(0,2,1,3))
    w_h = 1/g * bboxes[...,2:4]

    bboxes = torch.cat((x,y,w_h),dim=-1)

    class_labels = predictions[...,0:c].argmax(-1).unsqueeze(-1)
    best_confidence = best_confidence.unsqueeze(-1)

    preds_parsed = torch.cat((class_labels,best_confidence,bboxes),dim=-1).reshape(batch,g*g,6) 

    converted_predictions = []
    for i in range(batch):
        picture =[]
        for j in range(g*g):
            picture.append([x.item() for x in preds_parsed[i,j,:]])
        converted_predictions.append(picture)
    return converted_predictions 


def parse_labels(labels):
    # in tensor ex,7,7,8 -> out list numboxes,[]
    c = 3
    parsed_labels =[]
    for label in labels:
        l = []
        for row in label:
            for box in row:
                # if object exist
                if box[c] == 1 :
                    class_idx = torch.argmax(box[:c]).item()
                    l.append([class_idx, 1,box[c+1].item() ,box[c+2].item(),box[c+3].item(),box[c+4].item()])
        parsed_labels.append(l)

    return parsed_labels


def non_max_suppression(prediction,iou_threshold,prob_threshold): #Utils
    nms_boxes = []
    prediction = [box for box in prediction if box[1] > prob_threshold]
    prediction = sorted(prediction, key=lambda x: x[1], reverse=True) # sort based on prob threshold

    while prediction:
        box = prediction.pop(0)
        # get most confident box and remove other boxes that over lap with it
    

        prediction = [b for b in prediction if (b[0] != box[0] or iou(torch.Tensor(b[2:]),torch.Tensor(box[2:])) < iou_threshold)]

        
        nms_boxes.append(box)

    return nms_boxes  

def plot_image(image, boxes):
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    # add confidence rate
    ax.imshow(im)
    colors =['r','g','b']
    for box in boxes:
        
        c = colors[int(box[0])%3]
        conf = box[1]
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=3,
            edgecolor=c,
            facecolor="none",
        )
        ax.text(upper_left_x*width,(upper_left_y*height) - 3,str(conf)[:4],color="white",fontsize=6,fontweight='bold')
        ax.add_patch(rect)

    plt.show()


def save_image(image, boxes, src_path):
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    colors =['r','g','b']
    for box in boxes:
        
        c = colors[int(box[0])%3]
        conf = box[1]
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=3,
            edgecolor=c,
            facecolor="none",
        )
        ax.text(upper_left_x*width,(upper_left_y*height) - 3,str(conf)[:4],color="white",fontsize=6,fontweight='bold')
        ax.add_patch(rect)

    dest_path = src_path.split('.')
    dest_path = dest_path[0]+'_prediction'+dest_path[1]
    plt.savefig(dest_path)
    return dest_path



def mean_average_precision(predictions,labels,box_format='midpoint',iou_threshold=0.5,num_classes=3): 
    avg_precision = []
    #[idx,class, conf, x,y,w,h]
    for c in range(num_classes):
        dets = []
        truths = []
        for pred in predictions:
            if pred[1] == c:
                dets.append(pred)
        for label in labels:
            if label[1] == c:
                truths.append(label)
                
        # now we got all preds and truths of class c
        # should decide for each pred whether it is a TP or a FP
        
        TP = torch.zeros(len(dets))
        FP = torch.zeros(len(dets))

        #sort detections by confidence
        dets.sort(key=lambda x: x[2], reverse=True)
        
        for i,pred in enumerate(dets):
            for l in truths:
                if l[0] == pred[0]:
                    if iou(torch.tensor(l[3:]),torch.tensor(pred[3:])) > iou_threshold :
                        #if already predicited
                        if len(l) > 7:
                            FP[i] = 1
                            break
                        else:
                            TP[i] = 1
                            l.append(1)
                            # to indicate already predicted
                            break
            # if we have been through all boxes and none had a good enough iou with the pred
            # then it is a FP
            if FP[i] == 0 and TP[i] == 0:
                FP[i] = 1

        #cumilative sum
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        
        recalls = TP_cumsum / (len(truths) + 1e-07)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-07))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        #integeraton
        avg_precision.append(torch.trapz(precisions, recalls))
            
                        
    return sum(avg_precision) / len(avg_precision)             



def iou(box1, box2,format="midpoint"):
    if format == "midpoint":
        box1_x = box1[...,0:1]
        box1_y = box1[...,1:2]
        box1_w = box1[...,2:3]
        box1_h = box1[...,3:4]
        box2_x = box2[...,0:1]
        box2_y = box2[...,1:2]
        box2_w = box2[...,2:3]
        box2_h = box2[...,3:4]

        box1 = midToCorners(torch.cat((box1_x,box1_y,box1_w,box1_h),dim=-1))
        box2 = midToCorners(torch.cat((box2_x,box2_y,box2_w,box2_h),dim=-1))

        box1_x1 = box1[...,0:1]
        box1_y1 = box1[...,1:2]
        box1_x2 = box1[...,2:3]
        box1_y2 = box1[...,3:4]
        box2_x1 = box2[...,0:1]
        box2_y1 = box2[...,1:2]
        box2_x2 = box2[...,2:3]
        box2_y2 = box2[...,3:4]

        # box1_x1 = torch.sub(box1_x,box1_w,alpha=0.5) #box1_x-(box1_w/2)
        # box1_y1 = torch.sub(box1_y,box1_h,alpha=0.5) #box1_y-(box1_h/2)
        # box1_x2 = torch.add(box1_x,box1_w,alpha=0.5) #box1_x+(box1_w/2)
        # box1_y2 = torch.add(box1_y,box1_h,alpha=0.5) #box1_y+(box1_h/2)
        # box2_x1 = torch.sub(box2_x,box2_w,alpha=0.5) #box2_x-(box2_w/2)
        # box2_y1 = torch.sub(box2_y,box2_h,alpha=0.5) #box2_y-(box2_h/2)
        # box2_x2 = torch.add(box2_x,box2_w,alpha=0.5) #box2_x+(box2_w/2)
        # box2_y2 = torch.add(box2_y,box2_h,alpha=0.5) #box2_y+(box2_h/2)

        
    
    elif format == "corners":
        box1_x1 = box1[...,0:1]
        box1_y1 = box1[...,1:2]
        box1_x2 = box1[...,2:3]
        box1_y2 = box1[...,3:4]
        box2_x1 = box2[...,0:1]
        box2_y1 = box2[...,1:2]
        box2_x2 = box2[...,2:3]
        box2_y2 = box2[...,3:4]
        

    
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



def midToCorners(box): 
    #[x,y,w,h]
    assert type(box) == torch.Tensor
    
    x1 = torch.sub(box[...,0:1],box[...,2:3],alpha=0.5) #x-(w/2)
    y1 = torch.sub(box[...,1:2],box[...,3:4],alpha=0.5) #y-(h/2)
    x2 = torch.add(box[...,0:1],box[...,2:3],alpha=0.5) #x+(w/2)
    y2 = torch.add(box[...,1:2],box[...,3:4],alpha=0.5) #y+(h/2)
    return torch.cat((x1,y1,x2,y2),dim=-1)


def cornersToMid(box):
    #[x1,y1,x2,y2]
    assert type(box) == torch.Tensor
    
    w = torch.sub(box[...,2:3],box[...,0:1])
    h = torch.sub(box[...,3:4],box[...,1:2])
    x = torch.add(box[...,0:1],w,alpha = 0.5)
    y = torch.add(box[...,1:2],h,alpha = 0.5)
    return torch.cat((x,y,w,h),dim=-1)



def squared_err(x1,x2): #done
    sum = torch.sum((x1-x2)**2)
    return sum