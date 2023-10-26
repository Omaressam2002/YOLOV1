import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import patches
from utils import plot_image
import torchvision.transforms as transforms


class ImageNet(torch.utils.data.Dataset): #Datasets
    def __init__(self,classes,train = True,j=0):

        self.classes = classes
        
        if train :
            csv = 'ImageNet/LOC_train_solution.csv'
            self.img_path = 'ImageNet/ILSVRC/Data/CLS-LOC/train'
            self.data = []
            self.labels = []
            for dir in list(os.listdir(self.img_path)):
                if dir[:1] != "n": #to not allow any file in directory thats not a class folder to be processed
                    continue
                class_path = os.path.join(self.img_path,dir)
                pics = list(os.listdir(class_path))
                for i in range(60): # 60 images from each class
                    self.labels.append(self.classes[dir])
                    self.data.append(os.path.join(class_path,pics[ (i+60*j)%len(pics) ])) 
            self.train = train
        else :
            csv = 'ImageNet/LOC_val_solution.csv' 
            self.img_path = 'ImageNet/ILSVRC/Data/CLS-LOC/val'
            self.data = (pd.read_csv(csv))[:5000]
            self.train = train

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        if self.train:
            class_idx = self.labels[index]
            img_i_path = self.data[index]
        else : 
            class_id = (self.data.iloc[index,1].split())[0]
            class_idx = self.classes[class_id]
            img_id = self.data.iloc[index,0]
            path = img_id+'.jpeg'
            img_i_path = os.path.join(self.img_path,path)
        
        img = Image.open(img_i_path)
        img = transforms.Resize((448, 448))(img)
        img = transforms.ToTensor()(img)
        if img.shape[0] != 3:
            img = Image.open(img_i_path).convert("RGB")
            img = transforms.Resize((448, 448))(img)
            img = transforms.ToTensor()(img)
        
       
        return img,class_idx





class V1Dataset(torch.utils.data.Dataset):
    def __init__(self,dataDir= 'v1',splitDir = 'train', transform = None, grids = 7, classes = 3, test = False):
        
        assert dataDir.lower() != 'v3' , "dont use this version right now"

        
        self.transform = transform
        self.G = grids
        self.C = classes
        self.test = test
        self.pics = [] # pic directories
        self.labels = [] # boxes
        self.baseDir = os.path.join(os.path.join("Data",dataDir),splitDir)

        
        picsAndLabels = [file for file in sorted(os.listdir(self.baseDir)) if (file not in [".DS_Store" ,"_darknet.labels","labels.txt"])]
        length = len(picsAndLabels)//2
        
        for i in range(length):
            # pic path
            picPath = picsAndLabels[2*i]
            # label path with the base Dir
            labelPath = os.path.join(self.baseDir,picsAndLabels[(2*i)+1])
            # reading labels
            with open(labelPath) as f:
                lines = f.readlines()
            # all boxes in pic
            boxes = []
            for l in lines:
                class_label,x,y,w,h = l.replace("\n","").split()

                # extract the labels from the data in the following format
                # {"ball":0 , "player":1 , "hoop":2}
                # disregarding any extra labels and switching label number to match the format above
                
                if dataDir.lower() in ['v1','v2']:
                    box = self.filterV1(l)
                elif dataDir.lower() == 'v4':
                    box = self.filterV2(l)
                elif dataDir.lower() == 'v5':
                    box = self.filterV3(l) 
                
                
                # if the box is of a disregarded label
                if len(box) != 0 :
                    boxes.append(box)


            
            # to prevent adding pics that have only disregarded classes
            if len(boxes) != 0 :
                self.pics.append(picPath)
                self.labels.append(boxes)

    # V1, V2 , V3
    def filterV1(self,line):
        # ball 0 -> 0
        # palyer 2 -> 1
        # hoop 3 -> 2
        
        class_label,x,y,w,h = line.replace("\n","").split()
        box = []
        # case ball
        if class_label ==  '0' :
            box = [0,1,float(x),float(y),float(w),float(h)]
        # case player
        elif class_label == '2' :
            box = [1,1,float(x),float(y),float(w),float(h)]
        # case hoop
        elif class_label == '3':
            box = [2,1,float(x),float(y),float(w),float(h)]
        
        return box

    # V4
    def filterV2(self,line):
        # ball 0 -> 0
        # palyer 2 -> 1
        # hoop 1 -> 2
        
        class_label,x,y,w,h = line.replace("\n","").split()
        box = []
        # case ball
        if class_label ==  '0' :
            box = [0,1,float(x),float(y),float(w),float(h)]
        # case player
        elif class_label == '2' :
            box = [1,1,float(x),float(y),float(w),float(h)]
        # case hoop
        elif class_label == '1':
            box = [2,1,float(x),float(y),float(w),float(h)]
        
        return box

    # V5
    def filterV3(self,line):
        # ball 0 -> 0
        # ball 1 -> 0
        # player 3 -> 1
        # refree 4 -> 1
        # hoop 2 -> 2
        
        class_label,x,y,w,h = line.replace("\n","").split()
        box = []
        # case ball 
        if class_label ==  '1' :
            box = [0,1,float(x),float(y),float(w),float(h)]
        # case player or refree
        elif class_label in ['3','4'] :
            box = [1,1,float(x),float(y),float(w),float(h)]
        # case hoop
        elif class_label == '2':
            box = [2,1,float(x),float(y),float(w),float(h)]
        
        return box

    


    def __len__(self):
        return len(self.pics) 
    
    def __getitem__(self,index):
        # concatenate pic path with base Dir
        picFullPath = os.path.join(self.baseDir,self.pics[index])

        
        # open pic with PIL
        pic = Image.open(picFullPath)

        # resizing to match network input shape
        pic = transforms.Resize((448, 448))(pic)
        pic = transforms.ToTensor()(pic)

        boxes = self.labels[index]

        if self.transform:
            pic = self.transform(pic)


        label = torch.zeros((self.G,self.G,self.C+5))
        for box in boxes:
            x_grid_no = int((box[2]*self.G))

            y_grid_no = int((box[3]*self.G))


            if self.test :
                # for test cases to avoid remapping the boxes in global coordination
                # P.S. test samples can't be used in the loss function because out and in will be on a different scale
                x_offset = box[2]
                y_offset = box[3]
                w = box[4]
                h = box[5]
            else :
                x_offset = (box[2]*self.G)%1
                y_offset = (box[3]*self.G)%1
                
                w = box[4]*self.G
                h = box[5]*self.G

            label[y_grid_no,x_grid_no,self.C:] = torch.tensor([1.00 , x_offset, y_offset, w , h])
            label[y_grid_no,x_grid_no,box[0]]= 1


        return pic,label 
    
    def plotImage(self,index):
        
        pic = Image.open(os.path.join(self.baseDir,self.pics[index]))

        # just for resolution
        pic = np.array(transforms.Resize((1080, 1080))(pic))

        boxes = self.labels[index]

        plot_image(pic,boxes)