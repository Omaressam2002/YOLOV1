import torch
import pandas as pd
from PIL import Image
import os
import torchvision.transforms as transforms

class ImageNet(torch.utils.data.Dataset): #Datasets
    def __init__2(self,classes,train = True):
        if train :
            csv = 'ImageNet/LOC_train_solution.csv'
            self.img_path = 'ImageNet/ILSVRC/Data/CLS-LOC/train'
            self.train = train
        else :
            csv = 'ImageNet/LOC_val_solution.csv' 
            self.img_path = 'ImageNet/ILSVRC/Data/CLS-LOC/val'
            self.train = train
            
        self.data = (pd.read_csv(csv))[:60000] # only a part of the training set
        self.classes = classes

    
    def __init__(self,classes,train = True):

        self.classes = classes
        
        if train :
            csv = 'ImageNet/LOC_train_solution.csv'
            self.img_path = 'ImageNet/ILSVRC/Data/CLS-LOC/train'
            self.data = []
            self.labels = []
            for dir in list(os.listdir(self.img_path)):
                if dir[:1] != "n":
                    continue
                class_path = os.path.join(self.img_path,dir)
                pics = list(os.listdir(class_path))
                for i in range(60):
                    self.labels.append(self.classes[dir])
                    self.data.append(os.path.join(class_path,pics[i]))   
            self.train = train
        else :
            csv = 'ImageNet/LOC_val_solution.csv' 
            self.img_path = 'ImageNet/ILSVRC/Data/CLS-LOC/val'
            self.data = (pd.read_csv(csv))[:5000]
            self.train = train

    
    def __len__(self):
        return len(self.data)
        #return 256  #########################################CHANGE#################

    
    def __getitem__2(self,index):

        class_id = (self.data.iloc[index,1].split())[0] #n02017213 
        img_id = self.data.iloc[index,0] #n02017213_7894

        if self.train:
            path = class_id+'/'+img_id+'.jpeg'
        else:
            path = img_id+'.jpeg'
        
        img_i_path = os.path.join(self.img_path,path)

        img = Image.open(img_i_path)
        img = transforms.Resize((448, 448))(img)
        img = transforms.ToTensor()(img)
        if img.shape[0] != 3:
            img = Image.open(img_i_path).convert("RGB")
            img = transforms.Resize((448, 448))(img)
            img = transforms.ToTensor()(img)

        class_idx = self.classes[class_id]

        #label = torch.zeros((1,1000))
        #label[0,class_idx] = 1 #one hot encoded

        
        return img,class_idx

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



class VOCDataset(torch.utils.data.Dataset): #Datasets
    def __init__(self,csv,img_path,label_path,grids=7,boxes=2,classes=20,transform=None):
        self.data = pd.read_csv(csv)
        self.img_path = img_path
        self.label_path = label_path
        self.G = grids
        self.B = boxes
        self.C = classes
        self.transform = transform

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self,index):
        
        label_i_path = os.path.join(self.label_path,self.data.iloc[index,1])
        boxes = []
        
        with open(label_i_path) as f:
            lines = f.readlines()
            
            for l in lines:
                class_label,x,y,w,h = l.replace("\n","").split()
                boxes.append([int(class_label),float(x),float(y),float(w),float(h)])
            
        label = torch.zeros((self.G,self.G,self.C+5))
        for box in boxes:
            x_grid_no = int((box[1]*self.G))
            x_offset = (box[1]*self.G)%1

            y_grid_no = int((box[2]*self.G))
            y_offset = (box[2]*self.G)%1
            
            #  20      21        22       23   24
            # pobj  x_offset  y_offset    w    h
            label[y_grid_no,x_grid_no,20:] = torch.tensor([1.00 , x_offset, y_offset, box[3]*self.G , box[4]*self.G])
            label[y_grid_no,x_grid_no,box[0]]= 1 #class label
        
        img_i_path = os.path.join(self.img_path,self.data.iloc[index,0])
        img = Image.open(img_i_path)
        if self.transform:
            img,boxes  = self.transform(img,boxes) # will resize the image

        
        return img,label