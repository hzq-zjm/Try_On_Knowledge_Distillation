import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
import cv2
import sys


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(3, 112, 96)):   
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        self.imgs = [os.path.join(root, img[:-1]) for img in imgs]
        #self.imgs = np.random.permutation(imgs)      #对imgs随机排序

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])                                   

        self.height = input_shape[1]
        self.width  = input_shape[2]

        #normalize = T.Normalize(mean=[0.5], std=[0.5])   # 单通道

        if self.phase == 'train':
            self.transforms = T.Compose([
                #T.RandomCrop(self.input_shape[1:]),

                T.RandomHorizontalFlip(), 

                #随机长宽比裁剪 transforms.RandomResizedCrop
                #功能：随机大小，随机长宽比裁剪原始图片，最后将图片resize到设定好的size
                #参数：
                #size- 输出的分辨率
                # scale- 随机crop的大小区间，如scale=(0.08, 1.0)，表示随机crop出来的图片会在的0.08倍至1倍之间。
                #ratio- 随机长宽比设置
                # interpolation- 插值的方法，默认为双线性插值(PIL.Image.BILINEAR)
                #T.RandomResizedCrop((self.height,self.width), scale=(0.8, 1), ratio=(3. / 4., 4. / 3.) , interpolation=Image.BILINEAR),
                T.RandomRotation(20, expand=False),
                
                #对图像进行随机遮挡：
                #T.RandomErasing(p=1,# 概率
                #scale=(0.02, 0.33),#文献值，遮挡区域的面积
                #    ratio=(0.3, 3.3), #缩放值
                #    value=(254/255, 0, 0)),#遮挡区域的颜色填充，如果输入字符串就会是随机彩色像素值。
                
                #颜色变换模块，可以改变亮度，对比度，饱和度，色相hue
                T.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2), hue=(-0.5,0.5)),
                T.Resize((self.height, self.width)),
                T.ToTensor(),    #归一化到【0,1】之间，即除以255
                normalize      #经过normalize后，归一化到【-1,+1】之间
            ])

        elif self.phase == 'val':
            self.transforms = T.Compose([
                T.Resize((self.height, self.width)),
                T.ToTensor(), 
                normalize    
            ])            
        else:
            self.transforms = T.Compose([
                T.ToTensor(),   #归一化到【0,1】之间，即除以255
                normalize     #对数据按通道进行标准化，即先减均值，再除以标准差，注意是 hwc
            ])

    def __getitem__(self, index):
        sample = self.imgs[index]
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path)
        #data = data.convert('L')   #转化为灰度图
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='test',
                      input_shape=(3, 112, 96))

    trainloader = data.DataLoader(dataset, batch_size=128)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()   #将若干副图拼成一幅图，方便做数据展示
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))            #    h,w,c
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]    

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
