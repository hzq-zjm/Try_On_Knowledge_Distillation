# -*- coding: utf-8 -*-
'''Created on 2020-7-30  14:43
    @author: hzq
'''
from __future__ import print_function
import os
import cv2
import torch
from models.mobilenet_v1 import mobilenetv1
import numpy as np
import time
from config.config import Config
from torch.nn import Softmax
from PIL import Image
import shutil

def load_image(img_path):
    
    image = Image.open(img_path)
    image = image.resize((96, 112),Image.ANTIALIAS) 
    image = np.array(image, dtype = np.float32)
    image = image.transpose((2, 0, 1))  #n*c*h*w
    #print("after transpose : {}".format(image.shape ))   
    image = image[np.newaxis, :, :, :]
    #print("after newaxis : {}".format(image.shape ))   
    image -= 127.5
    image /= 127.5
    #print("after float32 : {}".format(image.shape ))   
    return image

if __name__ == "__main__":
      opt = Config()
      model = mobilenetv1(num_classes=opt.num_classes)
      #model = DataParallel(model)
      # load_model(model, opt.test_model_path)
      model.load_state_dict(torch.load('./checkpoints/mobilenet_v1_big_0.9975.pth'))
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.to(device)
      #model.to(torch.device("cuda"))
      model.eval()


      imgs_path = '/home/hzq/face_hzq/face_withmask_classify/戴口罩人脸_new'
      mask_path = '/home/hzq/face_hzq/face_withmask_classify/mask_result'
      no_mask_path = '/home/hzq/face_hzq/face_withmask_classify/no_mask_result'
      for img in os.listdir(imgs_path):
            img_path = imgs_path +'/'+img
            img = load_image(img_path)
            input = torch.from_numpy(img).float()
            input = input.to(device)
            output = model(input)
            softmax = Softmax(dim=1)  #按行求softmax
            print(output)
            output = softmax(output)
            output = output.data.cpu().numpy()
            print(output)
            pred = np.argmax(output)
            if pred ==1:
                  #print('With mask!')
                  shutil.copy(img_path, mask_path)
            else:
                  #print('No mask!')
                  shutil.copy(img_path,no_mask_path)
