from __future__ import print_function
import os
from dataset.dataset import Dataset
import torch
from torch.utils import data
import torch.nn.functional as F
from models.mobilenet_v1 import mobilenetv1
import torchvision
import torch.backends.cudnn as cudnn
import numpy as np
import random
import time
from config.config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR

from models.resnet import  resnet34


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


#tensorflow/kereas n*h*w*c
#pytorch   n*c*h*w
#caffe   n*c*h*w


if __name__ == '__main__':

      opt = Config()

      criterion = torch.nn.CrossEntropyLoss()
      criterion_distill = torch.nn.KLDivLoss()
      S_model = mobilenetv1(num_classes=opt.num_classes)
      T_model  = resnet34(pretrained=False)
      fc_inputs = T_model.fc.in_features
      T_model.fc = torch.nn.Linear(fc_inputs, 2)
      #下面加载（在自己数据集上训练好的）教师模型
      T_model.load_state_dict(torch.load('./checkpoints/resnet34_Teacher_20_0.9990.pth'),strict=False)  
      for param in T_model.parameters(): #固定参数不更新
            param.requires_grad = False
      T_model.cuda()
      S_model.cuda()
      cudnn.benchmark = True

      #定义训练数据加载模块
      start = time.time()
      train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
      trainloader = data.DataLoader(train_dataset,
                                    batch_size=opt.train_batch_size,
                                    shuffle=False,
                                    pin_memory = True,
                                    num_workers=opt.num_workers) 
      print('{} train iters per epoch,time cost {} s.'.format(len(trainloader),time.time()-start))
      
      #定义验证数据加载模块
      start = time.time()
      val_dataset = Dataset(opt.val_root, opt.val_list, phase='val', input_shape=opt.input_shape)
      valloader = data.DataLoader(val_dataset,
                                    batch_size=opt.val_batch_size,
                                    shuffle=False,
                                    pin_memory = True,
                                    num_workers=opt.num_workers) 
      print('{} val iters in one epoch.time cost {} s.'.format(len(valloader),time.time()-start)) 
      print('testdataset all num = {}'.format(len(val_dataset)))



      #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, T_model.parameters()), lr= lr=opt.lr)
        
      if opt.optimizer == 'sgd':
            optimizer = torch.optim.SGD([{'params':S_model.parameters()}],momentum =opt.momentum,
                                          lr=opt.lr, weight_decay=opt.weight_decay)
      else:
            optimizer = torch.optim.Adam([{'params':S_model.parameters()}],momentum =opt.momentum,
                                          lr=opt.lr, weight_decay=opt.weight_decay)
      
      #optimizer = torch.optim.SGD(S_model.parameters(),lr = opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
      #optimizer = torch.optim.Adam(S_model.parameters(),lr = opt.lr,amsgrad=True,weight_decay=opt.weight_decay)
  
      scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)
      start = time.time()
      T = 5 #温度
      for i in range( opt.max_epoch):
            '''
            '''
            #S_model.train()
            for ii, data in enumerate(trainloader):
                  start_0 =time.time()
                  data_input, label = data
                  data_input = data_input.cuda()#to(device)
                  #label = label.to(device).long()
                  label = label.cuda().long()
                  optimizer.zero_grad()  

                  S_model = S_model.cuda()    
                  S_logits =  S_model(data_input)
                  #交叉熵损失
                  loss_CE =  criterion(S_logits, label)

                  T_model = T_model.cuda()
                  T_logits =  T_model(data_input)
                  outputs_S = F.log_softmax(S_logits/T,dim=1)
                  outputs_T =          F.softmax(T_logits/T,dim=1)
                  #蒸馏损失
                  loss_KD = criterion_distill(outputs_S, outputs_T)  * T  * T 

                  loss = (1- opt.alpha) * loss_CE +   opt.alpha * loss_KD
    
                  loss.backward()
     
                  optimizer.step()

                  if (ii+1) % opt.print_freq == 0 :
                        S_logits = S_logits.data.cpu().numpy()
                        S_logits = np.argmax(S_logits, axis=1)
                        label = label.data.cpu().numpy()

                        acc = np.mean((S_logits == label).astype(int))
                        speed = (time.time() - start)/opt.print_freq
                        time_str = time.asctime(time.localtime(time.time()))
                        print('Train epoch:{}  , iter: {}  , {} s/iter , loss: {},  acc: {}'.format( (i+1), (ii+1), speed, loss.item(), acc))

                        start = time.time()
      
            scheduler.step()

            if (i+1) % opt.save_interval == 0:
                  save_model(S_model, opt.checkpoints_path, opt.backbone+"_Student", (i+1))

            

            S_model.eval() 
            val_losses = []
            correct_num = 0

            #在学生网络上测试
            with torch.no_grad():
                  for jj, val_data in enumerate(valloader):
                        val_data_input, val_label = val_data
                        val_data_input = val_data_input.cuda()#to(device)
                        val_label = val_label.cuda().long()#to(device).long()
                        
                        output = S_model(val_data_input)
                        val_loss = criterion(output, val_label)
                        val_losses.append(val_loss.cpu().numpy())

                        output = output.data.cpu().numpy()
                        output = np.argmax(output, axis=1)
                        val_label = val_label.data.cpu().numpy()

                        temp_array = np.array(output == val_label).astype(int)
            
                        correct_num+=np.sum(temp_array)       

            print('Epoch {}  , val  loss: {:.4f}, val accurary: {:.4f} '.format(i+1, np.mean(val_losses), correct_num/len(val_dataset) ))
 
