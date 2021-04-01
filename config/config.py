'''
Author: hzqq
Date: 2020-07-30 17:17:55
LastEditTime: 2021-03-31 16:43:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /face_withmask_classify/config/config.py
'''
class Config(object):

    backbone = 'mobilenetV1'
    num_classes = 2
    loss = 'CrossEntropyLoss'
    distill =  True   #开启蒸馏训练
    alpha = 0.9     

    train_root = './dataset'
    train_list = './dataset/train.txt'

    val_root = './dataset'
    val_list =  './dataset/val.txt'


    checkpoints_path = 'checkpoints'

    save_interval = 5

    train_batch_size =128  # batch size
    test_batch_size = 128  #test lfw

    val_batch_size = 128

    input_shape = (3, 112, 96)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 3  # how many workers for loading data
    print_freq = 100  # print info every N batch


    max_epoch = 50
    lr = 1e-3  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4

    gamma=0.1
    momentum = 0.9

