import torch.nn as nn
import utils.loader
from models import attention_Net
from utils.transform import *
from utils.loader import *
from utils.loss import *

class Config(object):
    ################### data ####################

    description = 'standard training using attention Network'
    ################### data ####################
    ## kinship typle e.g. F-D, M-D, F-S etc.
    kintype  = 'F-D'
    ## list path
    list_path = 'data/train_list/F-D.pkl'
    ## data path
    img_root  = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/frames/F-D'
    ## dataset
    Dataset =  NemoDataset
    ## transformer
    trans =  [attention_train_transform,attention_test_transform]
    ################### loader ####################

    sf_sequence = True
    ## shuflle the list after each epoch
    cross_shuffle = True
    ##
    sf_aln = True
    ###
    dataloder_shuffle = True

    ################### train ####################
    ######## model

    ## modelname
    model = attention_Net
    ## structure
    model_name = 'attenNet'
    ## image size
    imsize = (6,64,64)
    ## epoch numbers 60-100
    epoch_num = 60
    ## batch  (v)
    train_batch =  32
    test_batch = 32
    ## learning rate (v)
    lr    = 0.0001
    ## learning rate decay
    lr_decay = 0.5
    ##
    momentum = 0.9
    ##
    lr_milestones = [100]
    ## regularization
    weight_decay = 5e-4

    ## number of cross validation
    cross_num = 5
    ## how many steps show the loss
    show_lstep = 50
    ##  frequent of printing the evaluation acc
    prt_fr = 1

    ##  frequent of printing the training data acc
    prt_fr_train = 100

    ## loss
    loss = nn.CrossEntropyLoss
    ## optimal  #(v)
    optimal = 'adam'

    ######## record
    ## save the training accuracy
    save_tacc = True
    ## whether load pretrained model
    reload = ''
    ## save trained model
    save_ck = True
    ##
    logs_name = 'data/logs'
    ##
    savemis = False
    ##
    save_graph = False



class Mask(object):
    ################### data ####################

    description = 'standard training using attention + Mask Network'
    ################### data ####################
    ## kinship typle e.g. F-D, M-D, F-S etc.
    kintype  = 'F-D'
    ## list path
    list_path = 'data/train_list/F-D.pkl'
    ## data path
    img_root  = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/frames/F-D'
    ## dataset
    Dataset =  NemoDataset_attention_msk
    ## transformer
    trans =  [attention_train_transform,attention_test_transform]
    ################### loader ####################

    sf_sequence = True
    ## shuflle the list after each epo
    cross_shuffle = True
    ##
    sf_aln = True
    ###
    dataloder_shuffle = True

    ################### train ####################
    ######## model

    ## modelname
    model = attention_Net
    ## structure
    model_name = 'attenNet_mask'
    ## image size
    imsize = (6,64,64)
    ## epoch numbers 60-100
    epoch_num = 60
    ## batch  (v)
    train_batch =  32
    test_batch = 32
    ## learning rate (v)
    lr    = 0.0001
    ## learning rate decay
    lr_decay = 0.5
    ##
    momentum = 0.9
    ##
    lr_milestones = [100]
    ## regularization
    weight_decay = 5e-4

    ## number of cross validation
    cross_num = 5
    ## how many steps show the loss
    show_lstep = 50
    ##  frequent of printing the evaluation acc
    prt_fr = 1

    ##  frequent of printing the training data acc
    prt_fr_train = 60

    ## loss
    loss = nn.CrossEntropyLoss
    ## optimal  #(v)
    optimal = 'adam'

    ######## record
    ## save the training accuracy
    save_tacc = True
    ## whether load pretrained model
    reload = ''
    ## save trained model
    save_ck = True
    ##
    logs_name = 'data/logs'
    ##
    savemis = False
    ##
    save_graph = False




class mask_multi(object):
    ################### data ####################

    description = 'standard training using attention + Mask multi Network'
    ################### data ####################
    ## kinship typle e.g. F-D, M-D, F-S etc.
    kintype  = 'F-D'
    ## list path
    list_path = 'data/train_list/F-D.pkl'
    ## data path
    img_root  = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/frames/F-D'
    ## dataset
    Dataset =  NemoDataset_attention_multi
    ## transformer
    trans =  [attention_train_transform,attention_test_transform]
    ################### loader ####################

    sf_sequence = True
    ## shuflle the list after each epo
    cross_shuffle = True
    ##
    sf_aln = True
    ###
    dataloder_shuffle = True

    ################### train ####################
    ######## model

    ## modelname
    model = attention_Net
    ## structure
    model_name = 'attenNet_multi'
    ## image size
    imsize = (6,64,64)
    ## epoch numbers 60-100
    epoch_num = 60
    ## batch  (v)
    train_batch =  32
    test_batch = 32
    ## learning rate (v)
    lr    = 0.0001
    ## learning rate decay
    lr_decay = 0.5
    ##
    momentum = 0.9
    ##
    lr_milestones = [100]
    ## regularization
    weight_decay = 5e-4

    ## number of cross validation
    cross_num = 5
    ## how many steps show the loss
    show_lstep = 50
    ##  frequent of printing the evaluation acc
    prt_fr = 1

    ##  frequent of printing the training data acc
    prt_fr_train = 60

    ## loss
    loss = nn.CrossEntropyLoss
    ## optimal  #(v)
    optimal = 'adam'

    ######## record
    ## save the training accuracy
    save_tacc = True
    ## whether load pretrained model
    reload = ''
    ## save trained model
    save_ck = True
    ##
    logs_name = 'data/logs'
    ##
    savemis = False
    ##
    save_graph = False

####################### KinfaceW I&II ############################
# class kin_config(object):
#     ################### data ####################
#
#     des = 'train FaceWII'
#     ################### data ####################
#     ## kinship typle e.g. F-D, M-D, F-S etc.
#     kintype  = 'F-D'
#     ## list path
#     list_path = '/home/wei/Documents/DATA/kinship/KinFaceW-I/meta_data/fd_pairs.mat'
#     ## data path
#     img_root  = '/home/wei/Documents/DATA/kinship/KinFaceW-I/images/father-dau'
#     ## dataset
#     Dataset =  KinDataset
#     ## transformer
#     trans =  [attention_train_transform,attention_test_transform]
#     ################### loader ####################
#
#     sf_sequence = True
#     ## shuflle the list after each epoch
#     cross_shuffle = True
#     ##
#     sf_aln = True
#     ###
#     dataloder_shuffle = True
#
#     ################### train ####################
#     ######## model
#
#     ## modelname
#     model = attention_Net
#     ## structure
#     model_name = 'attenNet'
#     ## image size
#     imsize = (6,64,64)
#     ## epoch numbers
#     epoch_num = 300
#     ## batch
#     train_batch =  64
#     test_batch = 64
#     ## learning rate
#     lr    = 0.01
#     ## learning rate decay
#     lr_decay = 0.5
#     ##
#     momentum = 0.9
#     ##
#     # lr_milestones = [180, 250, 300, 400, 500, 550]
#     lr_milestones = [180, 250,400,500]
#     ## regularization
#     weight_decay = 5e-6
#
#     ## number of cross validation
#     cross_num = 5
#     ## how many steps show the loss
#     show_lstep = 1
#     ##  frequent of printing the evaluation acc
#     prt_fr = 1
#     ## loss
#     loss = nn.CrossEntropyLoss
#     ## optimal
#     optimal = 'sgd'
#
#     ######## record
#     ## save the training accuracy
#     save_tacc = False
#     ## whether load pretrained model
#     reload = ''
#     ## save trained model
#     save_ck = True
#     ##
#     logs_name = 'data/logs'
#     ##
#     savemis = False
#     ##
#     save_graph = False
