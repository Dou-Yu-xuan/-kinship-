import torch.nn as nn
import utils.loader
from models import cnn
from utils.transform import *
from utils.loader import *
from utils.loss import *

class basic_cnn(object):
    ################### data ####################

    description = 'standard training using cnn_basic Network'
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
    trans =  [cnn_train_transform,cnn_test_transform]
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
    model = cnn
    ## structure
    model_name = 'basic_cnn'
    ## image size
    imsize = (6,64,64)
    ## epoch numbers 60-100
    epoch_num = 60
    ## batch  (v)
    train_batch =  128
    test_batch = 32
    ## learning rate (v)
    lr    = 0.0001
    ## learning rate decay
    lr_decay = 0.5
    ##
    momentum = 0.9
    ##
    lr_milestones = [100]
    ## regularization (v)
    weight_decay = 0.005

    ## number of cross validation
    cross_num = 5
    ## how many steps show the loss
    show_lstep = 17
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


#############################
class cnn_points(object):
    ################### data ####################

    description = 'standard training using cnn_points Network'
    ################### data ####################
    ## kinship typle e.g. F-D, M-D, F-S etc.
    kintype  = 'F-D'
    ## list path
    list_path = 'data/train_list/F-D.pkl'
    ## data path
    img_root  = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/frames/F-D'
    ## dataset
    Dataset =  NemoDataset_cnn_points
    ## transformer
    trans =  [cnn_points_train_transform,cnn_points_test_transform]
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
    model = cnn
    ## structure
    model_name = 'cnn_points'
    ## image size
    imsize = (6,64,64)
    ## epoch numbers 60-100
    epoch_num = 60
    ## batch  (v)
    train_batch =  128
    test_batch = 32
    ## learning rate (v)
    lr    = 0.0001
    ## learning rate decay
    lr_decay = 0.5
    ##
    momentum = 0.9
    ##
    lr_milestones = [100]
    ## regularization (v)
    weight_decay = 0.001

    ## number of cross validation
    cross_num = 5
    ## how many steps show the loss
    show_lstep = 17
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

