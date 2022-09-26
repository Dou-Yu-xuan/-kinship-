from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np
import zipfile

# from dataset import ImageDataset
# from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
from tqdm import tqdm
import pickle
from sklearn.metrics import roc_curve, auc,accuracy_score

parser = argparse.ArgumentParser(description='PyTorch sphereface lfw')
parser.add_argument('--net','-n', default='sphere20a', type=str)
parser.add_argument('--model','-m', default='./model/sphere20a_20171020.pth', type=str)
args = parser.parse_args()

predicts=[]
net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load(args.model))
net.cuda()
net.eval()
net.feature = True

##################

ls = ['F-D', 'F-S', 'M-D', 'M-S', 'B-B', 'S-S', 'B-S']
img_root = '/local/wwang/Nemo/kin_simple/frames/'
best_threshold = 0.0
average_accuracy = 0.0
for current_lb in ls:
    # current_lb = ls[i]
    lb_pth = './label/{}.pkl'.format(current_lb)
    with open (lb_pth, 'rb') as fp:
            nemo_ls = pickle.load(fp)

    pairs = []
    for temp_lb in nemo_ls:
        for num_a in range(100):
            img1_pth = img_root + current_lb + '/'+ temp_lb[2] + '/' + 'frame{:0>3d}.jpg'.format(num_a) 
            img2_pth = img_root + current_lb + '/'+ temp_lb[3] + '/' + 'frame{:0>3d}.jpg'.format(num_a) 
            pairs.append((img1_pth, img2_pth, int(temp_lb[1])))

    y_true = [label for _, _, label in pairs]
    pair_list = [(img1, img2) for img1, img2, _ in pairs]
    all_paths = [o for o, _ in pair_list] + [o for _, o in pair_list]
###################
    predicts=[]
    for i, (img1_temp_pth,img2_temp_pth)in  tqdm(enumerate(pair_list)):


        img1 = cv2.imread(img1_temp_pth)
        img2 = cv2.imread(img2_temp_pth)
        img1 = cv2.resize(img1[:,11:148,:], (112, 96))
        img2 = cv2.resize(img2[:,11:148,:], (112, 96))

        # for i in range(len(imglist)):
        img1 = img1.transpose(2, 0, 1).reshape((1,3,112,96))
        img1 = (img1-127.5)/128.0

        img2 = img2.transpose(2, 0, 1).reshape((1,3,112,96)) #112,96
        img2 = (img2-127.5)/128.0
        imglist = [img1, img2]
        img = np.vstack(imglist)
        img = Variable(torch.from_numpy(img).float(),volatile=True).cuda()
        output = net(img)
        f = output.data
        f1,f2 = f[0],f[1]
        cosdistance = (f1.dot(f2)/(f1.norm()*f2.norm()+1e-5)).cpu().numpy()
        predicts.append(cosdistance)
    

    predicts = np.stack(predicts, axis=0)
    fpr, tpr, thresholds = roc_curve(y_true, predicts)
    y = tpr - fpr
    youden_index = np.argmax(y)
    op_thr = thresholds[youden_index]
    best_threshold += op_thr
    pred = predicts > op_thr ## set threshold
    acc = accuracy_score(y_true, pred)
    print("current_lb: {}, acc: {}".format(current_lb, acc))
    print("current_threshold: {}".format(op_thr))
    average_accuracy += acc
print("average_accuracy: {}".format(average_accuracy/len(ls)))
print("best_threshold: {}".format(best_threshold/7))


