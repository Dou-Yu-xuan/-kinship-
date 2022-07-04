from utils import lpq_top
import sys
import math
import numpy as np
from scipy import signal
from scipy.spatial import distance_matrix
from skimage import io
from skimage.color import rgb2gray
import pickle
import os
# img_pth = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/framses_resize64/F-D/f03-3'
# img_seq = []
from tqdm import tqdm
from utils.loader import cross_validation
from keras_vggface.vggface import VGGFace
from keras.engine import  Model
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image as keras_image
from torchvision import transforms
from facenet_pytorch import  InceptionResnetV1,fixed_image_standardization
from PIL import Image
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
import tensorflow as tf
# config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16} )
# sess = tf.Session(config=config)
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# K.set_session(sess)
# K.tensorflow_backend._get_available_gpus()
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def read_pkl(nemo_ls,im_pth):
    with open(nemo_ls, 'rb') as fp:
        ori_ls = pickle.load(fp)
    new_list = []
    for temp in ori_ls:
        im1_pth = os.path.join(im_pth, temp[2])
        im2_pth = os.path.join(im_pth, temp[3])
        new_list.append([temp[0],temp[1],im1_pth,im2_pth])
    return ori_ls,new_list


def get_img_seq(img_pth):
    img_seq = []
    for i in range(1, 100):
        image1 = rgb2gray(io.imread(img_pth + '/frame{:03d}.jpg'.format(i)))
        img_seq.append(image1)
    image_seq = np.array(img_seq)
    image_seq = image_seq.reshape((image_seq.shape[0], image_seq.shape[1], image_seq.shape[2], 1))
    return image_seq


def get_lpq_top(img_seq):
    feature = np.array([])
    w_sizes = [3.,5.,7.,9.,11.,13.,15.,17.]
    # w_sizes = [3.,5.]

    for w in w_sizes:
        win = np.array([w, w, w])
        lpq_top_f = lpq_top.LPQ_TOP(img_seq,winSize=win)
        feature = np.concatenate((feature,lpq_top_f),axis = 0)
    return feature


def lpq_top_feature_extractor(im1_pth, im2_pth):
    im1 = get_img_seq(im1_pth)
    im2 = get_img_seq(im2_pth)
    im1_fea = get_lpq_top(im1)
    im2_fea = get_lpq_top(im2)
    diff = np.absolute(im1_fea-im2_fea)/np.sum(im1_fea+im2_fea)
    # diff = np.linalg.norm(diff)
    return diff


class VGGFACE_EXTRACTOR():
    def __init__(self):
        vgg_model =  VGGFace(model='vgg16')
        layer_name = 'fc7'  # edit this line
        out = vgg_model.get_layer(layer_name).output
        self.model = Model(vgg_model.input, out)

    def load_model(self,pth):
        return
    def feature_extract(self,img):

        fea = self.model.predict(img)
        return fea

VGGFACE = VGGFACE_EXTRACTOR()


def get_vggface(img):
    img = keras_image.load_img(img, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)
    features = VGGFACE.feature_extract(x)
    return features

def vggface_extractor(im1_pth,im2_pth):
    # im1_fea = np.array([])
    # im2_fea = np.array([])
    # print('here now')
    for i in range(0, 100):
        image1 = im1_pth + '/frame{:03d}.jpg'.format(i)
        if i ==0:
            im1_fea = get_vggface(image1)
        else:
            im1_fea = np.concatenate((im1_fea, get_vggface(image1)),axis = 0)
        # print(6)
        image2 = im2_pth + '/frame{:03d}.jpg'.format(i)
        if i ==0:
            im2_fea = get_vggface(image2)
        else:
            im2_fea = np.concatenate((im2_fea, get_vggface(image2)),axis = 0)
    im1_fea = np.mean(im1_fea,axis = 0)
    im2_fea = np.mean(im2_fea,axis = 0)
    diff = np.absolute(im1_fea-im2_fea)/np.sum(im1_fea+im2_fea)
    # diff = np.linalg.norm(diff)
    return diff

def get_cross(kin_list,cross):

    lpg_ls = [i[2] for i in kin_list if i[0] in cross]
    vgg_ls = [i[3] for i in kin_list if i[0] in cross]
    lb = [i[1] for i in kin_list if i[0] in cross]
    return  lpg_ls,vgg_ls,lb

def save_file(files,name):
    with open('{}.pickle'.format(name),'wb') as ff:
        pickle.dump(files,ff)


def read_file(name):
    with open('{}.pickle'.format(name), 'rb') as handle:
        b = pickle.load(handle)
    return b

if __name__ =='__main__':
    train_ls = ['F-D', 'F-S', 'M-D', 'M-S', 'B-B', 'S-S', 'B-S']
    # train_ls = ['F-D']

    for kin_type in train_ls:

        lb_pth = './data/label/{}.pkl'.format(kin_type)
        img_pth = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(kin_type)
        nemo_ls,data_ls = read_pkl(lb_pth,img_pth)

        feature_ls = []

        for pair_temp in tqdm(data_ls):
            img1_pth = pair_temp[2]
            img2_pth = pair_temp[3]
            fea_lpq = lpq_top_feature_extractor(img1_pth,img2_pth)
            # train_lpq_top_feature_ls.append(fea_lpq)
            fea_vgg = vggface_extractor(img1_pth,img2_pth)
            # train_vggface_feature_ls.append(fea_vgg)
            # train_label_ls.append(pair_temp[1])
            feature_ls.append([pair_temp[0],pair_temp[1],fea_lpq,fea_vgg])


        save_file(feature_ls,'{}_features'.format(kin_type))
        # feature_ls = read_file('{}_features'.format(kin_type))


        ########################### cross validation
        print('training {}'.format(kin_type))
        mean_acc = 0.0
        for tra_id, tes_id in cross_validation(5):
            train_lpq_top_feature_ls,train_vggface_feature_ls,train_label_ls\
                = get_cross(feature_ls,tra_id)
            test_lpq_top_feature_ls,test_vggface_feature_ls,test_label_ls\
                = get_cross(feature_ls,tes_id)

            ################# train
            clf = make_pipeline(StandardScaler(),SVC(gamma='auto',probability=True))
            # clf = SVC(gamma='auto',probability=True)
            clf.fit(train_lpq_top_feature_ls,train_label_ls)

            clf2 = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
            # clf = SVC(gamma='auto',probability=True)
            clf2.fit(train_vggface_feature_ls, train_label_ls)
            ################## test
            predic1 = clf.predict_proba(test_lpq_top_feature_ls)
            predic2 = clf2.predict_proba(test_vggface_feature_ls)
            predic = predic1+predic2
            # print(predic)
            predic = np.argmin(predic,axis=1)
            acc = sum(predic==test_label_ls)/len(test_label_ls)
            mean_acc +=acc
        mean_acc = mean_acc/5
        print(mean_acc)
        print('#'*30)
