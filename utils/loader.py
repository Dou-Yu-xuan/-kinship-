import scipy.io
import os
import torch
from torch.utils.data import Dataset,DataLoader
import random
from random import shuffle
import pickle
from utils.transform import *
from PIL import Image
import copy
import matplotlib.pyplot as plt
import torchvision


class cross_validation():
    """
    get  n-folds cross validation,
    generate [1,2,3,4,...n]
    yeild [1,..del(remove),..n] and [remove]
    """
    def __init__(self,n):
        self.n = n

    def __iter__(self):
        for i in range(self.n,0,-1):
            train_ls = self._tra_ls(i)
            yield train_ls, [i]

    def _tra_ls(self,remove):
        return [i for i in range(1,self.n+1) if i !=remove ]



class NemoDataset(Dataset):
    def __init__(self,list_path,img_root, cross_vali = None,transform = None,
                 sf_sequence = False,cross_shuffle = False,sf_aln = False,test = False):

        """
        :param list_path:       folder list of training/testing dataset
        :param img_root:     image path
        :param cross_vali:    cross validation's folds: e.g. [1,2,4,5]
        :param transform:     add data augmentation
        :param sf_sequence:   shuffle the sequence order while training
        :param cross_shuffle: shuffle names among pair list
        :param sf_aln:        whether shuffle all names or only img2s' names
        :param test:          weather test
        """
        # kin_list is the whole 1,2,3,4,5 folds from mat
        self.kin_list = self._read_nemols(list_path)
        self.im_root = img_root
        self.transform = transform
        self.cout = 0
        self.test = test
        self.sf_sq = sf_sequence
        if cross_vali is not None:
            #extract matched folds e.g. [1,2,4,5]
            self.kin_list = self._get_cross(cross_vali)

        self.fr_list  = self._get_frames(self.kin_list)
        # if cross_shuffle:
        self.crf = cross_shuffle  # cross shuffle
        self.lth = len(self.kin_list)
        self.flth = len(self.fr_list)
        # store all img2/ all_(img1+img2) list for cross shuffle,
        self.img2_list = [i[3] for i in self.kin_list]
        self.alln_list = self.get_alln(self.kin_list)
        self.sf_aln = sf_aln
        # if self.test:
        #     self.kin_list = self._test_ls(self.kin_list,self.fr_list)
        # self._cross_shuffle()

    def get_alln(self,ls):
        all_name = []
        for i in ls:
            all_name.append(i[2])
            all_name.append(i[3])
        return all_name

    def __len__(self):
        return len(self.fr_list)

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        if not self.test:
            # extract img1
            img1_path = self.fr_list[item][2]
            img1 = Image.open(img1_path)
            # extract img2
            img2_path = self.fr_list[item][3]
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.fr_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1,img2))
            # after each epoch, shuffle once
            if self.crf:
                self.cout +=1
                if self.cout == self.flth:
                    self.cout = 0
                    self._cross_shuffle()
                    self.fr_list = self._get_frames(self.kin_list)

            return imgs,kin,img1_path,img2_path
        else:
            # extract img1
            img1_path = self.fr_list[item][2]
            img1 = Image.open(img1_path)

            # extract img2
            img2_path = self.fr_list[item][3]
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.fr_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1, img2))

            return imgs,kin,img1_path,img2_path

    def _cross_shuffle(self):
        """
        shuffle the second images name after each epoch
        :return:
        """
        if self.sf_aln:
            im2_ls = self.alln_list
        else:
            im2_ls = self.img2_list
        rand_lth = len(im2_ls)
        new_pair_list = []
        for pair_l in self.kin_list:
            if pair_l[1] == 0:
                new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                while pair_l[2].split('-')[0] == new_img2.split('-')[0]:
                    new_img2 = im2_ls[random.randint(0, rand_lth-1)]
                pair_l[3] = new_img2
            new_pair_list.append(pair_l)

        self.kin_list = new_pair_list

    def _read_nemols(self,nemo_ls):

        with open (nemo_ls, 'rb') as fp:
            nemo_ls = pickle.load(fp)

        return nemo_ls

    def _get_paired_frame_ls(self,ls):
        """
        :param ls: [1,1,'f03-3','f03-1']
        :return: matched frames pth, if one of the fold's
         frames is less, then extend these folder's frame
         to the same length
        """
        paired_ls = []
        m1_pth = os.path.join(self.im_root,ls[2])
        m2_pth = os.path.join(self.im_root,ls[3])
        m1 = sorted(os.listdir(m1_pth))
        m2 = sorted(os.listdir(m2_pth))
        if len(m1)>len(m2):
            while len(m1)>len(m2):
                m2 = m2*2
        elif len(m1)<len(m2):
            while len(m1)<len(m2):
                m1 = m1*2
        ########### forced to be lenghth of 400
        # if len(m1)<400:
        #     m1 = m1*2
        # if len(m2)<400:
        #     m2 = m2*2
        ##########

        if not self.test:
            if self.sf_sq:
                random.shuffle(m2)
                random.shuffle(m1)
        ######### forced to be length of 400
        # m1 = m1[:400]
        # m2 = m2[:400]
        #########
        for it1,it2 in zip(m1,m2):
            it1_pth = os.path.join(m1_pth,it1)
            it2_pth = os.path.join(m2_pth,it2)
            paired_ls.append([ls[0],ls[1],it1_pth,it2_pth])

        return paired_ls

    def _crs_kinlis(self,kls):
        """
        cross the pos and neg item
        :param kls:
        :return:
        """
        cr_ls = [] # get the cross fold number list
        for kl in kls:
            if kl[0] not in cr_ls:
                cr_ls.append(kl[0])

        new_list = []

        def get_crs(pos, neg): # all the negative label and positive label are crossed
            n_ls = []
            for a, b in zip(pos, neg):
                n_ls.append(a)
                n_ls.append(b)
            return n_ls

        for fd in cr_ls:
            pos = []
            neg = []
            for kl in kls:
                if kl[0] == fd:
                    if kl[1] == 1:
                        pos.append(kl)
                    else:
                        neg.append(kl)
            new_list += get_crs(pos, neg)
        return new_list

    def _get_frm_nms(self,ls):
        """
        get the frame list from one of the training
        data path(suppose all training fold have equal frames)
        :param ls:
        :return:
        """
        m_pth = os.path.join(self.im_root, ls[2])
        m = sorted(os.listdir(m_pth))
        return m

    def _get_balanced_paired_frames(self,kls):
        n_ls = self._crs_kinlis(kls)
        frm_nms = self._get_frm_nms(kls[0])
        all_nms = {}
        for i in range(2*len(n_ls)):
            shuffle(frm_nms)
            all_nms[i] = copy.deepcopy(frm_nms)
        frame_list =[]
        for i in range(len(frm_nms)):
            for j in range(len(n_ls)):
                m1_pth = os.path.join(self.im_root, n_ls[j][2])
                m2_pth = os.path.join(self.im_root, n_ls[j][3])
                im1_nm = os.path.join(m1_pth,all_nms[2*j][i])
                im2_nm = os.path.join(m2_pth,all_nms[2*j+1 ][i])
                frame_list.append([n_ls[j][0],n_ls[j][1],im1_nm,im2_nm])

        return frame_list


    def _get_frames(self,ls):
        if self.test:
            frame_list = []
            for item in ls:
                match_fms = self._get_paired_frame_ls(item)
                frame_list = frame_list + match_fms
        else:
            frame_list = self._get_balanced_paired_frames(ls)

        return frame_list

    def _get_cross(self,cross):

        return [i for i in self.kin_list if i[0] in cross]




class NemoDataset_attention_msk(NemoDataset):
    def __init__(self,list_path,img_root, cross_vali = None,transform = None,
                 sf_sequence = False,cross_shuffle = False,sf_aln = False,test = False):
        super().__init__(list_path,img_root, cross_vali ,transform,
                 sf_sequence ,cross_shuffle ,sf_aln,test)
        self.msk_dic = {1:'_1.jpg',2:'_2.jpg',3:'_3.jpg',4:'_4.jpg',5:'_5.jpg'}

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        if not self.test:


            msk_lb = random.randint(0,5)

            if msk_lb!=0:
                # extract img1
                img1_path = self.fr_list[item][2]
                img1_path = img1_path.replace('framses_resize64', 'mask')
                img1_path = img1_path.replace('.jpg',self.msk_dic[msk_lb])
                img1 = Image.open(img1_path)

                # extract img2
                img2_path = self.fr_list[item][3]
                img2_path = img2_path.replace('framses_resize64', 'mask')
                img2_path = img2_path.replace('.jpg', self.msk_dic[msk_lb])
                img2 = Image.open(img2_path)
            else:
                # extract img1
                img1_path = self.fr_list[item][2]
                img1 = Image.open(img1_path)

                # extract img2
                img2_path = self.fr_list[item][3]
                img2 = Image.open(img2_path)


            # get kin label 0/1
            kin = self.fr_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1, img2))
            # after each epoch, shuffle once
            if self.crf:
                self.cout += 1
                if self.cout == self.flth:
                    self.cout = 0
                    self._cross_shuffle()
                    self.fr_list = self._get_frames(self.kin_list)

            return imgs, kin, msk_lb, img1_path, img2_path
        else:
            # extract img1
            img1_path = self.fr_list[item][2]
            img1 = Image.open(img1_path)

            # extract img2
            img2_path = self.fr_list[item][3]
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.fr_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = torch.cat((img1, img2))

            return imgs, kin, img1_path, img2_path


class NemoDataset_attention_multi(NemoDataset):
    def __init__(self,list_path,img_root, cross_vali = None,transform = None,
                 sf_sequence = False,cross_shuffle = False,sf_aln = False,test = False):
        super().__init__(list_path,img_root, cross_vali ,transform,
                 sf_sequence ,cross_shuffle ,sf_aln,test)
        self.msk_dic = {0:'.jpg',1:'_1.jpg',2:'_2.jpg',3:'_3.jpg',4:'_4.jpg',5:'_5.jpg'}

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        if not self.test:

            read_list = [0,1,2,3,4,5]
            random.shuffle(read_list)
            kin = []
            img_temp=[]
            for msk_lb in read_list:
                # extract img1
                img1_path = self.fr_list[item][2]
                if msk_lb != 0:
                    img1_path = img1_path.replace('framses_resize64', 'mask')
                img1_path = img1_path.replace('.jpg',self.msk_dic[msk_lb])
                img1 = Image.open(img1_path)

                # extract img2
                img2_path = self.fr_list[item][3]
                if msk_lb != 0:
                    img2_path = img2_path.replace('framses_resize64', 'mask')
                img2_path = img2_path.replace('.jpg', self.msk_dic[msk_lb])
                img2 = Image.open(img2_path)

                # get kin label 0/1
                kin = self.fr_list[item][1]
                if self.transform:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)

                imgs = torch.cat((img1, img2))

                img_temp.append(imgs)

            # after each epoch, shuffle once
            if self.crf:
                self.cout += 1
                if self.cout == self.flth:
                    self.cout = 0
                    self._cross_shuffle()
                    self.fr_list = self._get_frames(self.kin_list)

            return img_temp[0],img_temp[1],img_temp[2],img_temp[3],img_temp[4],img_temp[5],\
                   read_list[0],read_list[1],read_list[2],read_list[3],read_list[4],read_list[5],kin

        else:
            img_temp =[]
            read_list = [0,1,2,3,4,5]
            for msk_lb in [0,1,2,3,4,5]:
                # extract img1
                img1_path = self.fr_list[item][2]
                if msk_lb !=0:
                    img1_path = img1_path.replace('framses_resize64', 'mask')
                img1_path = img1_path.replace('.jpg', self.msk_dic[msk_lb])
                img1 = Image.open(img1_path)

                # extract img2
                img2_path = self.fr_list[item][3]
                if msk_lb != 0:
                    img2_path = img2_path.replace('framses_resize64', 'mask')
                img2_path = img2_path.replace('.jpg', self.msk_dic[msk_lb])
                img2 = Image.open(img2_path)

                # get kin label 0/1
                kin = self.fr_list[item][1]
                if self.transform:
                    img1 = self.transform(img1)
                    img2 = self.transform(img2)

                imgs = torch.cat((img1, img2))

                img_temp.append(imgs)

            return img_temp[0], img_temp[1], img_temp[2], img_temp[3], img_temp[4], img_temp[5], \
                   read_list[0], read_list[1], read_list[2], read_list[3], read_list[4], read_list[5],kin


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class NemoDataset_cnn_points(NemoDataset):

    def __getitem__(self, item):

        if torch.is_tensor(item):
            item = item.tolist()
        if not self.test:
            # extract img1
            img1_path = self.fr_list[item][2]
            img1 = Image.open(img1_path)
            # extract img2
            img2_path = self.fr_list[item][3]
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.fr_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            # imshow(torchvision.utils.make_grid(img1,nrow=5))
            imgs =[]
            for im1, im2 in zip(img1,img2):
                imgs.append(torch.cat((im1, im2)))

            # after each epoch, shuffle once
            if self.crf:
                self.cout += 1
                if self.cout == self.flth:
                    self.cout = 0
                    self._cross_shuffle()
                    self.fr_list = self._get_frames(self.kin_list)

            return imgs[0],imgs[1],imgs[2],imgs[3],imgs[4],imgs[5],imgs[6],imgs[7],imgs[8],imgs[9],kin
        else:
            # extract img1
            img1_path = self.fr_list[item][2]
            img1 = Image.open(img1_path)

            # extract img2
            img2_path = self.fr_list[item][3]
            img2 = Image.open(img2_path)
            # get kin label 0/1
            kin = self.fr_list[item][1]
            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            imgs = []
            for im1, im2 in zip(img1, img2):
                imgs.append(torch.cat((im1, im2)))


            return imgs[0],imgs[1],imgs[2],imgs[3],imgs[4],imgs[5],imgs[6],imgs[7],imgs[8],imgs[9],kin




if __name__=='__main__':


    train_ls = '../data/label/B-B.pkl'
    data_pth = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/framses_resize64/B-B'
    nemo_data = NemoDataset_cnn_points(train_ls,data_pth,[1,2,3,4],transform= cnn_points_train_transform,sf_sequence= True,cross_shuffle =True,sf_aln = True,
                            test = False)
    nemoloader = DataLoader(nemo_data,shuffle=True)
    for j in range(3):
        for i,data in enumerate(nemoloader):
            # print(i)
            pass
        print(i)