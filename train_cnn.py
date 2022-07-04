from utils.base_train import Base_train
from config import  CNN_config
import argparse
import torch


class cnn_points_train(Base_train):
    # def __init__(self,config):
    #     super(mask_train, self).__init__(config)

    def each_train(self, data):
        # get the inputs; data is a list of [inputs, labels]
        x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,labels = data


        x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,labels = x1.to(self.device),x2.to(self.device),x3.to(self.device),\
                                                x4.to(self.device),x5.to(self.device),x6.to(self.device),\
                                                x7.to(self.device),x8.to(self.device),x9.to(self.device), \
                                                x10.to(self.device),labels.to(self.device)


        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        kin= self.net(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10)

        loss = self.criterion(kin, labels)

        loss.backward()
        self.optimizer.step()

        return loss

    def each_test(self,dloader,t='test'):

        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            for data in dloader:
                x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, labels = data

                x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, labels = x1.to(self.device), x2.to(self.device), x3.to(self.device),\
                                                                  x4.to(self.device), x5.to(self.device), x6.to(self.device), \
                                                                  x7.to(self.device), x8.to(self.device), x9.to(self.device), \
                                                                  x10.to(self.device), labels.to(self.device)
                outputs = self.net(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total,correct




if __name__=='__main__':

    ###### training cnn_basic network
    ##################################################################


    parser = argparse.ArgumentParser(description='PyTorch kinship verification')
    parser.add_argument('--type', type=str,
                        default=None,
                        help='specific type of training data')

    args = parser.parse_args()

    train_ls = ['F-D','F-S','M-D','M-S','B-B','S-S','B-S']
    if args.type:
        print("#"*20,"Starting training {}".format(args.type),"#"*20)
        train_ls = [str(args.type)]

    for kin_type in train_ls:

        CNN_config.basic_cnn.kintype = kin_type
        ## list path

        CNN_config.basic_cnn.list_path = './data/label/{}.pkl'.format(kin_type)
        ## data path
        CNN_config.basic_cnn.img_root = '../../../DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(kin_type)
        netmodel = Base_train(CNN_config.basic_cnn)

        netmodel.cross_run()

    ##### training

    # parser = argparse.ArgumentParser(description='PyTorch kinship verification')
    # parser.add_argument('--type', type=str,
    #                     default=None,
    #                     help='specific type of training data')
    #
    # args = parser.parse_args()
    #
    # train_ls = ['F-D','F-S','M-D','M-S','B-B','S-S','B-S']
    # if args.type:
    #     print("#"*20,"Starting training {}".format(args.type),"#"*20)
    #     train_ls = [str(args.type)]
    #
    # for kin_type in train_ls:
    #
    #     CNN_config.cnn_points.kintype = kin_type
    #     ## list path
    #
    #     CNN_config.cnn_points.list_path = './data/label/{}.pkl'.format(kin_type)
    #     ## data path
    #     CNN_config.cnn_points.img_root = '../../../DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(kin_type)
    #     netmodel = cnn_points_train(CNN_config.cnn_points)
    #
    #     netmodel.cross_run()


