from utils.base_train import Base_train
from config import  attention_Net_config
import argparse
import torch


class mask_train(Base_train):
    # def __init__(self,config):
    #     super(mask_train, self).__init__(config)

    def each_train(self, data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels,mask_label, _, _  = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)


        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        kin,le,re,ns,lm,rm = self.net(inputs)

        kin_loss = self.criterion(kin, labels)
        le_lb = (mask_label ==1).long().to(self.device)
        le_loss = self.criterion(le, le_lb)
        re_lb = (mask_label==2).long().to(self.device)
        re_loss = self.criterion(re,re_lb)
        ns_lb = (mask_label==3).long().to(self.device)
        ns_loss = self.criterion(ns,ns_lb)
        lm_lb = (mask_label==4).long().to(self.device)
        lm_loss = self.criterion(lm,lm_lb)
        rm_lb = (mask_label==5).long().to(self.device)
        rm_loss = self.criterion(rm,rm_lb)


        loss = 5*kin_loss + le_loss+re_loss+ns_loss+lm_loss+rm_loss
        loss.backward()
        self.optimizer.step()

        return loss

    def each_test(self,dloader,t='test'):
        if t=='test':
            correct = 0
            total = 0
            self.net.eval()
            with torch.no_grad():
                for data in dloader:
                    images, labels, _, _ = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs,le,re,ns,lm,rm = self.net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            return total,correct
        else:
            correct = 0
            total = 0
            self.net.eval()
            with torch.no_grad():
                for data in dloader:
                    images, labels,mask_label, _, _ = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs, le, re, ns, lm, rm = self.net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            return total, correct


class mask_multi_train(Base_train):

    def each_train(self, data):
        # get the inputs; data is a list of [inputs, labels]
        dataout = data

        for dit in range(13):
            dataout[dit] = dataout[dit].to(self.device)

        x0 = dataout[0]
        x1 = dataout[1]
        x2 = dataout[2]
        x3 = dataout[3]
        x4 = dataout[4]
        x5 = dataout[5]

        imlk_0 = dataout[6]
        imlk_1 = dataout[7]
        imlk_2 = dataout[8]
        imlk_3 = dataout[9]
        imlk_4 = dataout[10]
        imlk_5 = dataout[11]

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # # forward + backward + optimize

        kin, prmk_0,prmk_1,prmk_2,prmk_3,prmk_4,prmk_5 = self.net(x0,x1,x2,x3,x4,x5)
        #
        lmk_loss = 0
        kin_loss = self.criterion(kin, dataout[-1])
        for pr_mk_per,lm_label in zip([prmk_0,prmk_1,prmk_2,prmk_3,prmk_4,prmk_5],
                                      [imlk_0,imlk_1,imlk_2,imlk_3,imlk_4,imlk_5]):

            le_lb = (lm_label == 1).long().to(self.device)
            le_loss = self.criterion(pr_mk_per[0], le_lb)
            re_lb = (lm_label == 2).long().to(self.device)
            re_loss = self.criterion(pr_mk_per[1], re_lb)
            ns_lb = (lm_label == 3).long().to(self.device)
            ns_loss = self.criterion(pr_mk_per[2], ns_lb)
            lm_lb = (lm_label == 4).long().to(self.device)
            lm_loss = self.criterion(pr_mk_per[3], lm_lb)
            rm_lb = (lm_label == 5).long().to(self.device)
            rm_loss = self.criterion(pr_mk_per[4], rm_lb)
            lmk_loss += (le_loss+re_loss+ns_loss+lm_loss+rm_loss)

        loss = 50*kin_loss+lmk_loss

        loss.backward()
        self.optimizer.step()
        #
        return loss

    def each_test(self, dloader, t='test'):

        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            for data in dloader:
                dataout = data

                for dit in range(13):
                    dataout[dit] = dataout[dit].to(self.device)

                x0 = dataout[0]
                x1 = dataout[1]
                x2 = dataout[2]
                x3 = dataout[3]
                x4 = dataout[4]
                x5 = dataout[5]

                labels = dataout[12]

                outputs, prmk_0, prmk_1, prmk_2, prmk_3, prmk_4, prmk_5 = self.net(x0, x1, x2, x3, x4, x5)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total, correct

if __name__=='__main__':

    ###### training attention-only network
    ##################################################################

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
    #     attention_Net_config.Config.kintype = kin_type
    #     ## list path
    #
    #     attention_Net_config.Config.list_path = './data/label/{}.pkl'.format(kin_type)
    #     ## data path
    #     attention_Net_config.Config.img_root = '../../../DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(kin_type)
    #     netmodel = Base_train(attention_Net_config.Config)
    #
    #     netmodel.cross_run()

    ##### training mask+attention network
    ###################################################################
    #
    # parser = argparse.ArgumentParser(description='PyTorch kinship verification')
    # parser.add_argument('--type', type=str,
    #                     default=None,
    #                     help='specific type of training data')
    #
    # args = parser.parse_args()
    #
    # train_ls = ['F-D', 'F-S', 'M-D', 'M-S', 'B-B', 'S-S', 'B-S']
    # if args.type:
    #     print("#" * 20, "Starting training {}".format(args.type), "#" * 20)
    #     train_ls = [str(args.type)]
    #
    # for kin_type in train_ls:
    #     attention_Net_config.Mask.kintype = kin_type
    #     ## list path
    #
    #     attention_Net_config.Mask.list_path = './data/label/{}.pkl'.format(kin_type)
    #     ## data path
    #     attention_Net_config.Mask.img_root = '../../../DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(
    #         kin_type)
    #     netmodel = mask_train(attention_Net_config.Mask)
    #
    #     netmodel.cross_run()

    ##### training mask+attention MULTI network
    ###################################################################
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

        attention_Net_config.mask_multi.kintype = kin_type
        ## list path

        attention_Net_config.mask_multi.list_path = './data/label/{}.pkl'.format(kin_type)
        ## data path
        attention_Net_config.mask_multi.img_root = '../../../DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(kin_type)
        netmodel = mask_multi_train(attention_Net_config.mask_multi)

        netmodel.cross_run()
