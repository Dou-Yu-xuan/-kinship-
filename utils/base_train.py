from utils.loader import cross_validation
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime,date
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from config import  attention_Net_config



class Base_train(object):
    """
    basic training class
    """

    def __init__(self, config):
        ## init param
        self.kintype = config.kintype
        self.epoch_num = config.epoch_num
        self.list_path = config.list_path
        self.img_root = config.img_root
        self.show_lstep = config.show_lstep
        self.batch = config.train_batch
        self.model = config.model
        self.lr = config.lr
        self.lr_de = config.lr_decay
        self.mom = config.momentum
        self.lr_mil = config.lr_milestones
        self.sf_sq = config.sf_sequence
        self.model_name = config.model_name
        self.cs = config.cross_shuffle
        self.cr_num = config.cross_num
        self.print_frq = config.prt_fr
        self.prt_fr_train = config.prt_fr_train
        self.sf_aln = config.sf_aln
        self.reload = config.reload
        self.save_ck = config.save_ck
        self.we_de = config.weight_decay
        self.logs_name = config.logs_name
        self.Dataset = config.Dataset
        self.savemis = config.savemis
        self.test_batch = config.test_batch
        self.save_tacc = config.save_tacc
        self.loss = config.loss
        self.optim = config.optimal
        self.dl_sf = config.dataloder_shuffle
        self.description = config.__dict__
        ##transform
        if config.trans is not None:
            self.train_transform = config.trans[0]
            self.test_transform = config.trans[1]

        else:
            self.train_transform = config.train_transform
            self.test_transform = config.test_transform
        ## add save model params
        if config.save_ck:
            self.ck_pth = 'data/checkpoints/{}-{}'.format(config.model_name, config.kintype)
            if not os.path.isdir(self.ck_pth):
                os.makedirs(self.ck_pth)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if config.save_graph:
            net = getattr(config.model, config.model_name)()
            net.to(self.device)
            graph_in = torch.randn(2, config.imsize[0], config.imsize[1], config.imsize[2]).to(self.device)
            with SummaryWriter('{}/{}/graph'.format(self.logs_name, self.model_name)) as ww:
                ww.add_graph(net, graph_in)
                ww.add_text('{}_parameters'.format(self.model_name), net.__repr__())
                ww.add_text('Training Description', config.description)
                ww.add_text('Training parameters',
                            'Training parameters:{}'.format(self.__dict__))  # loss and optimizer
        self.test_acc = 0


    def cross_run(self):
        """
        run cross validation and save
        :return:
        """
        with SummaryWriter('{}/{}-{}/{}_kv_cross_description'.format(self.logs_name, self.model_name, self.kintype,
                                                                     datetime.now())) as ww:
            ww.add_text('{}_{}_parameters'.format(self.logs_name, self.model_name),
                        '{}'.format(self.description).replace(',', '<br />'))

        total_acc = 0
        for tra_id, tes_id in cross_validation(self.cr_num):
            print('#' * 10 + 'cross validation{}'.format(self.cr_num + 1 - tes_id[0]) + '#' * 10)
            self.run(tra_id, tes_id, self.cr_num + 1 - tes_id[0])
            acc = self.test_acc
            total_acc += acc

        with SummaryWriter('{}/{}-{}/{}_kv_cross_vali_final'.format(self.logs_name, self.model_name, self.kintype,
                                                                    datetime.now())) as ww:
            ww.add_scalar(tag='final test accuracy',
                          scalar_value=total_acc / self.cr_num)
            print('the accuracy after cross validation is {}'.format(total_acc / self.cr_num))

    def run(self, train_id=None, test_id=None, w_num=1):
        """
        :param train_id: cross validation split list [1,2,3,4]
        :param test_id: [5]
        :param w_num: the number of validation 6-test_id
        :return:
        """

        self.writer = SummaryWriter(
            '{}/{}-{}/{}_kv_cross_vali0{}'.format(self.logs_name, self.model_name, self.kintype, datetime.now(),
                                                  w_num))
        if train_id is None:
            train_id = [1, 2, 3, 4]
            test_id = [5]

        train_set = self.Dataset(self.list_path, self.img_root, cross_vali = train_id, transform=self.train_transform,
                                 sf_sequence=self.sf_sq, cross_shuffle=self.cs, sf_aln=self.sf_aln)
        test_set = self.Dataset(self.list_path, self.img_root, cross_vali= test_id, transform=self.test_transform, test=True)
        train_loader = DataLoader(train_set, batch_size=self.batch, shuffle=self.dl_sf)
        test_loader = DataLoader(test_set, batch_size=self.test_batch)
        # final_loader = DataLoader(test_set, batch_size=1)
        self.train(train_loader, test_loader, w_num)
        self.writer.close()



    def objects_fun(self):
        self.net = getattr(self.model, self.model_name)()
        self.net.to(self.device)
        self.criterion = self.loss()
        if self.optim == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.mom, weight_decay=self.we_de)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.we_de)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.lr_mil, gamma=self.lr_de)
        # return net, criterion, optimizer, scheduler

    def each_train(self,data):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, _, _ = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self.net(inputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, train_loader, test_loader, w_num):
        ### get model

        self.objects_fun()

        ### add model reload
        if self.reload != '':
            checkpoints = torch.load(self.reload)
            self.net.load_state_dict(checkpoints['arch'])

        global_step = 0
        # self.net.train()
        epoch = 0
        for epoch in range(self.epoch_num):  # loop over the dataset multiple times

            print('epoch: {}'.format(epoch))
            running_loss = 0.0

            self.net.train()
            for i, data in enumerate(train_loader, 0):
                #  train each data
                loss = self.each_train(data)
                # print statistics
                running_loss += loss.item()
                # running_loss = loss.item()

                if i % self.show_lstep == (self.show_lstep - 1):
                    # print loss
                    print('[epoch %d, global step %5d] loss: %.3f' %
                          (epoch + 1, global_step + 1, running_loss / self.show_lstep))

                    # ...log the running loss
                    self.writer.add_scalar(tag='training loss',
                                           scalar_value=running_loss / self.show_lstep,
                                           global_step=global_step)

                    running_loss = 0.0

                # update global step
                global_step += 1
            if self.save_tacc:
                if (epoch + 1) % self.prt_fr_train == 0:
                    self.acc_records( train_loader, epoch, t='train')
            if (epoch + 1) % self.print_frq == 0:
                self.acc_records(test_loader, epoch, t='test')

            # update learning rate
            self.scheduler.step()
        self.mis_record( test_loader, w_num, self.savemis)
        # self.acc_records(net, test_loader, epoch, t='test')
        ## save model
        if self.save_ck:
            torch.save({
                'epoch': epoch,
                'arch': self.net.state_dict(),
                'optimizer_state': self.optimizer.state_dict()
            }, '{}/{}-{}-cv{}-{}-{}.pth'.format(self.ck_pth,
                                                self.model_name, self.kintype, w_num, date.today(),
                                                datetime.now().hour))


    def each_test(self,dloader,t = 'train'):
        correct = 0
        total = 0
        self.net.eval()
        with torch.no_grad():
            for data in dloader:
                images, labels, _, _ = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return total,correct

    def acc_records(self, dloader, epoch, t='train'):
        """
        :param dloader: train loader or test loader
        :param epoch: training epochs
        :param t: 'train' or 'test'
        :return: record accuracy
        """

        total,correct = self.each_test(dloader,t)

        print('Accuracy of the network on the %s images: %d %%' % (t,
                                                                   100 * correct / total))
        acc = correct / total
        self.writer.add_scalar(tag='{}_accuracy/epoch'.format(t),
                               scalar_value=acc,
                               global_step=epoch)




    def mis_record(self,  dloader, w_num, savemis):
        """
        mismatch record
        :param dloader: testloader
        :param w_num: writer cross valid number: 0-5
        :return:
        """
        total,correct = self.each_test(dloader)

        #  add writer for misclassified pairs
        if savemis:
            pass
            #        TODO: check the saving function
            # for i, val in enumerate(predicted == labels):
            #     if not val:
            #         # probs = [F.softmax(el, dim=0)[i].item() for i, el in zip(predicted, outputs)]
            #         probs = [it.item() for it in F.softmax(outputs[i], dim=0)]
            #         img1, img2 = images[i:i + 1][:, 0:3, :, :], images[i:i + 1][:, 3:6, :, :]
            #         img1 = img1 / 2 + 0.5
            #         img2 = img2 / 2 + 0.5
            #         imgs = torch.cat((img1, img2))
            #         if predicted[i] == 1:
            #             self.writer.add_images(tag='{}/false_positive_cross_0{}_{}/{}probs:({:.4},{:.4})'.
            #                                    format(self.model_name, w_num, timg_1[i], timg_2[i],
            #                                           probs[0], probs[1]),
            #                                    img_tensor=imgs)
            #         else:
            #             self.writer.add_images(tag='{}/false_negative_cross_0{}_{}/{}probs:({:.4},{:.4})'.
            #                                    format(self.model_name, w_num, timg_1[i], timg_2[i],
            #                                           probs[0], probs[1]),
            #                                    img_tensor=imgs)
        self.test_acc = correct / total

    def test(self):
        pass


if __name__=='__main__':


    attention_Net_config.Config.kintype = 'B-B'
    ## list path

    attention_Net_config.Config.list_path = '../data/label/B-B.pkl'
    ## data path
    attention_Net_config.Config.img_root = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/framses_resize64/B-B'
    netmodel = Base_train(attention_Net_config.Config)

    netmodel.cross_run()
