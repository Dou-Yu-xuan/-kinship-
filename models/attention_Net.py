import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
try:
    from torchviz import make_dot
except:
    pass


class res_unit(nn.Module):
    """
    this is the attention module before Residual structure
    """
    def __init__(self,channel,up_size = None):
        """

        :param channel: channels of input feature map
        :param up_size: upsample size
        """
        super(res_unit,self).__init__()
        self.pool = nn.MaxPool2d(2,2)
        self.conv = nn.Conv2d(channel,channel,3,padding=1)
        if up_size == None:
            self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        else:
            self.upsample = nn.Upsample(size=(up_size,up_size), mode='bilinear', align_corners=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        identity = x
        x = self.pool(x)
        x = self.conv(x)
        x = self.upsample(x)
        x = self.sigmoid(x)
        x = torch.mul(identity,x)
        return x


class attenNet(nn.Module):
    """
    the attention Module in <Learning part-aware attention networks for kinship verification>
    """
    def __init__(self):
        super(attenNet,self).__init__()
        self.conv1 = nn.Conv2d(6,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.at1 = res_unit(32)
        self.at2 = res_unit(64)
        self.at3 = res_unit(128,up_size=9)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear((9*9*128),512)
        # self.dp  = nn.Dropout()
        self.fc2 = nn.Linear(512,2)

    def forward(self,x):
        """
        :param x: 6x64x64
        :return:
        """
        x = self.conv1(x)
        identity1 = x
        x = self.at1(x)
        x = identity1+x
        x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        identity2 = x
        x = self.at2(x)
        x = identity2 + x
        x = self.bn2(x)
        x = self.pool(F.relu((x)))
        x = self.conv3(x)
        identity3 = x
        x = self.at3(x)
        x = identity3 + x
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(-1, 9*9*128)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        # x = self.dp(x)
        x = self.fc2(x)
        return x



class attenNet_mask(nn.Module):
    """
    the attention Module in <Learning part-aware attention networks for kinship verification>
    """
    def __init__(self):
        super(attenNet_mask,self).__init__()
        self.conv1 = nn.Conv2d(6,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.at1 = res_unit(32)
        self.at2 = res_unit(64)
        self.at3 = res_unit(128,up_size=9)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear((9*9*128),512)
        # self.dp  = nn.Dropout()
        # kin
        self.fc2 = nn.Linear(512,2)
        # left eye
        self.le = nn.Linear(512,2)
        # right eye
        self.re = nn.Linear(512,2)
        # nose
        self.ns = nn.Linear(512,2)
        # lm
        self.lm = nn.Linear(512,2)
        # rm
        self.rm = nn.Linear(512,2)

    def forward(self,x):
        """
        :param x: 6x64x64
        :return:
        """
        x = self.conv1(x)
        identity1 = x
        x = self.at1(x)
        x = identity1+x
        x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        identity2 = x
        x = self.at2(x)
        x = identity2 + x
        x = self.bn2(x)
        x = self.pool(F.relu((x)))
        x = self.conv3(x)
        identity3 = x
        x = self.at3(x)
        x = identity3 + x
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(-1, 9*9*128)
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        # x = self.dp(x)
        kin = self.fc2(x)
        le = self.le(x)
        re = self.re(x)
        ns = self.ns(x)
        lm = self.lm(x)
        rm = self.rm(x)

        return kin,le,re,ns,lm,rm


class attenNet_feat(nn.Module):
    """
    the attention Module in <Learning part-aware attention networks for kinship verification>
    """
    def __init__(self):
        super(attenNet_feat,self).__init__()
        self.conv1 = nn.Conv2d(6,32,5)
        self.conv2 = nn.Conv2d(32,64,5)
        self.conv3 = nn.Conv2d(64,128,5)
        self.at1 = res_unit(32)
        self.at2 = res_unit(64)
        self.at3 = res_unit(128,up_size=9)
        self.pool = nn.MaxPool2d(2,2)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)


    def forward(self,x):
        """
        :param x: 6x64x64
        :return:
        """
        x = self.conv1(x)
        identity1 = x
        x = self.at1(x)
        x = identity1+x
        x = self.bn1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        identity2 = x
        x = self.at2(x)
        x = identity2 + x
        x = self.bn2(x)
        x = self.pool(F.relu((x)))
        x = self.conv3(x)
        identity3 = x
        x = self.at3(x)
        x = identity3 + x
        x = self.bn3(x)
        x = F.relu(x)
        x = x.view(-1, 9*9*128)
        return x



class attenNet_multi(nn.Module):
    """
    the attention Module in <Learning part-aware attention networks for kinship verification>
    """
    def __init__(self):
        super(attenNet_multi,self).__init__()
        self.feat  = attenNet_feat()
        self.fc1_0 = nn.Linear((9*9*128),512)
        self.fc1_1 = nn.Linear((9*9*128),512)
        self.fc1_2 = nn.Linear((9*9*128),512)
        self.fc1_3 = nn.Linear((9*9*128),512)
        self.fc1_4 = nn.Linear((9*9*128),512)
        self.fc1_5 = nn.Linear((9*9*128),512)

        # no mask
        self.no = nn.Linear(512,2)
        # left eye
        self.le = nn.Linear(512,2)
        # right eye
        self.re = nn.Linear(512,2)
        # nose
        self.ns = nn.Linear(512,2)
        # lm
        self.lm = nn.Linear(512,2)
        # rm
        self.rm = nn.Linear(512,2)
        # kin
        self.kin = nn.Linear(512*6,2)


    def forward(self,x_0,x_1,x_2,x_3,x_4,x_5):
        """
        :param x: 6x64x64
        :return:
        """
        x_0 = self.feat(x_0)
        x_0 = self.fc1_0(x_0)

        x_1 = self.feat(x_1)
        x_1 = self.fc1_1(x_1)

        x_2 = self.feat(x_2)
        x_2 = self.fc1_2(x_2)

        x_3 = self.feat(x_3)
        x_3 = self.fc1_3(x_3)

        x_4 = self.feat(x_4)
        x_4 = self.fc1_4(x_4)

        x_5 = self.feat(x_5)
        x_5 = self.fc1_5(x_5)

        kin = torch.cat((x_0,x_1,x_2,x_3,x_4,x_5),dim=1)
        kin = self.kin(kin)


        le_0 = self.le(x_0)
        re_0 = self.re(x_0)
        ns_0 = self.ns(x_0)
        lm_0 = self.lm(x_0)
        rm_0 = self.rm(x_0)
        ldmk_0 = [le_0,re_0,ns_0,lm_0,rm_0]

        le_1 = self.le(x_1)
        re_1 = self.re(x_1)
        ns_1 = self.ns(x_1)
        lm_1 = self.lm(x_1)
        rm_1 = self.rm(x_1)
        ldmk_1 = [le_1,re_1,ns_1,lm_1,rm_1]

        le_2 = self.le(x_2)
        re_2 = self.re(x_2)
        ns_2 = self.ns(x_2)
        lm_2 = self.lm(x_2)
        rm_2 = self.rm(x_2)
        ldmk_2 = [le_2,re_2,ns_2,lm_2,rm_2]


        le_3 = self.le(x_3)
        re_3 = self.re(x_3)
        ns_3 = self.ns(x_3)
        lm_3 = self.lm(x_3)
        rm_3 = self.rm(x_3)
        ldmk_3 = [le_3,re_3,ns_3,lm_3,rm_3]


        le_4 = self.le(x_4)
        re_4 = self.re(x_4)
        ns_4 = self.ns(x_4)
        lm_4 = self.lm(x_4)
        rm_4 = self.rm(x_4)
        ldmk_4 = [le_4,re_4,ns_4,lm_4,rm_4]


        le_5 = self.le(x_5)
        re_5 = self.re(x_5)
        ns_5 = self.ns(x_5)
        lm_5 = self.lm(x_5)
        rm_5 = self.rm(x_5)
        ldmk_5 = [le_5,re_5,ns_5,lm_5,rm_5]

        return kin,ldmk_0,ldmk_1,ldmk_2,ldmk_3,ldmk_4,ldmk_5




if __name__=='__main__':


    inputs = torch.randn(1,6,64,64)
    net = attenNet()
    y = net(inputs)
    print(y)

    g = make_dot(y)
    g.view('attention_Net_graph')


