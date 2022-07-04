import numpy as np
from skimage.feature import local_binary_pattern,hog
from PIL import Image
import torch
from matplotlib import pyplot as plt
import cv2
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve,roc_auc_score
import  os
from tqdm import tqdm
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.preprocessing import image as keras_image
from torchvision import transforms
from facenet_pytorch import  InceptionResnetV1,fixed_image_standardization
from PIL import Image

def get_pixel(img, center, x, y):
    new_value = 0

    try:
        # If local neighbourhood pixel
        # value is greater than or equal
        # to center pixel values then
        # set it to 1
        if img[x][y] >= center:
            new_value = 1

    except:
        # Exception is required when
        # neighbourhood value of a center
        # pixel value is null i.e. values
        # present at boundaries.
        pass

    return new_value


# Function for calculating LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x][y]

    val_ar = []

    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))

    # top
    val_ar.append(get_pixel(img, center, x - 1, y))

    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))

    # right
    val_ar.append(get_pixel(img, center, x, y + 1))

    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))

    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))

    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))

    # left
    val_ar.append(get_pixel(img, center, x, y - 1))

    # Now, we need to convert binary
    # values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]

    val = 0

    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]

    return val


def LBP_EXTRACTOR(img):
    """
    ### https://www.geeksforgeeks.org/create-local-binary-pattern-of-an-image-using-opencv-python/

    :param img:
    :return:
    """
    # settings for LBP
    # radius = 2
    # n_points = 8 * radius
    #
    # features =local_binary_pattern(img, n_points, radius, 'uniform')
    img_bgr = img

    height, width, _ = img_bgr.shape

    # We need to convert RGB image
    # into gray one because gray
    # image has one channel only.
    img_gray = cv2.cvtColor(img_bgr,
                            cv2.COLOR_BGR2GRAY)

    n_batch = 4
    height_batch = int(height / n_batch)
    width_batch = int(width / n_batch)

    get_batches = lambda x, i, j: x[i * height_batch:(i + 1) * height_batch, j * width_batch:(1 + j) * width_batch]
    img_lbp = []
    # Create a numpy array as
    # the same height and width
    # of RGB image
    for temp_h in range(4):
        for temp_w in range(4):
            temp_lbp = np.zeros((height_batch, width_batch),
                                np.uint8)
            for i in range(0, width_batch):
                for j in range(0, width_batch):
                    temp_lbp[i, j] = lbp_calculated_pixel(get_batches(img_gray, temp_h, temp_w), i, j)

            # n_bins = int(img_lbp.max() + 1)
            hist, _ = np.histogram(temp_lbp, bins=256, range=(0, 256), density=True)
            img_lbp = np.concatenate((img_lbp, hist))

    features = img_lbp
    return features


def SIFT_EXTRACTOR(img,color='GRAY'):
    try:
        sift = cv2.xfeatures2d.SIFT_create()
    except:
        sift = cv2.SIFT_create()
    if color == 'GRAY':
        temp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(temp, None)
        return des
    else:
        kp, des1 = sift.detectAndCompute(img[:, :, 0], None)
        kp, des2 = sift.detectAndCompute(img[:, :, 0], None)
        kp, des3 = sift.detectAndCompute(img[:, :, 0], None)
        des = np.concatenate((des1, des2, des3))
        return des


def HOG_EXTRACTOR(img):
    fea = hog(img, orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2))
    return fea

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


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class tensor_dim(object):
    def __init__(self):
        pass
    def __call__(self, sample):
        return torch.unsqueeze(sample,0)

trans = transforms.Compose([transforms.Resize(160),
                            np.float32,
                            transforms.ToTensor(),
                            tensor_dim(),
                            fixed_image_standardization])

class FaceNet_EXTRACTOR():
    def __init__(self,prtrain = 'casia-webface'):
        self.model = InceptionResnetV1(pretrained=prtrain).eval().to(device)


    def load_model(self,pth):
        return

    def feature_extract(self,img):
        return self.model(img).detach().cpu()

Facenet = FaceNet_EXTRACTOR()

def mDML(img):
    features = []
    return features


def read_pkl(nemo_ls,im_pth):
    with open(nemo_ls, 'rb') as fp:
        ori_ls = pickle.load(fp)
    new_list = []
    for temp in ori_ls:
        for i in range(10):
            im1_pth = os.path.join(im_pth, temp[2]) + '/frame{:03d}.jpg'.format(i)
            im2_pth = os.path.join(im_pth, temp[3]) + '/frame{:03d}.jpg'.format(i)
            new_list.append([temp[0],temp[1],im1_pth,im2_pth])
    return ori_ls,new_list



def img_blocks(img,b_size,stride):
    img_blks = []
    w,h = img.shape[0],img.shape[1]
    bx_w, bx_y = b_size[0], b_size[1]
    times_w = int((w-b_size[0])/stride+1)
    times_y = int((h-b_size[0])/stride+1)
    for i in range(times_w):
        for j in range(times_y):
            img_blks.append(img[(stride)*i:(stride)*i+bx_w,(stride)*j:stride*j+bx_y])
    return img_blks


def Feature_extractor(img, b_size=[16,16], stride=8, tp='LBP'):

    if tp == 'LBP':
        img = cv2.imread(img)
        features = LBP_EXTRACTOR(img)
        return  features

    elif tp == 'SIFT':
        img = cv2.imread(img)
        im_blks = img_blocks(img, b_size, stride)
        features = np.array([])
        for im in im_blks:
            fea = SIFT_EXTRACTOR(im)
            if fea is None:
                fea = np.zeros(128)
            else:
                fea = np.mean(fea, axis=0)
            features = np.concatenate((features, fea))
        return features

    elif tp =='HOG':
        img = cv2.imread(img)
        features = HOG_EXTRACTOR(img)
        return features

    elif tp == 'vggface':
        img = keras_image.load_img(img, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)
        features = VGGFACE.feature_extract(x)[0]
        return  features
    elif tp =='facenet':
        img = Image.open(img)
        img = trans(img).to(device)
        features = Facenet.feature_extract(img)[0].numpy()
        return features



def similarity_scores(A,B):
    # return  A.dot(B)/ (np.linalg.norm(A, axis=1) * np.linalg.norm(B))
    return  cosine_similarity(np.expand_dims(A, axis=0), np.expand_dims(B, axis=0))[0][0]



def save_score(files,name):
    with open('{}.pickle'.format(name),'wb') as ff:
        pickle.dump(files,ff)


def read_score(name):
    with open('{}.pickle'.format(name), 'rb') as handle:
        b = pickle.load(handle)
    return b


if __name__ =='__main__':

    train_ls = ['F-D', 'F-S', 'M-D', 'M-S', 'B-B', 'S-S', 'B-S']
    tp = ['LBP','SIFT','HOG','vggface','facenet']


    for kin_type in train_ls:

        lb_pth = '../data/label/{}.pkl'.format(kin_type)
        img_pth = '/home/wei/Documents/DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(kin_type)

        nemo_ls,data_ls = read_pkl(lb_pth,img_pth)

        score_ls = {}
        for tp_item in tp:
            score_ls[tp_item] = []

        label_ls = []

        for pair_temp in tqdm(data_ls):
            img1 = pair_temp[2]
            img2 = pair_temp[3]
            label_ls.append(pair_temp[1])

            for tp_item in tp:
                fea_img1 = Feature_extractor(img1,tp=tp_item)
                fea_img2 = Feature_extractor(img2,tp=tp_item)
                scores = similarity_scores(fea_img1,fea_img2)
                score_ls[tp_item].append(scores)

        save = False
        if save:
            save_score(label_ls, 'label')
            for tp_item in tp:
                save_score(score_ls[tp_item],tp_item)

        ld_save= False
        if ld_save:
            score_ls ={}
            for tp_item in tp:
                score_ls[tp_item]= read_score(tp_item)


        plt.figure()

        for tp_item in tp:
            fpr, tpr, _ = roc_curve(label_ls, score_ls[tp_item])
            auc = roc_auc_score(label_ls, score_ls[tp_item])
            plt.plot(fpr, tpr,  lw=2, label='{} (AUC: {:0.2f})'.format(tp_item,auc))

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC:0.50)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of {}'.format(kin_type))
        plt.legend(loc="best")
        # plt.show()
        plt.savefig('{}.png'.format(kin_type))











