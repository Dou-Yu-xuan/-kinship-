from mtcnn import MTCNN
import cv2
import os
import numpy as np
from tqdm import  tqdm

def add_mask(image,position,pth):
    temp = np.copy(image)
    cv2.rectangle(temp,
                  (position[0]-5, position[1]-5),(position[0]+5, position[1]+5),
                  (255,255,255),-1)
    cv2.imwrite(pth, cv2.cvtColor(temp, cv2.COLOR_RGB2BGR))
    return temp

type_ls = ['F-D','F-S','M-D','M-S','S-S','B-S','B-B']
for ktype in type_ls:
    pth =  '../../../../DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(ktype)

    target = '../../../../DATA/kinship/Nemo/kin_simple/framses_resize64/{}'.format(ktype)


    folder_ls = sorted(os.listdir(pth))
    detector = MTCNN()
    tem_1,tem_2,tem_3,tem_4,tem_5 = [],[],[],[],[]

    for fd in tqdm(folder_ls):
        fd_pth = os.path.join(pth,fd)
        tar_pth = fd_pth.replace('framses_resize64','mask')
        if not os.path.exists(tar_pth):
            os.makedirs(tar_pth)
        for img in sorted(os.listdir(fd_pth)):
            im_pth = os.path.join(fd_pth,img)
            img_temp = cv2.cvtColor(cv2.imread(im_pth), cv2.COLOR_BGR2RGB)
            result = detector.detect_faces(img_temp)

            if len(result)<1:
                print(im_pth)
                svpth = os.path.join(tar_pth, img.replace('.jpg', '_1.jpg'))
                cv2.imwrite(svpth, cv2.cvtColor(tem_1, cv2.COLOR_RGB2BGR))

                svpth = os.path.join(tar_pth, img.replace('.jpg', '_2.jpg'))
                cv2.imwrite(svpth, cv2.cvtColor(tem_2, cv2.COLOR_RGB2BGR))

                svpth = os.path.join(tar_pth, img.replace('.jpg', '_3.jpg'))
                cv2.imwrite(svpth, cv2.cvtColor(tem_3, cv2.COLOR_RGB2BGR))

                svpth = os.path.join(tar_pth, img.replace('.jpg', '_4.jpg'))
                cv2.imwrite(svpth, cv2.cvtColor(tem_4, cv2.COLOR_RGB2BGR))

                svpth = os.path.join(tar_pth, img.replace('.jpg', '_5.jpg'))
                cv2.imwrite(svpth, cv2.cvtColor(tem_5, cv2.COLOR_RGB2BGR))

                continue


            keypoints = result[0]['keypoints']
            left_eye_p = keypoints['left_eye']
            right_eye_p = keypoints['right_eye']
            nose_p = keypoints['nose']
            mouth_left_p = keypoints['mouth_left']
            mouth_right_p = keypoints['mouth_right']


            svpth = os.path.join(tar_pth,img.replace('.jpg','_1.jpg'))
            tem_1 = add_mask(img_temp, left_eye_p, svpth)

            svpth = os.path.join(tar_pth, img.replace('.jpg', '_2.jpg'))
            tem_2 = add_mask(img_temp, right_eye_p, svpth)

            svpth = os.path.join(tar_pth, img.replace('.jpg', '_3.jpg'))
            tem_3 = add_mask(img_temp, nose_p, svpth)

            svpth = os.path.join(tar_pth, img.replace('.jpg', '_4.jpg'))
            tem_4 = add_mask(img_temp, mouth_left_p, svpth)

            svpth = os.path.join(tar_pth, img.replace('.jpg', '_5.jpg'))
            tem_5 = add_mask(img_temp, mouth_right_p, svpth)




# detector = MTCNN()
#
# image = cv2.cvtColor(cv2.imread("frame000.jpg"), cv2.COLOR_BGR2RGB)
# result = detector.detect_faces(image)
#
# # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
# bounding_box = result[0]['box']
# keypoints = result[0]['keypoints']
#
# left_eye_p = keypoints['left_eye']
# right_eye_p = keypoints['right_eye']
# nose_p = keypoints['nose']
# mouth_left_p = keypoints['mouth_left']
# mouth_right_p = keypoints['mouth_right']

