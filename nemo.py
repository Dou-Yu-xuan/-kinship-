import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from insightface.utils.face_align import norm_crop
from insightface import model_zoo
from pathlib import Path
import shutil
import os
from pathlib import Path
from tqdm import tqdm
import cv2
from sklearn.metrics import roc_curve, auc,accuracy_score
from matplotlib import pyplot as plt
from dataset import FamiliesDataset, PairDataset, ImgDataset
import pickle
from typing import List, Tuple, Callable
import mytypes as t


def ensure_path(cur: Path) -> Path:
    if not cur.exists():
        os.makedirs(str(cur))
    return cur


def prepare_images(root_dir: Path, output_dir: Path) -> None:
    whitelist_dir = 'MID'
    detector = model_zoo.get_model('retinaface_r50_v1')
    detector.prepare(ctx_id=-1, nms=0.4)
    for family_path in tqdm(root_dir.iterdir()):
        for person_path in family_path.iterdir():
            if not person_path.is_dir() or not person_path.name.startswith(whitelist_dir):
                continue
            output_person = ensure_path(output_dir / person_path.relative_to(root_dir))
            for img_path in person_path.iterdir():
                img = cv2.imread(str(img_path))
                bbox, landmarks = detector.detect(img, threshold=0.5, scale=1.0)
                output_path = output_person / img_path.name
                if len(landmarks) < 1:
                    print(f'smth wrong with {img_path}')
                    continue
                warped_img = norm_crop(img, landmarks[0])
                cv2.imwrite(str(output_path), warped_img)


def prepare_test_images(root_dir: Path, output_dir: Path):
    detector = model_zoo.get_model('retinaface_r50_v1')
    detector.prepare(ctx_id=-1, nms=0.4)
    output_dir = ensure_path(output_dir)
    for img_path in root_dir.iterdir():
        img = cv2.imread(str(img_path))
        bbox, landmarks = detector.detect(img, threshold=0.5, scale=1.0)
        output_path = output_dir / img_path.name
        if len(landmarks) < 1:
            print(f'smth wrong with {img_path}')
            warped_img = cv2.resize(img, (112, 112))
        else:
            warped_img = norm_crop(img, landmarks[0])
        cv2.imwrite(str(output_path), warped_img)


def norm(emb):
    return np.sqrt(np.sum(emb ** 2))


def cosine(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))


def euclidean(emb1, emb2):
    return -norm(emb1 - emb2)


def predict(model: Callable[[Path, Path], t.Labels],
            pair_list: List[t.PairPath]) -> t.Labels:
    predictions = []
    for idx, (path1, path2) in tqdm(enumerate(pair_list), total=len(pair_list)):
        cur_prediction = model(path1, path2)
        predictions.append(cur_prediction)
    return np.stack(predictions, axis=0)


class CompareModel(object):

    def __init__(self,
                 model_name: str = 'arcface_r100_v1',
                 ctx: mx.Context = mx.cpu()):
        if model_name == 'arcface_r100_v1':
            model = model_zoo.get_model(model_name)
            if ctx.device_type.startswith('cpu'):
                ctx_id = -1
            else:
                ctx_id = ctx.device_id
            model.prepare(ctx_id=ctx_id)
            self.model = model.model
        else:
            sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, 0)
            sym = sym.get_internals()['fc1_output']
            model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
            data_shape = (1,3,112,112)
            model.bind(data_shapes=[('data', data_shape)], for_training=False)
            model.set_params(arg_params, aux_params)
            #warmup
            data = mx.nd.zeros(shape=data_shape)
            db = mx.io.DataBatch(data=(data,))
            model.forward(db, is_train=False)
            embedding = model.get_outputs()[0].asnumpy()
            self.model = model
        self.embeddings = dict()
        self.metric = cosine

    def get_embedding(self, im_path: Path) -> t.Embedding:
        if not im_path in self.embeddings:
            if im_path.is_dir():
                # calculate average embedding for subject
                subject_embeddings = []
                emb = []
                for cur_img in im_path.iterdir():
                    img = mx.img.imread(str(cur_img)).transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
                    batch = mx.io.DataBatch([img])
                    self.model.forward(batch, is_train=False)
                    emb.append(self.model.get_outputs()[0][0].asnumpy())
                self.embeddings[im_path] = np.mean(emb, axis=0)
            else:
                img = mx.img.imread(str(im_path))
                img = mx.ndarray.image.resize(img,(112,112)).transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
                batch = mx.io.DataBatch([img])
                self.model.forward(batch, is_train=False)
                self.embeddings[im_path] = self.model.get_outputs()[0][0].asnumpy()
        return self.embeddings[im_path]

    def prepare_all_embeddings(self,
                               paths: List[Path],
                               **loader_params) -> None:
        data = mx.gluon.data.DataLoader(ImgDataset(paths), shuffle=False, **loader_params)
        raise NotImplementedError
    def __call__(self, path1: Path, path2: Path) -> t.Labels:
        emb1 = self.get_embedding(path1)
        emb2 = self.get_embedding(path2)
        return self.metric(emb1, emb2)




if __name__ == '__main__':

    # prepare_test_images(Path('/local/wwang/Nemo/kin_simple/frames/'), Path('./nemo_crop/'))
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
            for num_a in range(1):
                img1_pth = img_root + current_lb + '/'+ temp_lb[2] + '/' + 'frame{:0>3d}.jpg'.format(num_a) 
                img2_pth = img_root + current_lb + '/'+ temp_lb[3] + '/' + 'frame{:0>3d}.jpg'.format(num_a) 
                pairs.append((Path(img1_pth), Path(img2_pth), int(temp_lb[1])))

        
        y_true = [label for _, _, label in pairs]
        pair_list = [(img1, img2) for img1, img2, _ in pairs]
        all_paths = [o for o, _ in pair_list] + [o for _, o in pair_list]
        ctx = mx.gpu(0)
        model = CompareModel(ctx=ctx)
        model.metric = cosine
        pair_list = pair_list
        y_true = y_true
        predictions = predict(model, pair_list)
        fpr, tpr, thresholds = roc_curve(y_true, predictions)
        y = tpr - fpr
        youden_index = np.argmax(y)
        op_thr = thresholds[youden_index]
        best_threshold += op_thr
        pred = predictions > 0.1 ## set threshold
        acc = accuracy_score(y_true, pred)
        print("current_lb: {}, acc: {}".format(current_lb, acc))
        average_accuracy += acc
    print("average_accuracy: {}".format(average_accuracy/len(ls)))
    print("best_threshold: {}".format(best_threshold/7))
    







   


    