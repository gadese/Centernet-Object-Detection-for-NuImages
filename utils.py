from math import floor
import numpy as np
import cv2
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from collections import namedtuple
from typing import List, Union

from trainingconfig import config
from data_aug import *

# # Utils
def normalize_image(image):
    """Normalize the image for the Hourglass network.
    # Arguments
      image: BGR uint8
    # Returns
      float32 image with the same shape as the input
    """
    mean = [0.40789655, 0.44719303, 0.47026116]
    std = [0.2886383, 0.27408165, 0.27809834]
    return ((np.float32(image) / 255.) - mean) / std

def heatmap_nu(bbox, label):
    def get_coords(in_bbox):
        xs,ys,wid,hei=[],[],[],[]
        for box in in_bbox:
            box = [int(b) for b in box]

            # NuImages are in the format [X1, Y1, X2, Y2]
            x1, y1, x2, y2 = box
            xs.append((x1 + x2)//2)
            ys.append((y1 + y2)//2)
            wid.append(abs(x2-x1))
            hei.append(abs(y2-y1))

        return xs, ys, wid, hei

    def get_heatmap(p_x, p_y,in_label):
        # Ref: https://www.kaggle.com/diegojohnson/centernet-objects-as-points
        X1 = np.linspace(1, config.in_size, config.in_size)
        Y1 = np.linspace(1, config.in_size, config.in_size)
        [X, Y] = np.meshgrid(X1, Y1)
        X = X - floor(p_x)
        Y = Y - floor(p_y)
        D2 = X * X + Y * Y
        sigma_ = 10
        E2 = 2.0 * sigma_ ** 2
        Exponent = D2 / E2
        heatmap = np.exp(-Exponent)
        # hm = heatmap[:, :, np.newaxis]
        hm = np.zeros((config.in_size, config.in_size, config.num_classes_nu))
        hm[:,:,in_label] = heatmap
        return hm

    coors = []
    size = 20 #Change this value when downsizing input img size?
    y_ = size
    while y_ > -size - 1:
        x_ = -size
        while x_ < size + 1:
            coors.append([x_, y_])
            x_ += 1
        y_ -= 1

    u, v, w, h = get_coords(bbox)

    if len(bbox) == 0:
        u = np.array([config.in_size//2])
        v = np.array([config.in_size//2])
        w = np.array([10])
        h = np.array([10])

    hm = np.zeros((config.in_size,config.in_size,config.num_classes_nu))
    width = np.zeros((config.in_size,config.in_size,1))
    height = np.zeros((config.in_size,config.in_size,1))
    for i in range(len(u)):
        for coor in coors:
            try:#For a 20x20 square centered on (u,v), we have the width/height normalized by the output map size (this gives the w/h per output unit)
                width[int(v[i])+coor[0], int(u[i])+coor[1]] = w[i] / config.out_size
                height[int(v[i])+coor[0], int(u[i])+coor[1]] = h[i] / config.out_size
            except:
                pass
        heatmap = get_heatmap(u[i], v[i], label[i])
        hm[:,:] = np.maximum(hm[:,:,:],heatmap[:,:,:])

    hm = cv2.resize(hm, (config.out_size,config.out_size))#[:,:,None]
    hm = hm / np.max(hm)
    width = cv2.resize(width, (config.out_size,config.out_size))[:,:,None]
    height = cv2.resize(height, (config.out_size,config.out_size))[:,:,None]
    return hm, width, height

# # Dataset
class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras"""
    def __init__(self, df, mode='fit', batch_size=4, dim=(128, 128), n_channels=3,
                 n_classes=3, shuffle=True, aug=1.0, val_gen=False):#, nbr_batches=10):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.mode = mode
        self.base_path = config.IMAGE_PATH
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.random_state = config.seed
        self.aug = aug
        self.dflength = int(np.floor(self.df.shape[0] / self.batch_size))
        self.val_gen = val_gen

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.dflength

    def __getitem__(self, index):
        X, X_orig, bboxes, labels = self.get_img_box_pairs_nu(index, aug=self.aug)

        if self.mode == 'fit':
            y = self._generate_y_multiboxes(bboxes, labels)

            return X, y

        elif self.mode == 'predict':
            return np.array(X)

        elif self.mode == 'visualize':
            return np.array(X), np.array(X_orig), bboxes, labels

        else:
            raise AttributeError('The mode parameter should be set to "fit" or "predict" or "visualize".')

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.df.shape[0])
        if self.shuffle:
            np.random.seed(self.random_state)
            np.random.shuffle(self.indexes)

    def get_img_box_pairs_nu(self, index, aug=0):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        items = [self.df.iloc[[k]].values[0] for k in indexes]
        imgNames = []
        boxes = []
        labels = []
        for item in items:
            imgNames.append(item[0])
            boxes.append([[np.float64(coord) for coord in box] for box in item[1]])
            labels.append(item[2])
        Xs = self.__generate_X_nu(imgNames)

        transforms = Sequence(self.random_aug_nu(aug))
        X, bboxes = [], []

        for im, box in zip(Xs, boxes):
            im_, box_ = transforms(im, box)
            X.append(im_)
            bboxes.append(box_)

        return np.array(X), np.array(Xs), np.array(bboxes), np.array(labels) #Image resized/normalized, Image original, bbox on resized

    def __generate_X_nu(self, imgNames):
        """Generates data containing batch_size samples"""
        X = []

        for img_path in imgNames:
            parts = img_path.split('/')
            newname = '/media/gadese/SSDexternal/Google Drive/NuImages/samples/'+parts[-2]+'/'+parts[-1]
            img = self._load_rgb(newname)

            X.append(img)

        return np.array(X)

    def _generate_y_multiboxes(self, all_bboxes, all_labels):
            y1 = []#width, height
            y2 = []#class and x,y

            for bboxes, labels in zip(all_bboxes, all_labels):
                mask, width, height = heatmap_nu(bboxes, labels)
                y1.append(np.concatenate([mask,width,height], axis=-1)) #heatmap for position/class, width, height
                y2.append(mask)

            y1 = np.array(y1)
            y2 = np.array(y2)
            return [y1,y2]

    def _load_grayscale(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.
        img = np.expand_dims(img, axis=-1)
        return img

    def _load_rgb(self, img_path, pair=False):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        return img

    def random_aug_nu(self, prob=0.5):
            augments = []
            if not self.val_gen:
              if random.random() < prob:
                  augments.append(RandomHorizontalFlip(0.2, dim2coord=False))
              if random.random() < prob:
                  augments.append(RandomScale(0.1, diff=True, dim2coord=False))
              if random.random() < prob:
                  augments.append(RandomTranslate(0.05, dim2coord=False))
              if random.random() < prob:
                  augments.append(RandomRotate(10, dim2coord=False))
              if random.random() < prob:
                  augments.append(RandomShear(0.05, dim2coord=False))
              if random.random() < prob:
                  augments.append(RandomColorShift(0.1))
            augments.append(Normalize())
            augments.append(Resize((config.in_size, config.in_size), dim2coord=False))
            return augments

# # IOU/Precision Utils
# Ref: https://www.kaggle.com/pestipeti/competition-metric-details-script

Box = namedtuple('Box', 'xmin ymin xmax ymax')

def calculate_iou(gt: List[Union[int, float]],
                  pred: List[Union[int, float]],
                  form: str = 'pascal_voc') -> float:
    """Calculates the IoU.

    Args:
        gt: List[Union[int, float]] coordinates of the ground-truth box
        pred: List[Union[int, float]] coordinates of the prdected box
        form: str gt/pred coordinates format
            - pascal_voc: [xmin, ymin, xmax, ymax]
            - coco: [xmin, ymin, w, h]
    Returns:
        IoU: float Intersection over union (0.0 <= iou <= 1.0)
    """
    if form == 'coco':
        bgt = Box(gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3])
        bpr = Box(pred[0], pred[1], pred[0] + pred[2], pred[1] + pred[3])
    else:
        bgt = Box(gt[0], gt[1], gt[2], gt[3])
        bpr = Box(pred[0], pred[1], pred[2], pred[3])


    overlap_area = 0.0
    union_area = 0.0

    # Calculate overlap area
    dx = min(bgt.xmax, bpr.xmax) - max(bgt.xmin, bpr.xmin)
    dy = min(bgt.ymax, bpr.ymax) - max(bgt.ymin, bpr.ymin)

    if (dx > 0) and (dy > 0):
        overlap_area = dx * dy

    # Calculate union area
    union_area = (
            (bgt.xmax - bgt.xmin) * (bgt.ymax - bgt.ymin) +
            (bpr.xmax - bpr.xmin) * (bpr.ymax - bpr.ymin) -
            overlap_area
    )

    return overlap_area / union_area

def find_best_match(gts, predd, threshold=0.5, form='pascal_voc'):
    """Returns the index of the 'best match' between the
    ground-truth boxes and the prediction. The 'best match'
    is the highest IoU. (0.0 IoUs are ignored).

    Args:
        gts: Coordinates of the available ground-truth boxes
        predd: Coordinates of the predicted box
        threshold: Threshold
        form: Format of the coordinates

    Return:
        Index of the best match GT box (-1 if no match above threshold)
    """
    best_match_iou = -np.inf
    best_match_idx = -1

    for gt_idx, ggt in enumerate(gts):
        iou = calculate_iou(ggt, predd, form=form)

        if iou < threshold:
            continue

        if iou > best_match_iou:
            best_match_iou = iou
            best_match_idx = gt_idx

    return best_match_idx

def calculate_precision(preds_sorted, gt_boxes, threshold=0.5, form='coco'):
    """Calculates precision per at one threshold.

    Args:
        :param preds_sorted:
        :param form:
        :param threshold:
        :param gt_boxes:
    """
    tp = 0
    fp = 0
    fn = 0

    fn_boxes = []

    for pred_idx, pred in enumerate(preds_sorted):
        best_match_gt_idx = find_best_match(gt_boxes, pred, threshold=threshold, form='coco')

        if best_match_gt_idx >= 0:
            # True positive: The predicted box matches a gt box with an IoU above the threshold.
            tp += 1

            # Remove the matched GT box
            gt_boxes = np.delete(gt_boxes, best_match_gt_idx, axis=0)

        else:
            # No match
            # False positive: indicates a predicted box had no associated gt box.
            fn += 1
            fn_boxes.append(pred)

    # False negative: indicates a gt box had no associated predicted box.
    fp = len(gt_boxes)
    precision = tp / (tp + fp + fn)
    return precision, fn_boxes, gt_boxes

def calculate_image_precision(preds_sorted, gt_boxes, thresholds=(0.5), form='coco', debug=False):

    n_threshold = len(thresholds)
    image_precision = 0.0

    for threshold in thresholds:
        precision_at_threshold, _, _ = calculate_precision(preds_sorted, gt_boxes, threshold=threshold, form=form)
        if debug:
            print("@{0:.2f} = {1:.4f}".format(threshold, precision_at_threshold))

        image_precision += precision_at_threshold / n_threshold

    return image_precision

def process_img(img, mode='training'):
    img = cv2.resize(img, (config.in_size, config.in_size))
    if mode == 'training':
        img = normalize_image(img)

    return img

def show_result(test_img, sample_id, preds, labels, gt_boxes, gt_labels, dimToCoordLabels=True):
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    shape = test_img.shape
    # print(shape)

    for idx, pred_box in enumerate(preds):
        fullsize_box = inverse_resize_bbox([pred_box], shape, (config.in_size, config.in_size), dimToCoord=True)[0]
        y1 = fullsize_box[1]
        x1 = fullsize_box[0]
        y2 = y1 + fullsize_box[3]
        x2 = x1 + fullsize_box[2]
        cv2.rectangle(
            test_img,
            (x1, y1),
            (x2, y2),
            (220, 0, 0), 2
        )
        # cv2.putText(test_img, labels[idx], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (220, 0, 0), 2)

    if gt_boxes is not None:
        for gt_idx, gt_box in enumerate(gt_boxes):
            # print(gt_box)
            fullsize_box = inverse_resize_bbox([gt_box], shape, (config.in_size, config.in_size), dimToCoord=dimToCoordLabels)[0]
            # print(fullsize_box)
            if dimToCoordLabels:
              y1gt = int(np.floor(fullsize_box[1]))
              x1gt = int(np.floor(fullsize_box[0]))
              y2gt = y1gt + int(np.floor(fullsize_box[3]))
              x2gt = x1gt + int(np.floor(fullsize_box[2]))
            else:
              y1gt = int(fullsize_box[1])
              x1gt = int(fullsize_box[0])
              y2gt = int(fullsize_box[3])
              x2gt = int(fullsize_box[2])
            cv2.rectangle(test_img, (x1gt, y1gt), (x2gt, y2gt), (0, 0, 220), 2)
            # cv2.putText(test_img, gt_labels[gt_idx], (x1gt, y2gt), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 220), 2)

    ax.set_axis_off()
    ax.imshow(test_img)
    ax.set_title("RED: Predicted | BLUE - Ground-truth(if available)")
    plt.show()

def calcmAP(decoded_model, threshold=0.3, datagen = None):

  iou_thresholds = [x for x in np.arange(0.5, 0.76, 0.05)]

  precision = []

  for idx in range(len(datagen)):
      img_test, img_test_orig, boxes, _ = datagen.get_img_box_pairs_nu(idx)
      decoded_pred = decoded_model.predict(img_test)

      for i in range(decoded_pred.shape[0]):
          pred_box,scores=[],[]
          for detection in decoded_pred[i]: #[xs, ys, scores, classes, width, height]
              if detection[2] > threshold:
                  x, y, score, _, width, height = detection
                  pred_box.append([max(x-(width/2.), 0), max(y-(height/2.), 0), width, height])
                  scores.append(score)

          pred_box = np.array(pred_box, dtype=np.int32)
          scores = np.array(scores)

          preds_sorted_idx = np.argsort(scores)[::-1]
          preds_sorted = pred_box[preds_sorted_idx]

          if len(boxes) > 0:
              image_precision = calculate_image_precision(preds_sorted, boxes[i], thresholds=iou_thresholds, form='coco', debug=False)
              precision.append(image_precision)
          else:
              if len(preds_sorted) > 0:
                  precision.append(0)

  precision = np.array(precision)
  return np.mean(precision)

import tensorflow as tf
class SaveBestmAP(tf.keras.callbacks.Callback):
  def __init__(self, path):
    super(SaveBestmAP, self).__init__()
    self.best_weights = None
    self.path = path

  def on_train_begin(self, logs=None):
    self.best = 0

  def on_epoch_end(self, epoch, logs=None):
    current = calcmAP(self.model, 0.4)
    if np.greater(current, self.best):
      self.best = current
      self.best_weights = self.model.get_weights()
      print(f'Best mAP: {current}, saving...')
      self.model.save_weights(self.path)
    else:
      print(f'Current mAP: {current}')

  def on_train_end(self, logs=None):
    print(f'Loading best model...')
    self.model.load_weights(self.path)