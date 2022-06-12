# # Imports and Load data
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
import sklearn
import random
import matplotlib.pyplot as plt
from ast import literal_eval

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from model import HourglassNetwork, add_decoder
from loss import *
from utils import DataGenerator, show_result, calcmAP
from trainingconfig import config

"""
Environment setup
"""
import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(config.seed)

"""
Loading data
"""
df_nu = pd.read_csv('/media/gadese/SSDexternal/Google Drive/NuImages/img_info.csv', converters={'Bboxes': literal_eval, 'Labels': literal_eval})
label_enc_nu = sklearn.preprocessing.LabelEncoder()
label_enc_nu.fit([label for objects in df_nu['Labels'] for label in objects])
labels_nu = df_nu['Labels'].apply(lambda x: label_enc_nu.transform(x))
nbr_labels_nu = label_enc_nu.classes_.size
df_nu['Labels'] = labels_nu

# df_nu.columns=["imgPath", "Bboxes", "Labels"] # Boxes are (x1, y1, x2, y2)
# df_nu['Bboxes'] = df_nu['Bboxes'].apply(literal_eval)
# df_nu['Labels'] = df_nu['Labels'].apply(literal_eval)

train_df_nu, val_df_nu = train_test_split(df_nu, random_state=config.seed, test_size=0.15)
train_df_nu = train_df_nu[0:10]
val_df_nu = val_df_nu[0:50]

"""
Visualize sample from data
Nu images BBoxes are Left_X, Top_Y, Right_X, Bottom_Y
"""
sample_data_nu = train_df_nu.iloc[[0]]
img_name_nu= sample_data_nu['Imgpath'].values[0]
label_nu = sample_data_nu['Labels'].values[0]

namespath = img_name_nu.split('/')
testname = '/media/gadese/SSDexternal/Google Drive/NuImages/samples/'+namespath[-2]+'/'+namespath[-1]
img_nu = cv2.cvtColor(cv2.imread(f"{testname}"), cv2.COLOR_BGR2RGB)

img_nu_ = img_nu.copy()
img_nu_resized = cv2.resize(img_nu_, (config.in_size, config.in_size))
for my_box in sample_data_nu['Bboxes'].values[0]:
    left_nu, top_nu, right_nu, bottom_nu = my_box
    top_nu_resized = top_nu * config.in_size // img_nu_.shape[0]
    left_nu_resized = left_nu * config.in_size // img_nu_.shape[1]
    bottom_nu_resized = bottom_nu * config.in_size // img_nu_.shape[0]
    right_nu_resized = right_nu * config.in_size // img_nu_.shape[1]

    cv2.rectangle(
        img_nu_resized,
        (left_nu_resized, top_nu_resized),
        (right_nu_resized, bottom_nu_resized),
        (220, 0, 0), 2
    )
# plt.imshow(img_nu_);plt.show()

"""
Build model and data generators
"""
train_gen_nu = DataGenerator(
    df=train_df_nu,
    batch_size=config.batch_size,
    dim=(1600,900),
    n_classes=config.num_classes_nu,
    shuffle=True
)
val_gen_nu = DataGenerator(
    df=val_df_nu,
    batch_size=config.batch_size,
    dim=(1600,900),
    n_classes=config.num_classes_nu,
    shuffle=True
)

xtest_nu, ytest_nu = train_gen_nu.__getitem__(1) #Random example

kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'inres': (config.in_size, config.in_size),
}
heads = { #Change these numbers to change the number of outputs
    'regr': 2, #number of values to regress; here 2: width and height
    'confidence': config.num_classes_nu #number of classes;
}
model = HourglassNetwork(heads=heads, **kwargs) #outputs: head1 = 2 channels for w,h; head2 = 1 channel for x,y heat map (confidence of having an object at (x,y))

opt = Adam(lr=config.lr, clipnorm=1)
model.compile(optimizer=opt, loss=[regress_loss, conf_loss], loss_weights=[3, 2])

# # Training
checkpoint1 = ModelCheckpoint(#save current best
    'trained_models/hourglass-label-best.h5',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto'
)

checkpoint2 = ModelCheckpoint(#save latest
    'trained_models/hourglass-label-latest.h5',
    monitor='loss',
    verbose=1,
    save_best_only=False,
    save_weights_only=True,
    mode='auto'
)
checkpoint1_full = ModelCheckpoint(#save current best
    'trained_models/hourglass-label-full-best.h5',
    monitor='loss',
    verbose=1,
    save_best_only=True,
    mode='auto'
)

checkpoint2_full = ModelCheckpoint(#save latest
    'trained_models/hourglass-label-full-latest.h5',
    monitor='loss',
    verbose=1,
    save_best_only=False,
    mode='auto'
)

reducelr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.25,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

"""
Train model and visualize losses
"""

# history = model.fit(
#     train_gen_nu,
#     validation_data=val_gen_nu,
#     epochs=config.epochs,
#     callbacks=[reducelr, checkpoint1, checkpoint2, tensorboard_callback, checkpoint1_full, checkpoint2_full],
#     use_multiprocessing=False,
# )
# #
# plt.plot(history.history['regr.1.1_loss'])
# plt.plot(history.history['val_regr.1.1_loss'])
# plt.title('regr loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# plt.plot(history.history['confidence.1.1_loss'])
# plt.plot(history.history['val_confidence.1.1_loss'])
# plt.title('confidence loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()

model.load_weights("trained_model/VertexAI/hourglass-label-best.h5")
# model.load_weights("hourglass1-label.h5")


"""
Evaluate model and visualize output heatmaps for a sample
"""
# results = model.evaluate(val_gen_nu, use_multiprocessing=False, workers=4)
# print("Val loss:", results)

X_test, y_test = val_gen_nu.__getitem__(1)
plt.imshow(X_test[0,:,:,:])
plt.show()

# regr, hm = model.predict(X_test)
#
# plt.imshow(tf.sigmoid(hm[0][:, :, 0]))
# plt.show()
# plt.imshow(regr[0][:, :, 0])
# plt.show()
# plt.imshow(regr[0][:, :, 1])
# plt.show()

"""
View bboxes prediction on sample images
"""
decoded_model = add_decoder(model)
#
for batch in range(10):
    img_test, img_test_orig, boxes, gt_labels = val_gen_nu.get_img_box_pairs_nu(batch)
    # decoded_model.run_eagerly = True
    decoded_pred = decoded_model.predict(img_test)

    for i in range(decoded_pred.shape[0]):
        pred_box, scores, labels = [], [], []
        for detection in decoded_pred[i]:  # [xs, ys, scores, classes, width, height]
            if detection[2] > 0.3:
                x, y, score, label, width, height = detection
                pred_box.append([max(x - (width / 2.), 0), max(y - (height / 2.), 0), width, height])
                scores.append(score)
                labels.append(label)

        pred_box = np.array(pred_box, dtype=np.int32)
        scores = np.array(scores)
        labels = np.array(labels)

        preds_sorted_idx = np.argsort(scores)[::-1]
        preds_sorted = pred_box[preds_sorted_idx]
        labels_sorted = labels[preds_sorted_idx]
        labels_decoded = label_enc_nu.inverse_transform([int(label) for label in labels_sorted])
        gt_labels_decoded = label_enc_nu.inverse_transform([int(label) for label in gt_labels[i]])
        # plt.imshow(draw_rect_nu(images, bboxes, dim2coord_=False)); plt.show()
        show_result(img_test_orig[i], sample_id=i, preds=preds_sorted, labels=labels_decoded, gt_boxes=boxes[i], gt_labels = gt_labels_decoded,
                    dimToCoordLabels=False)

