# # Imports and Load data
import pandas as pd
import cv2
import os
from sklearn.model_selection import train_test_split
import sklearn
import random
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau

from model import HourglassNetwork, add_decoder
from loss import *
from utils import DataGenerator, show_result
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
df = pd.read_csv('train.csv')
label_enc = sklearn.preprocessing.LabelEncoder()
label_enc.fit(df['Label'])
labels = label_enc.transform(df['Label'])
nbr_labels = label_enc.classes_.size
df['Label'] = labels

train_df, val_df = train_test_split(df, random_state=config.seed, test_size=0.15)
train_df = train_df[0:10]
val_df = val_df[0:10]

"""
Visualize sample from data
"""
sample_data = train_df.iloc[[0]]
img_name, top, left, width, height, label = sample_data.values[0]

img = cv2.cvtColor(cv2.imread(f"{config.IMAGE_PATH}{img_name}"), cv2.COLOR_BGR2RGB)
img_ = img.copy()
cv2.rectangle(
    img_,
    (left, top),
    (left+width, top+height),
    (220, 0, 0), 2
)

# plt.imshow(img_);plt.show()

img_resized = cv2.resize(img, (config.in_size, config.in_size))
top_resized = top * config.in_size // config.bbox_img_size[0]
left_resized = left * config.in_size // config.bbox_img_size[1]
height_resized = height * config.in_size // config.bbox_img_size[0]
width_resized = width * config.in_size // config.bbox_img_size[1]

cv2.rectangle(
    img_resized,
    (left_resized, top_resized),
    (left_resized+width_resized, top_resized+height_resized),
    (220, 0, 0), 2
)
# plt.imshow(img_resized);plt.show()

"""
Build model and data generators
"""

train_gen = DataGenerator(
    df=train_df,
    batch_size=config.batch_size,
    dim=(config.in_size,config.in_size),
    n_classes=config.num_classes,#n_classes=nbr_labels,
    shuffle=True
)

val_gen = DataGenerator(
    df=val_df,
    batch_size=config.batch_size,
    dim=(config.in_size,config.in_size),
    n_classes=config.num_classes,#n_classes=nbr_labels,
    shuffle=True
)
# xtest, ytest = train_gen.__getitem__(1)
# regr = regress_loss(ytest, ytest)
# conf = conf_loss(ytest, ytest)

kwargs = {
    'num_stacks': 2,
    'cnv_dim': 256,
    'inres': (config.in_size, config.in_size),
}
heads = { #Change these numbers to change the number of outputs
    'regr': 2, #number of values to regress; here, width/height
    'confidence': config.num_classes #number of classes; here, 1
}
model = HourglassNetwork(heads=heads, **kwargs) #outputs: head1 = 2 channels for w,h; head2 = 1 channel for x,y heat map (confidence of having an object at (x,y))

opt = Adam(lr=config.lr, clipnorm=1)
model.compile(optimizer=opt, loss=[regress_loss, conf_loss], loss_weights=[5, 1])

# # Training
checkpoint1 = ModelCheckpoint(#save current best
    'hourglass1-label.h5',
    monitor='loss',
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    mode='auto'
)

checkpoint2 = ModelCheckpoint(#save latest
    'hourglass1-label-2.h5',
    monitor='loss',
    verbose=0,
    save_best_only=False,
    save_weights_only=True,
    mode='auto'
)

reducelr = ReduceLROnPlateau(
    monitor='loss',
    factor=0.25,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

"""
Train model and visualize losses
"""

# history = model.fit_generator(
#     train_gen,
#     validation_data=val_gen,
#     epochs=config.epochs,
#     callbacks=[reducelr, checkpoint1, checkpoint2],#, savemAP],
#     use_multiprocessing=True,
# )
#
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

# model.load_weights("trained_model/hourglass1-test2.h5")
model.load_weights("trained_model/hourglass1-labels.h5")
# model.load_weights("hourglass1-label.h5")


"""
Evaluate model and visualize output heatmaps for a sample
"""
results = model.evaluate(val_gen, use_multiprocessing=False, workers=4)
print("Val loss:", results)

X_test, y_test = val_gen.__getitem__(1)
regr, hm = model.predict(X_test)
#
# plt.imshow(tf.sigmoid(hm[0][:,:,0]))
# plt.show()
# plt.imshow(regr[0][:,:,0])
# plt.show()
# plt.imshow(regr[0][:,:,1])
# plt.show()

"""
View bboxes prediction on sample images
"""
decoded_model = add_decoder(model)
#
img_test, img_test_orig, boxes = val_gen.get_pair_to_vizualise(1)
decoded_pred = decoded_model.predict(img_test)

for i in range(decoded_pred.shape[0]):
    pred_box,scores,labels=[],[],[]
    for detection in decoded_pred[i]: #[xs, ys, scores, classes, width, height]
        if detection[2] > 0.25:
            x, y, score, label, width, height = detection
            pred_box.append([max(x-(width/2.), 0), max(y-(height/2.), 0), width, height])
            scores.append(score)
            labels.append(label)

    pred_box = np.array(pred_box, dtype=np.int32)
    scores = np.array(scores)
    labels = np.array(labels)

    preds_sorted_idx = np.argsort(scores)[::-1]
    preds_sorted = pred_box[preds_sorted_idx]
    labels_sorted = labels[preds_sorted_idx]

    show_result(img_test_orig[i], 1, preds_sorted, None, labels_sorted)
#
