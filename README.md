# Autonomous car object detection

## Project Overview
My goal with this project was to become familiar with the inner workings of the CenterNet architecture for Object Detection, without using a model Zoo.

The problem is a typical object detection problem; I am using the NuImages dataset (https://www.nuscenes.org/nuimages), where typical self-driving car objects (cars, trucks, traffic signs, people, etc.) are identified with a bounding box, as well as a label.

## Baseline
The code goes through the whole model training pipeline: loading the data, visualizing data, train/dev split, preprocessing, creating a data generator, building the model with hourglass architecture, loading pre-trained weights, transfer learning (training) and evaluating. 

The current version of the model is functional and able to localize the most of the relevant objects in the image. 

### Baseline example prediction
Here are some predictions obtained from the model. As we can see, the model is able to succesfully detect the speed signs. The IOU between the bounding box and the ground-truth can definitely be increased, but the current performance is most likely due to the very low number of training epochs(2 epochs). This should be easy to fix with more training, giving a bigger weight to the loss function related to the bounding box, some data augmentation and model-tuning.

![Prediction 1](./images/Figure_1_simplesuccess.png)

![Prediction 2](./images/Figure_3_goodperformance.png)

The model is also quite robust to hard scenarios: here, the model needs to detect the reflection of objects rather than objects themselves and actually does pretty well.

![Prediction 3](./images/Figure_4_hardscenario.png)

### Baseline weakpoints and edge cases
The baseline is obviously not perfect, as we can see in the following example. We see that some objects or not found (or are misclassified) by the model. 

![Edge case 1](./images/Figure1_smallerrors.png)

## Label classification
Label classification is currently functional. Objectness is generally good (the detector correctly finds cars and people as objects of interest). However, performance is slightly worse when it comes to the bounding boxes dimensions. The boxes are correctly placed, but in most cases are bigger than they should be, which reduces IoU between the ground-truth and predicted boxes.

This could possibly be due to a lack of data. There are a few possible things to try in order to improve this part:
1. Training for longer
2. Giving a bigger weight to the part of the loss related to bounding box dimensions
3. Data augmentation (simulate more data)

## Data augmentation
Currently, the following transforms are supported for data augmentation:
1. Random Horizontal Flip, according to a probability p
2. Random Scale, randomly sampled from a specified range
3. Random Translate, randomly sampled from a specified range
4. Random Rotate, randomly sampled from a specified range
5. Random Shear, randomly sampled from a specified range
6. Random Color Shift, randomly shifts every channel independantly according to a factor randomly sampled from a specified range
7. Resize (letterbox transform) - This is used to resize the images to the desired input size for the network

During training, transforms are randomly applied to training image. However, the current implementation shouldn't be seen as generating "new" samples, but rather as "adding noise" to the dataset at every epoch. This should probably be changed.

### TO DO
- [x] Add label classification
- [x] Data is currently very limited, so I plan on adding data augmentation
- [x] Adding support for multiple objects in an image(heatmap generation)
- [ ] Adding MixUp data augmentation
- [ ] Trying adversarial training for increased robustness

### References
1. https://github.com/see--/keras-centernet
2. https://www.kaggle.com/c/mapmyindia2/data
3. https://www.kaggle.com/greatgamedota/centernet-baseline-keras-training
4. https://blog.paperspace.com/data-augmentation-for-bounding-boxes/
