# Speed Signs detection

## Project Overview
My goal with this project was to become familiar with the CenterNet architecture for Object Detection.
It is still on-going, and I will be adding more features as time goes. 

The problem is a typical object detection problem; I am using a small Kaggle dataset (https://www.kaggle.com/c/mapmyindia2/data) where speed limit signs are identified with a bounding box, as well as a label for the associated speed limit (e.g. a max speed 80 sign would have label '80').

### Data format

    Img_Name - name of processed image
    Top - distance of label from top
    Left - distance of label from left
    Width - width of label
    Height - height of the label
    Label - label

## Current Baseline
The code goes through the whole model training pipeline: loading the model, visualizing data, train/dev split, preprocessing, creating a data generator, training and evaluating. 

The current version of the model is functional and able to localize the speed signs in images. Since I am building the model from the ground-up, it currently doesn't tell the difference between classes, although that is the next step.

### Example prediction
Here are some predictions obtained from the model. As we can see, the model is able to succesfully detect the speed signs. The IOU between the bounding box and the ground-truth can definitely be increased, but the current performance is most likely due to the very limited number of training images. This should be easy to fix with some data augmentation and model-tuning.

![Prediction 1](./images/Figure_1.png)

![Prediction 2](./images/Figure_3.png)

The following image shows that even with heavy distortion (caused by heatwaves), the model is still able to correctly identify the speed sign. In other words, the model is somewhat robust already.

![Prediction 3](./images/Figure_2_distorted.png)

### Current weakpoints and edge cases
These are the cases that need to be worked on in the next steps. In the first one, the sign is rotated sideways; the model is still able to detect it, but the bounding box is barely over the sign. In the second one, we see that the smaller sign on the left is split into two boxes.

![Edge case 1](./images/Figure_4_sideways.png)

![Edge case 2](./images/Figure_5_multiple.png)
AMvcGhT7VVjujz
### TO DO
1. Add label classification
2. Data is currently very limited, so I plan on adding simple data augmentation
3. Adding MixUp data augmentation
4. Trying adversarial training for increased robustness

### References
1. https://github.com/see--/keras-centernet
2. https://www.kaggle.com/c/mapmyindia2/data
3. https://www.kaggle.com/greatgamedota/centernet-baseline-keras-training
4. https://blog.paperspace.com/data-augmentation-for-bounding-boxes/