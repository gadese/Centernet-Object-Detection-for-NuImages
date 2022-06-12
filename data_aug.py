import random

from bbox_aug_utils import *

class Sequence(object):

    """Initialise Sequence object

    Apply a Sequence of transformations to the images/boxes.

    Parameters
    ----------
    augmentations : list
        List containing Transformation Objects in Sequence they are to be
        applied

    probs : int or list
        If **int**, the probability with which each of the transformation will
        be applied. If **list**, the length must be equal to *augmentations*.
        Each element of this list is the probability with which each
        corresponding transformation is applied

    Returns
    -------
    Sequence
        Sequence Object


    Example call: transforms = Sequence([RandomHorizontalFlip(1, dim2coord=True), RandomScale(0.2, diff = True, dim2coord=True), RandomRotate(10, dim2coord=True)])

    """
    def __init__(self, augmentations, probs = 1):

        self.augmentations = augmentations
        self.probs = probs

    def __call__(self, images, bboxes):
        bboxes = np.array(bboxes)
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)#plt.imshow(draw_rect(images, bboxes)); plt.show()
                # plt.imshow(draw_rect_nu(images, bboxes, dim2coord_=False)); plt.show()
        return images, bboxes

class RandomHorizontalFlip(object):

    """Randomly horizontally flips the Image with the probability *p*

    Parameters
    ----------
    p: float - The probability with which the image is flipped
    dim2coord: bool - Whether to change bounding boxes from (x,y,width,height) to (x1,y1,x2,y2)

    Returns
    -------
    numpy.ndaaray - Flipped image in the numpy format of shape `HxWxC`
    numpy.ndarray - Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    Usage:
        hor_flip = RandomHorizontalFlip(0.5); img, bboxes = hor_flip(img, bboxes, dim2coord=False)

    """
    def __init__(self, p=0.5, dim2coord = False):
        self.p = p
        self.dim2coord=dim2coord

    def __call__(self, img, bboxes):
        img_center = np.array(img.shape[:2])[::-1]/2
        img_center = np.hstack((img_center, img_center))

        if self.dim2coord:
            bboxes = dim2coord(bboxes)

        if random.random() < self.p:
            img =  img[:,::-1,:] #flip img
            bboxes[:,[0,2]] += 2*(img_center[[0,2]] - bboxes[:,[0,2]]) #shift bbox

            box_w = abs(bboxes[:,0] - bboxes[:,2])

            bboxes[:,0] -= box_w
            bboxes[:,2] += box_w

        if self.dim2coord:
            bboxes = coord2dim(bboxes)

        return img, bboxes

class RandomScale(object):
    """Randomly scales an image

    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the
        tuple
    dim2coord: bool - Whether to change bounding boxes from (x,y,width,height) to (x1,y1,x2,y2)
    diff: bool - Whether to keep aspect ratio (if False, ratio is maintained)

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, scale = 0.2, diff = False, dim2coord = False):
        self.scale = scale
        self.dim2coord=dim2coord

        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)

        self.diff = diff

    def __call__(self, img, bboxes):
        #Choose a random digit to scale by
        img_shape = img.shape
        if self.dim2coord:
            bboxes = dim2coord(bboxes)
        if self.diff:# Find new random scale factor
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        img = cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)#Resize to new dimensions
        bboxes[:,:4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]#Resize bbox

        # canvas = np.zeros(img_shape, dtype = np.float64)#Black image
        canvas = np.zeros(img_shape, dtype = np.uint8)#Black image

        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])
        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]#black image up to x_lim,y_lim becomes the resized image. Ensures we crop if new scale is bigger

        img = canvas
        bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]], 0.25)#If bbox is now outside the image, clip it

        if self.dim2coord:
            bboxes = coord2dim(bboxes)

        return img, bboxes

class RandomTranslate(object):
    """Randomly Translates the image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the
        tuple
    dim2coord: bool - Whether or not to change bounding boxes from (x,y,width,height) to (x1,y1,x2,y2)
    diff: bool - Whether or not to keep aspect ratio (if False, ratio is maintained)

    Returns
    -------

    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, translate = 0.2, diff = False, dim2coord = False):
        self.translate = translate
        self.dim2coord=dim2coord

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1

        else:
            assert (self.translate > 0) & (self.translate < 1)
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, bboxes):
        #Chose a random digit to scale by
        img_shape = img.shape
        if self.dim2coord:
            bboxes = dim2coord(bboxes)

        #translate the image
        #percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        if not self.diff:
            translate_factor_y = translate_factor_x

        canvas = np.zeros(img_shape, dtype=np.uint8)

        #get the top-left corner co-ordinates of the shifted image
        corner_x = int(translate_factor_x*img.shape[1])
        corner_y = int(translate_factor_y*img.shape[0])

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]

        #Destination for mask
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas

        #Shift bboxes and clip those outside the image
        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.1)#Make sure we always return a bounding box or it can crash

        if self.dim2coord:
            bboxes = coord2dim(bboxes)

        return img, bboxes


class RandomRotate(object):
    """Randomly rotates an image


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the
        tuple

    dim2coord: bool - Whether or not to change bounding boxes from (x,y,width,height) to (x1,y1,x2,y2)

    Returns
    -------

    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, angle = 10, dim2coord=False):
        self.angle = angle
        self.dim2coord=dim2coord

        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"

        else:
            self.angle = (-self.angle, self.angle)

    def __call__(self, img, bboxes):
        if self.dim2coord:
            bboxes = dim2coord(bboxes)

        angle = random.uniform(*self.angle)

        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2

        img = rotate_im(img, angle)
        corners = get_corners(bboxes)

        corners = np.hstack((corners, bboxes[:,4:]))


        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
        new_bbox = get_enclosing_box(corners)


        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h

        img = cv2.resize(img, (w,h))

        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]

        bboxes  = new_bbox
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)

        if self.dim2coord:
            bboxes = coord2dim(bboxes)

        return img, bboxes

class RandomShear(object):
    """Randomly shears an image in horizontal direction


    Bounding boxes which have an area of less than 25% in the remaining in the
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.

    Parameters
    ----------
    shear_factor: float or tuple(float)
        if **float**, the image is sheared horizontally by a factor drawn
        randomly from a range (0 `shear_factor`). If **tuple**,
        the `shear_factor` is drawn randomly from values specified by the
        tuple
    dim2coord: bool - Whether or not to change bounding boxes from (x,y,width,height) to (x1,y1,x2,y2)

    Returns
    -------

    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, shear_factor = 0.2, dim2coord=False):
        self.shear_factor = shear_factor
        self.dim2coord=dim2coord

        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"
        else:
            self.shear_factor = (0, self.shear_factor)
        # self.shear_factor[0] = 0
        # self.shear_factor = random.uniform(*self.shear_factor)

    def __call__(self, img, bboxes):
        if self.dim2coord:
            bboxes = dim2coord(bboxes)

        shear_factor = random.uniform(*self.shear_factor)

        w,h = img.shape[1], img.shape[0]

        # if shear_factor < 0:
        #     img, bboxes = RandomHorizontalFlip(p=1)(img,bboxes)

        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])

        nW =  img.shape[1] + abs(shear_factor*img.shape[0])

        bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int)


        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        # if shear_factor < 0:
        #     img, bboxes = RandomHorizontalFlip(p=1)(img,bboxes)
        img = cv2.resize(img, (w,h))

        scale_factor_x = nW / w

        bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1]

        if self.dim2coord:
            bboxes = coord2dim(bboxes)

        return img, bboxes

class Resize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet

    The aspect ratio is maintained. The longer side is resized to the input
    size of the network, while the remaining space on the shorter side is filled
    with black color. **This should be the last transform applied to the img**

    Parameters
    ----------
    inp_dim : tuple(int)
        tuple containing the size to which the image will be resized.

    dim2coord: bool - Whether or not to change bounding boxes from (x,y,width,height) to (x1,y1,x2,y2)

    Returns
    -------

    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, inp_dim, dim2coord=False):
        self.inp_dim = inp_dim
        self.dim2coord=dim2coord

    def __call__(self, img, bboxes):
        if self.dim2coord:
            bboxes = dim2coord(bboxes)
        img_ = img.copy()
        w,h = img.shape[1], img.shape[0]
        img = letterbox_image(img, self.inp_dim)

        scale = min(self.inp_dim[0]/h, self.inp_dim[1]/w)#[1361 312 1405 350], scale=0.267
        bboxes[:,:4] *= (scale)#[363 83 374 93]

        new_w = scale*w#512
        new_h = scale*h#288
        inp_dim = self.inp_dim

        del_h = (inp_dim[0] - new_h)/2#112
        del_w = (inp_dim[1] - new_w)/2#0

        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)# [0 112 0 112]

        bboxes[:,:4] += add_matrix #[363 195 374 205]

        # img = img.astype(np.float64)
        # img = img.astype(np.uint8)
        # test = inverse_resize_bbox(bboxes, img_.shape, inp_dim, dimToCoord=False)


        if self.dim2coord:
            bboxes = coord2dim(bboxes)

        return img, bboxes

class SimpleResize(object):
    """Resize the image without preserving aspect ratio

    Parameters
    ----------
    inp_dim : tuple(int)
        tuple containing the size to which the image will be resized.

    Returns
    -------

    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`

    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box

    """

    def __init__(self, inp_dim):
        self.inp_dim = inp_dim

    def __call__(self, img, bboxes):

        w,h = img.shape[1], img.shape[0]#[719 243 17 16]
        img = cv2.resize(img, self.inp_dim)

        bboxes[:,0] = bboxes[:,0] * self.inp_dim[1] // w
        bboxes[:,1] = bboxes[:,1] * self.inp_dim[0] // h
        bboxes[:,2] = bboxes[:,2] * self.inp_dim[1] // w
        bboxes[:,3] = bboxes[:,3] * self.inp_dim[0] // h

        # test = inverse_simpleresize_bbox(bboxes, (h, w), (512, 512))

        return img, bboxes

# https://stepup.ai/custom_data_augmentation_keras/
class RandomColorShift(object):
    """Randomly shifts the color of an image

    Bounding boxes and image dimensions are maintained, we only slightly change pixel values

    Parameters
    ----------
    range: float
        The percentage interval by which we shift the values from the original ones. E.g. range=0.2 would output values between [0.8, 1.2]* the original value

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        The unchanged bounding box
    """

    def __init__(self, range = 0.2):
        self.range = range

        assert self.range > 0, "Please input a positive float"

    def __call__(self, img, bboxes):

        img = img.astype(np.uint16)
        ranges = (1-self.range, 1+self.range)
        for channel in range(img.shape[-1]):
            scale = np.random.uniform(ranges[0], ranges[1])
            img[:,:, channel] = img[:,:, channel] * scale
        img = np.clip(img, 0, 255)

        return img, bboxes

class Normalize(object):
    """Normalizes the image according to pre-training data

    Bounding boxes and image dimensions are maintained, only pixel values are changes

    Parameters
    ----------

    Returns
    -------

    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`

    numpy.ndarray
        The unchanged bounding box
    """

    def __init__(self):
        pass

    def __call__(self, img, bboxes):
        return normalize_image(img), bboxes
