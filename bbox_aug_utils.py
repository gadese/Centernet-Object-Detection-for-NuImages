import cv2
import numpy as np

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

def dim2coord(bboxes):
    boxes = []
    for bbox in bboxes:
        boxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    return np.array(boxes)


def coord2dim(bboxes):
    boxes = []
    for bbox in bboxes:
        boxes.append([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
    return np.array(boxes)


def bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])


def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image

    Parameters
    ----------
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`

    alpha: float
        If the fraction of a bounding box left in the image after being clipped is
        less than `alpha` the bounding box is dropped.

    Returns
    -------

    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    ar_ = (bbox_area(bbox))
    x_min = np.maximum(bbox[:, 0], clip_box[0]).reshape(-1, 1)
    y_min = np.maximum(bbox[:, 1], clip_box[1]).reshape(-1, 1)
    x_max = np.minimum(bbox[:, 2], clip_box[2]).reshape(-1, 1)
    y_max = np.minimum(bbox[:, 3], clip_box[3]).reshape(-1, 1)
    x_max = np.maximum(x_max, 0).reshape(-1, 1)
    y_max = np.maximum(y_max, 0).reshape(-1, 1)

    bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:, 4:]))

    return bbox


def rotate_im(image, angle):
    """Rotate the image.

    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black.

    Parameters
    ----------

    image : numpy.ndarray
        numpy image

    angle : float
        angle by which the image is to be rotated

    Returns
    -------

    numpy.ndarray
        Rotated Image

    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #    image = cv2.resize(image, (w,h))
    return image


def get_corners(bboxes):
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    """
    width = (bboxes[:, 2] - bboxes[:, 0]).reshape(-1, 1)
    height = (bboxes[:, 3] - bboxes[:, 1]).reshape(-1, 1)

    # top left
    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)

    # top right
    x2 = x1 + width
    y2 = y1

    # bottom left
    x3 = x1
    y3 = y1 + height

    # bottom right
    x4 = bboxes[:, 2].reshape(-1, 1)
    y4 = bboxes[:, 3].reshape(-1, 1)

    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))

    return corners


def rotate_box(corners, angle, cx, cy, h, w):
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int
        height of the image

    w : int
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1, 2)
    corners = np.hstack((corners, np.ones((corners.shape[0], 1), dtype=type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M, corners.T).T

    calculated = calculated.reshape(-1, 8)

    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for rotated corners of a bounding box

    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    Returns
    -------

    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    """
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]

    xmin = np.min(x_, 1).reshape(-1, 1)
    ymin = np.min(y_, 1).reshape(-1, 1)
    xmax = np.max(x_, 1).reshape(-1, 1)
    ymax = np.max(y_, 1).reshape(-1, 1)

    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8:]))

    return final


def inverse_resize_bbox(bboxes, orig_dim, inp_dim, dimToCoord=True):
    if dimToCoord:
        bboxes = dim2coord(bboxes)
    else:
        bboxes = np.array(bboxes)
    w, h = orig_dim[1], orig_dim[0]
    scale = min(inp_dim[0] / w, inp_dim[1] / h)

    new_w = w * scale
    new_h = h * scale

    del_h = (inp_dim[0] - new_h) / 2
    del_w = (inp_dim[1] - new_w) / 2

    add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)

    bboxes[:, :4] -= add_matrix

    bboxes[:, :4] = bboxes[:, :4] / (scale)

    if dimToCoord:
        bboxes = coord2dim(bboxes)
    return bboxes


def inverse_simpleresize_bbox(bboxes, orig_dim, inp_dim, dimToCoord=False):
    if dimToCoord:
        bboxes = dim2coord(bboxes)
    else:
        bboxes = np.array(bboxes)

    w, h = orig_dim[1], orig_dim[0]

    bboxes[:, 0] = bboxes[:, 0] * w // inp_dim[1]
    bboxes[:, 1] = bboxes[:, 1] * h // inp_dim[0]
    bboxes[:, 2] = bboxes[:, 2] * w // inp_dim[1]
    bboxes[:, 3] = bboxes[:, 3] * h // inp_dim[0]

    if dimToCoord:
        bboxes = coord2dim(bboxes)
    return bboxes


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding

    Parameters
    ----------

    img : numpy.ndarray
        Image

    inp_dim: tuple(int)
        shape of the reszied image

    Returns
    -------

    numpy.ndarray:
        Resized image

    '''

    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h))

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0, dtype=np.float64)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def draw_rect_nu(im, cords, color=None, dim2coord_=True):
    """Draw the rectangle on the image

    Parameters
    ----------

    im : numpy.ndarray
        numpy image

    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    Returns
    -------

    numpy.ndarray
        numpy image with bounding boxes drawn on it
        :param im:
        :param cords:
        :param dim2coord_:
        :param color:

    """

    im = im.copy()
    if dim2coord_:
        cords = dim2coord(cords)
    else:
        cords = np.array(cords)

    cords = cords.reshape(-1, 4)
    if not color:
        color = [255, 255, 255]
    for cord in cords:
        pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])

        pt1 = int(pt1[0]), int(pt1[1])
        pt2 = int(pt2[0]), int(pt2[1])

        im = cv2.rectangle(im.copy(), pt1, pt2, color, int(max(im.shape[:2]) / 200))
    return im

