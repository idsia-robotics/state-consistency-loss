import cv2
import numpy as np
from tf.transformations import euler_from_quaternion


def quaternion2yaw(q):
    '''Converts a quaternion into the respective z euler angle.

    Args:
            q: a quaternion, composed of X, Y, Z, W.

    Returns:
             The euler angle part for the z axis a.k.a. yaw.
    '''
    if hasattr(q, 'x'):
        q = [q.x, q.y, q.z, q.w]

    return euler_from_quaternion(q)[2]


def mktr(x, y):
    """Returns a translation matrix given x and y."""
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])


def mkrot(theta):
    """Returns a rotation matrix given theta."""
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([[cos, -sin, 0],
                     [sin, cos, 0],
                     [0, 0, 1]])


def mktransf(pose):
    """Returns a trasnformation matrix given a (x, y, theta) pose."""
    cos = np.cos(pose[2])
    sin = np.sin(pose[2])
    return np.array([[cos, -sin, pose[0]],
                     [sin, cos, pose[1]],
                     [0, 0, 1]])


def lerp(x, y, a):
    """Linear interpolation between two points by a fixed amount.

    Args:
            x: the first point.
            y: the second point.
            a: the percentage between the two points.

    Returns:
            the interpolated point.
    """
    return (1 - a) * x + a * y


def jpeg2np(image, size=None, normalize=False):
    '''Converts a jpeg image in a 2d numpy array of BGR pixels and resizes it to the given size (if provided).

    Args:
            image: a compressed BGR jpeg image.
            size: a tuple containing width and height, or None for no resizing.
            normalize: a boolean flag representing wether or not to normalize the image.

    Returns:
            the raw, resized image as a 2d numpy array of BGR pixels.
    '''
    compressed = np.fromstring(image, np.uint8)
    raw = cv2.imdecode(compressed, cv2.IMREAD_COLOR)

    if size:
        raw = cv2.resize(raw, size)

    if normalize:
        raw = (raw - raw.mean()) / raw.std()

    return raw


def bgr_tensor_to_rgb_numpy(tensor):
    """Converts a pytorch tensor containing a BGR image to an RGB numpy array

    Args:
            tensor: the BGR tensor to be converted.

    Returns:
            the BGR tensors converted in an RGB numpy array.
    """
    im = tensor.cpu().numpy().transpose([1, 2, 0])  # D x H x W to H x W x D
    im = im[:, :, ::-1]  # bgr to rgb
    im = (im - im.min()) / (im.max() - im.min())  # normalize in [0, 1]
    return im


def map_to_image(tensor, size=(5, 32), spacing=(0, 9), bgcolor=np.array((255, 255, 255)),
                 occupied_color=np.array((0, 0, 200)), empty_color=np.array((30, 230, 30))):
    """Renders a pytorch tensor containing an occupancy map to an RGB numpy array

    Args:
            tensor: the tensor containing an occupancy map.

    Returns:
            the occupancy map rendered as an RGB numpy array.
    """
    r, c = tuple(tensor.shape)
    map_r = r * size[0] + (r - 1) * spacing[0]
    map_c = c * size[1] + (c - 1) * spacing[1]

    values = tensor.cpu().numpy().astype(float)
    values = np.rot90(values, 1)
    values = np.fliplr(values)

    map = np.full((map_r, map_c, 3), bgcolor, dtype=np.uint8)
    for i in range(r):
        for j in range(c):
            val = values[i, j]
            if val == -1:  # if label is missing display gray box
                color = np.array((128, 128, 128))
            elif val == 0:
                color = empty_color
            elif val == 1:
                color = occupied_color
            elif 0 < val and val < 1:
                color = lerp(empty_color, occupied_color, val)
            else:
                raise ValueError('Encountered unexpected value ' +
                                 str(val) + ' in the occupancy map')

            map[i * (size[0] + spacing[0]):i * (size[0] + spacing[0]) + size[0],
                j * (size[1] + spacing[1]):j * (size[1] + spacing[1]) + size[1], :] = color

    return map
