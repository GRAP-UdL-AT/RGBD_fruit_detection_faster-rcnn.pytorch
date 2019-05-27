# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Re-written by Jordi Gené-Mola, based on code from Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def im_list_to_blob(ims, RGB, NIR, DEPTH):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    if RGB & NIR & DEPTH:
      blob = np.zeros((num_images, max_shape[0], max_shape[1], 5),
                        dtype=np.float32)
    elif (RGB & NIR) | (RGB & DEPTH) | (NIR & DEPTH):
      blob = np.zeros((num_images, max_shape[0], max_shape[1], 4),
                        dtype=np.float32)
    else:
      blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size,RGB,NIR,DEPTH):
    """Mean subtract and scale an image for use in a blob."""

    im = im.astype(np.float32, copy=False)
    if RGB:
      p_means=pixel_means[:,:,:3]
      if NIR:
        p_means = np.concatenate((p_means, pixel_means[:, :, 3:4]), axis=2)
      if DEPTH:
        p_means = np.concatenate((p_means, pixel_means[:, :, 4:5]), axis=2)
    elif NIR:
      if not DEPTH:
        p_means = np.concatenate((pixel_means[:, :, 3:4], pixel_means[:, :, 3:4], pixel_means[:, :, 3:4]), axis=2)
      else:
        p_means = np.concatenate((pixel_means[:, :, 3:5], pixel_means[:, :, 3:5]), axis=2)
    elif DEPTH:
      p_means = np.concatenate((pixel_means[:, :, 4:5], pixel_means[:, :, 4:5], pixel_means[:, :, 4:5]), axis=2)
    else:
      print('Any color space was selected')

    im -= p_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale
