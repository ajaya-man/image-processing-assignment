import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def correlation_filter_with_padding(img, filter):
    """
    Uses Spatial correlation function to perform filter using the mask
    This works best with the filter size with odd dimensions eg {(3*3), (5,5)}
    """
    filter_height, filter_width = filter.shape
    f_row = filter_height // 2
    f_col = filter_width // 2
    height, width = img.shape
    corr_img = np.zeros(shape=(height + f_row * 2, width + f_col * 2))
    corr_img[f_row:height + f_row, f_col:width + f_col] = img  # Zero padded Image(Boundary filled with zeros)
    img_copy = np.empty(shape=img.shape)
    for i in range(0, height):
        for j in range(0, width):
            snap = corr_img[i: i + filter_height, j: j + filter_width]
            img_copy[i, j] = np.sum(snap * filter)
    return img_copy


def median_filter(img, filter_size=3):
    f_row = filter_size // 2
    f_col = filter_size // 2
    n_rows, n_cols = img.shape
    corr_img = np.zeros(shape=(n_rows + f_row * 2, n_cols + f_col * 2))
    corr_img[f_row:n_rows + f_row, f_col:n_cols + f_col] = img  # Zero padded Image(Boundary filled with zeros)
    img_copy = np.empty(shape=(img.shape))
    for i in range(0, n_rows):
        for j in range(0, n_cols):
            img_copy[i, j] = np.median(corr_img[i: i + f_row * 2 + 1, j: j + f_col * 2 + 1])
    return img_copy


def correlation_with_no_padding(img, filter):
    # Obtain number of rows and columns
    # of the image
    m, n = img.shape

    # Convolve the 3X3 mask over the image
    img_new = np.zeros([m, n])
    filter_height, filter_width = filter.shape

    start_row = filter_height // 2
    start_column = filter_width // 2
    for i in range(start_row, m - start_row):
        for j in range(start_column, n - start_column):
            temp = np.sum(img[i - start_row:i + start_row + 1, j - start_column:j + start_column + 1] * filter)
            img_new[i, j] = temp

    img_new = img_new.astype(np.uint8)
    return img_new


def median_filter_with_no_padding(img, mask_size):
    # Obtain number of rows and columns
    # of the image
    m, n = img.shape

    img_new = np.zeros([m, n])

    start_row = mask_size // 2
    start_column = mask_size // 2
    for i in range(start_row, m - start_row):
        for j in range(start_column, n - start_column):
            temp = np.median(img[i - start_row:i + start_row + 1, j - start_column:j + start_column + 1])
            img_new[i, j] = temp

    img_new = img_new.astype(np.uint8)
    return img_new
