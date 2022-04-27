import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy as sp


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


def filter_image_in_frequency(img, h_uv):
    # Preprocessing: To
    u, v = np.indices(img.shape)
    shift_mat = np.power(-1, u + v)  # Matrix of (-1)^(x+y)
    # Multipy the img by (-1)^(x+y)
    shifted_img = img * shift_mat

    # Calculate DFT of the image
    dft_img = sp.fft.fft2(shifted_img, s=h_uv.shape)
    # Perform Multiplication of filter and DFT of the image
    g_uv = dft_img * h_uv
    # Calculate the Inverse Fourier Tranform of the result
    inv_fft_g = sp.fft.ifft2(g_uv)
    # Only Take the Real Values
    inv_fft_g_real = np.real(inv_fft_g)
    # Crop the image to the original image size
    result = inv_fft_g_real[:img.shape[0], :img.shape[1]]
    # Undo the shifting
    return result * shift_mat


def ideal_freq_low_pass_filter(size, cutoff_freq):
    ideal_lp_filter = np.zeros(shape=size, dtype=np.int)
    M_div2, N_div2 = size[0] / 2, size[1] / 2
    u, v = np.indices(size, dtype=np.int)

    response = np.sqrt(np.power(u - M_div2, 2) + np.power(v - N_div2, 2))
    ideal_lp_filter[response < cutoff_freq] = 1
    return ideal_lp_filter


def ideal_freq_high_pass_filter(size, cutoff_freq):
    ideal_hp_filter = 1 - ideal_freq_low_pass_filter(size, cutoff_freq)
    return ideal_hp_filter


def gaussian_low_pass_filter(size, cutoff_freq):
    u, v = np.indices(size, dtype=np.int)
    M_div2, N_div2 = size[0] / 2, size[1] / 2
    lp_filter = np.exp((-2 / np.power(cutoff_freq, 2)) * (np.power(u - M_div2, 2) + np.power(v - N_div2, 2)))
    return lp_filter


def gaussian_high_pass_filter(size, cutoff_freq):
    gauss_hp_filter = 1 - gaussian_low_pass_filter(size, cutoff_freq)
    return gauss_hp_filter


def butterworth_low_pass_filter(size, cutoff_freq, order):
    u, v = np.indices(size, dtype=np.int)
    M_div2, N_div2 = size[0] / 2, size[1] / 2

    d_uv2 = np.power(u - M_div2, 2) + np.power(v - N_div2, 2)  # Distance Squared
    d_uv_div_d0 = d_uv2 / np.power(cutoff_freq, 2)  # Distance Squared/D0^2
    divisor = 1 + np.power(d_uv_div_d0, order)  # 1 + (Distance Squared/D0^2)^n

    return 1 / divisor  # 1/(1 + (Distance Squared/D0^2)^n)


def butterworth_high_pass_filter(size, cutoff_freq, order):
    butterworth_lp_filter = 1 - butterworth_low_pass_filter(size, cutoff_freq, order)
    return butterworth_lp_filter
