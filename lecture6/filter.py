import cv2

from libraries import *

letter_a = os.path.join(os.path.dirname(__file__), "../assets/letter_a.tif")
image = cv2.imread(letter_a, 0)

fft_size = (2048, 2048)

ideal_lp_filter = ideal_freq_low_pass_filter(size=fft_size, cutoff_freq=512)
cv2.imshow("Ideal LP filter", ideal_lp_filter.astype('uint8'))
cv2.imshow("Ideal LP filter image", filter_image_in_frequency(image, ideal_lp_filter))
cv2.waitKey(100)

gauss_lp_filter = gaussian_low_pass_filter(fft_size, 1000)
cv2.imshow("Gaussian LP filter", gauss_lp_filter)
cv2.imshow("Gaussian LP filter image", filter_image_in_frequency(image, gauss_lp_filter))
cv2.waitKey(100)

butterworth_lp_filter = butterworth_low_pass_filter(fft_size, 200, order=20)
cv2.imshow("Butterworth LP filter", butterworth_lp_filter)
cv2.imshow("Butterworth LP filter image", filter_image_in_frequency(image, butterworth_lp_filter))
cv2.waitKey(100)

ideal_hp_filter = ideal_freq_high_pass_filter(size=fft_size, cutoff_freq=512)
cv2.imshow("Ideal HP filter", ideal_hp_filter.astype('uint8'))
cv2.imshow("Ideal HP filter image", filter_image_in_frequency(image, ideal_hp_filter))
cv2.waitKey(100)

gauss_hp_filter = gaussian_high_pass_filter(fft_size, 1000)
cv2.imshow("Gaussian HP filter", gauss_hp_filter)
cv2.imshow("Gaussian HP filter image", filter_image_in_frequency(image, gauss_hp_filter))
cv2.waitKey(100)

butterworth_hp_filter = butterworth_high_pass_filter(fft_size, 200, order=10)
cv2.imshow("Butterworth HP filter", butterworth_hp_filter)
cv2.imshow("Butterworth HP filter image", filter_image_in_frequency(image, butterworth_hp_filter))
cv2.waitKey(100)

cv2.waitKey(0)
cv2.destroyAllWindows()
