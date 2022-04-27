from libraries import *

noisy_pcb = os.path.join(os.path.dirname(__file__), "../assets/noisy_pcb.tif")
image = cv2.imread(noisy_pcb, 0)

cv2.imshow("Original", image)

filter = np.ones([3, 3], dtype=int)
filter = filter / (3 * 3)
averaged_image = correlation_with_no_padding(image, filter)
cv2.imshow("Averaged filtered image", averaged_image)
cv2.waitKey(100)

median_filtered_image = median_filter_with_no_padding(image, mask_size=5)
# median_filtered_image = cv2.medianBlur(image, 5)
cv2.imshow("Median filtered image", median_filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
