from libraries import *

cameraman = os.path.join(os.path.dirname(__file__), "../assets/cameraman.tif")
image = cv2.imread(cameraman, 0)

cv2.imshow("Original", image)

sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
sobel_x_filtered_image = correlation_filter_with_padding(image, sobel_x)

sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y_filtered_image = correlation_filter_with_padding(image, sobel_y)

cv2.imshow("Sobel x filtered image", sobel_x_filtered_image)
cv2.imshow("Sobel y filtered image", sobel_y_filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
