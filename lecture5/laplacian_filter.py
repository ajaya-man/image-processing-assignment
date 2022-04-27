from libraries import *

moon = os.path.join(os.path.dirname(__file__), "../assets/moon.tif")
image = cv2.imread(moon, 0)

cv2.imshow("Original", image)

laplacian_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
laplacian_filtered_image = correlation_filter_with_padding(image, laplacian_filter)

cv2.imshow("Laplacian Filter", laplacian_filtered_image)
cv2.imshow("Sharpened image", image - laplacian_filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
