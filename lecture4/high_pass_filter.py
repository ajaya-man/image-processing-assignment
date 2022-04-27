from libraries import *

breast_mammogram = os.path.join(os.path.dirname(__file__), "../assets/breast_mammogram.tif")
image = cv2.imread(breast_mammogram, 0)

cv2.imshow("Original", image)

high_pass_filter = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
high_pass_filtered_image = correlation_filter_with_padding(image, high_pass_filter)

cv2.imshow("High pass filtered image", high_pass_filtered_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
