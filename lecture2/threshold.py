from libraries import *

# Read the image
moon = os.path.join(os.path.dirname(__file__), "../assets/moon.tif")
image = cv2.imread(moon)

threshold = 127
max = 255

threshold_image = (image > threshold).astype('uint8') * max

cv2.imshow("Original", image)
cv2.imshow("Threshold", threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
