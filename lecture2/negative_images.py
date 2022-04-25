from libraries import *

# Point Processing Example: Negative Images

# Breast Mammogram example
# Read original image
breast_mammogram = os.path.join(os.path.dirname(__file__), "../assets/breast_mammogram.tif")
image = cv2.imread(breast_mammogram)

# To find the maximum grey level value in the image
L = image.max()

# Maximum grey level value minus the original image gives the negative image
negative = L - image

# Open the images in two windows, press any key to close the image windows
cv2.imshow('original', image)
cv2.imshow('negative', negative)
cv2.waitKey(0)
cv2.destroyAllWindows()