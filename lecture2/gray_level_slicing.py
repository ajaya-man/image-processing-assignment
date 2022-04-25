from libraries import *

# Read original image
forest = os.path.join(os.path.dirname(__file__), "../assets/forest.tif")
image = cv2.imread(forest, 0)

# Read original image
# road = os.path.join(os.path.dirname(__file__), "../assets/road.tif")
# image = cv2.imread(road, 0)

#  Find width and height of image
row, column = image.shape

#  Create a zeros array to store the sliced image
img1 = np.zeros((row, column), dtype='uint8')

#  Specify the min and max range
min_range = 80
max_range = 140

# Loop over the input image and if pixel value lies in desired range set it to 255
# otherwise set it to desired value
for i in range(row):
    for j in range(column):
        if min_range < image[i, j] < max_range:
            img1[i, j] = 255
        else:
            img1[i, j] = image[i - 1, j - 1]

cv2.imshow('Original', image)
cv2.imshow('slicedimage', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
