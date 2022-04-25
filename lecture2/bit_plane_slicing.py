from libraries import *

# Read original image
# flower = os.path.join(os.path.dirname(__file__), "../assets/flower.png")
# image = cv2.imread(flower, 0)

dollar = os.path.join(os.path.dirname(__file__), "../assets/dollar.png")
image = cv2.imread(dollar, 0)

cv2.imshow("Original", image)
for bit in range(8):
    bit_image = (image & (1 << bit)).astype('bool').astype('uint8')*255
    cv2.imshow("bit plane slice image, bit: " + str(bit), bit_image)

cv2.waitKey(0)
cv2.destroyAllWindows()