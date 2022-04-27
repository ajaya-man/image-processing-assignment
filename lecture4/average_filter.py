from libraries import *

mask_size = [3, 5, 9, 15, 35]

letter_a = os.path.join(os.path.dirname(__file__), "../assets/letter_a.tif")
image = cv2.imread(letter_a, 0)

cv2.imshow("Original", image)

for mask in mask_size:
    # Develop Averaging filter mask
    filter = np.ones([mask, mask], dtype=int)
    filter = filter / (mask * mask)
    averaged_image = correlation_with_no_padding(image, filter)
    cv2.imshow(str(mask) + " X " + str(mask) + " avg filter", averaged_image)
    cv2.waitKey(100)

cv2.waitKey(0)
cv2.destroyAllWindows()
