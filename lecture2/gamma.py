from libraries import *

# Read the image
fractured_spine = os.path.join(os.path.dirname(__file__), "../assets/fractured_spine.tif")
spine_image = cv2.imread(fractured_spine)

aerial = os.path.join(os.path.dirname(__file__), "../assets/washed_out_aerial_image.tif")
aerial_image = cv2.imread(aerial)

cv2.imshow("Original image", spine_image)
for gamma in [0.3, 0.4, 0.6, 1, 1.5, 1.8]:
    # Apply gamma correction.
    gamma_transformation = np.array(255 * (spine_image / 255) ** gamma, dtype='uint8')
    cv2.imshow('gamma_transformed' + str(gamma) + '.jpg', gamma_transformation)

cv2.imshow("Original aerial image", aerial_image)
for gamma in [3.0, 4.0, 5.0]:
    # Apply gamma correction.
    gamma_transformation = np.array(255 * (aerial_image / 255) ** gamma, dtype='uint8')
    cv2.imshow('gamma_transformed aerial image ' + str(gamma), gamma_transformation)

cv2.waitKey(0)
cv2.destroyAllWindows()
