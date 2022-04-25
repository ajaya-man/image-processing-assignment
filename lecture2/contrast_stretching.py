from libraries import *


# Find line equations by calculating slopes
def Contrast_stretch(p, r1, s1, r2, s2):
    if 0 <= p <= r1:
        equation = (s1 / r1) * p
    elif r1 < p <= r2:
        equation = ((s2 - s1) / (r2 - r1)) * (p - r1) + s1
    else:
        equation = ((255 - s2) / (255 - r2)) * (p - r2) + s2
    return equation


# Read original image
seed = os.path.join(os.path.dirname(__file__), "../assets/seed.tif")
image = cv2.imread(seed)

# Initialize range
r1_img = image.min()
s1_img = 0
r2_img = image.max()
s2_img = 255

pixelVal_vec = np.vectorize(Contrast_stretch)

# Contrast stretching
contrast = pixelVal_vec(image, r1_img, s1_img, r2_img, s2_img)

cv2.imshow("Original", image)
cv2.imshow('seed contrast', contrast)
cv2.waitKey(0)
cv2.destroyAllWindows()
