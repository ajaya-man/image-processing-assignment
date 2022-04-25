# Logarithmic transformation

from libraries import *

point_light = os.path.join(os.path.dirname(__file__), "../assets/point_light.tif")
img = cv2.imread(point_light)

# constant c of log correction equation
c = 1

lookUpTable = np.empty((1, 256), np.uint8)
for i in range(256):
    lookUpTable[0, i] = np.clip(c * np.log10(1 + i / 255) * 255, 0, 255)

log_transformed = cv2.LUT(img, lookUpTable)

cv2.imshow("Original", img)
cv2.imshow("Transformed", log_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
