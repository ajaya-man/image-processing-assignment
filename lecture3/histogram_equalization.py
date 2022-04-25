from libraries import *

# Read the image
seed = os.path.join(os.path.dirname(__file__), "../assets/seed.tif")
image = cv2.imread(seed)

# convert our image into a numpy array
img_array = np.asarray(image)

# put pixels in a 1D array by flattening out img array
flat = img_array.flatten()

# show the histogram
plt.hist(flat, bins=50)
plt.title("Histogram Original Image")
plt.show()


# create our own histogram function
def get_histogram(image, bins):
    # array with size of bins, set to zeros
    histogram = np.zeros(bins)

    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1

    # return our final result
    return histogram


# create our cumulative sum function
def cumsum(a):
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)
    return np.array(b)


hist = get_histogram(flat, 256)
plt.plot(hist)
plt.title("Histogram of flattened image")
plt.show()

# execute the fn
cs = cumsum(hist)
# display the result
plt.plot(cs)
plt.title("Histogram of cumulative sum")
plt.show()

# re-normalize cumsum values to be between 0-255
# numerator & denomenator
nj = (cs - cs.min()) * 255
N = cs.max() - cs.min()

# re-normalize the cdf
cs = nj / N
plt.plot(cs)
plt.title("Normalized cumulative sum")
plt.show()

# cast it back to uint8 since we can't use floating point values in images
cs = cs.astype('uint8')
plt.plot(cs)
plt.title("Integer values")
plt.show()

# get the value from cumulative sum for every index in flat, and set that as img_new
img_new = cs[flat]
# we see a much more evenly distributed histogram
plt.hist(img_new, bins=50)
plt.title("Equalized histogram")
plt.show()

# put array back into original shape since we flattened it
img_new = np.reshape(img_new, image.shape)
img_new

# set up side-by-side image display
fig = plt.figure()
fig.set_figheight(15)
fig.set_figwidth(15)

fig.add_subplot(1, 2, 1)
plt.imshow(image, cmap='gray')

# display the new image
fig.add_subplot(1, 2, 2)
plt.imshow(img_new, cmap='gray')

plt.show(block=True)
