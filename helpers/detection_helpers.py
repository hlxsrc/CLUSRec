# import the necessary packages
import imutils

# Receives:
#   image - input image (loop with it) 
#   step - step size (pixels to skip)
#   ws - windows size (width and height of the window)
def sliding_window(image, step, ws):

    # slide a window across the image
    # looop over rows
    for y in range(0, image.shape[0] - ws[1], step):
        # loop over columns
        for x in range(0, image.shape[1] - ws[0], step):

            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])

# Receives: 
#   image - input image (generate multi-scale representations)
#   scale - scale factor (how much the image is resized at each layer)
#   minSize - minimum size (controls min size of output image)
def image_pyramid(image, scale=1.5, minSize=(224, 224)):

    # yield the original image
    yield image

    # keep looping over the image pyramid
    while True:

        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image
