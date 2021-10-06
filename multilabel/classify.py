# USAGE
# python classify.py --image tests/test_01.jpg

# import the necessary packages
from configuration import config
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils  
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-m", "--model", required=True,
#	help="path to trained model model")
#ap.add_argument("-l", "--labelbin", required=True,
#	help="path to label binarizer")
ap.add_argument("-i", "--input", required=True,
	help="path to input image/text file of image paths")
args = vars(ap.parse_args())

# determine the input file type, but assume that we're working with
# single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]
# if the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:
    # load the image paths in our testing file
    imagePaths = open(args["input"]).read().strip().split("\n")

# load our object detector and label binarizer from disk
print("[INFO] loading network...")
model = load_model(config.MODEL_PATH)
lb = pickle.loads(open(config.LB_PATH, "rb").read())

total = 0

# loop over the images that we'll be testing using our bounding box
# regression model
for imagePath in imagePaths:

    # load the image
    image = cv2.imread(imagePath)
    output = imutils.resize(image, width=400)
 
    # pre-process the image for classification
    image = cv2.resize(image, (96, 96))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image then find the indexes of the two class
    # labels with the *largest* probability
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]
    idxs = np.argsort(proba)[::-1][:2]

    # loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):
        # build the label and draw the label on the image
        label = "{}: {:.2f}%".format(lb.classes_[j], proba[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25), 
    	    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show the probabilities for each of the individual labels
    for (label, p) in zip(lb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))

    # show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(0)

    # if the `k` key was pressed, write the *original* frame to disk
    # so we can later process it and use it for face recognition
    if key == ord("k"):
        p = "output/" + os.path.basename(imagePath)
        print(p)
        cv2.imwrite(p, output)
        total += 1

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

# print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
