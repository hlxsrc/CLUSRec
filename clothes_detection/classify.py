# USAGE
# python classify.py --input tests/test_01.jpg --configname <configuration>

# Enable/disable debugging logs (0,1,2,3)
# 0 -> all, 3 -> none
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils  
import pickle
import cv2
import os

# Import configuration file
from configuration import config

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
   help="Path to input image/text file of image paths")
ap.add_argument("-cn", "--configname", required=True,
   help="Name of desired configuration")
ap.add_argument("-cf", "--configfile", required=False,
   help="Name of configuration file")
args = vars(ap.parse_args())

# Get selected configuration as dictionary
config_dict = config.read_configuration(config=args["configname"])

# Create output paths
paths_dict = config.create_paths(args["configname"], config_dict)

# Determine the input file type
# Assume we are working with single input image
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

# If the file type is a text file
# we need to process multiple images
if "text/plain" == filetype:
    # load the image paths in our testing file
    imagePaths = open(args["input"]).read().strip().split("\n")

# Load our object detector and label binarizer from disk
print("[INFO] loading network...")
model = load_model(paths_dict["model"])
lb = pickle.loads(open(paths_dict["binarizer"], "rb").read())

# Get image dimensions
IMAGE_DIMS = config_dict["imageDimensions"]
total = 0

# Loop over the test images
for imagePath in imagePaths:

    # Load the image
    image = cv2.imread(imagePath)
    output = imutils.resize(image, width=400)
 
    # Pre-process the image for classification
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # Classify the input image 
    print("[INFO] classifying image...")
    proba = model.predict(image)[0]

    # Find the indexes of the two class
    # labels with the largest probability
    idxs = np.argsort(proba)[::-1][:2]

    # Loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):

        # Build the label and draw the label on the image
        label = "{}: {:.2f}%".format(lb.classes_[j], proba[j] * 100)
        cv2.putText(output, label, (10, (i * 30) + 25), 
    	    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the probabilities for each of the individual labels
    for (label, p) in zip(lb.classes_, proba):
        print("{}: {:.2f}%".format(label, p * 100))

    # Show the output image
    cv2.imshow("Output", output)
    key = cv2.waitKey(0)

    # If the 'k' key was pressed, write the output image to disk
    if key == ord("k"):
        p = "output/" + os.path.basename(imagePath)
        print(p)
        cv2.imwrite(p, output)
        total += 1

    # If the 'q' key was pressed, break from the loop
    elif key == ord("q"):
        break

# Print the total faces saved and do a bit of cleanup
print("[INFO] {} images stored".format(total))
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
