# USAGE:
# python predict.py --input path/to/input --configname <configuration>

# Enable/disable debugging logs (0,1,2,3)
# 0 -> all, 3 -> none
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import numpy as np
import mimetypes
import argparse
import imutils
import pickle
import cv2

# Import configuration file
from configuration import config

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
        help="path to input image/text file of image paths")
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
filetype = mimetypes.guess_type(args["input"])[0]
imagePaths = [args["input"]]

# If the file type is a text file, then we need to process *multiple*
# images
if "text/plain" == filetype:

    # Load the filenames in our testing file
    filenames = open(args["input"]).read().strip().split("\n")
    # Initialize our list of image paths
    imagePaths = []

    # Loop over the filenames
    for f in filenames:

        # Update our image paths list
        imagePaths.append(f)

# Load the object detector and label binarizer from disk
print("[INFO] loading object detector...")
model = load_model(paths_dict["model"])
lb = pickle.loads(open(paths_dict["binarizer"], "rb").read())

# Get image dimensions
IMAGE_DIMS = config_dict["imageDimensions"]

# Loop over the images 
for imagePath in imagePaths:

    # Load the input image (in Keras format) from disk
    print("[INFO] image: ", imagePath)
    image = load_img(imagePath, target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]))
    # Preprocess image scaling the pixel intensities to the range [0, 1]
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict the bounding box of the object along with the class
    # label
    (boxPreds, labelPreds) = model.predict(image)
    (startX, startY, endX, endY) = boxPreds[0]

    # Determine the class label with the largest predicted
    # probability
    i = np.argmax(labelPreds, axis=1)
    print(labelPreds)
    print(str(labelPreds[0,0] - labelPreds[0,1]))
    label = lb.classes_[i][0]

    # Load the input image (in OpenCV format), resize it such that it
    # fits on our screen, and grab its dimensions
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    
    # Scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    
    # Draw the predicted bounding box and class label on the image
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (0, 255, 0), 2)
    cv2.rectangle(image, (startX, startY), (endX, endY),
        (0, 255, 0), 2)
    
    # Show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
