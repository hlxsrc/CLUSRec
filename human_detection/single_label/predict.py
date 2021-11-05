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
import cv2

# Import configuration file
from configuration import config

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
   help="Path to input image/text file of image filenames")
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

# Load our trained bounding box regressor from disk
print("[INFO] loading object detector...")
model = load_model(paths_dict["model"])

# Get image dimensions
IMAGE_DIMS = config_dict["imageDimensions"]

# Loop over the images
for imagePath in imagePaths:

    # Load the input image (in Keras format) from disk 
    image = load_img(imagePath, target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]))
    # Preprocess image scaling the pixel intensities to the range [0, 1]
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make bounding box predictions on the input image
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds
    
    # Load the input image (in OpenCV format) 
    image = cv2.imread(imagePath)
    # Resize it 
    image = imutils.resize(image, width=600)
    # Grab its dimensions
    (h, w) = image.shape[:2]
    
    # Scale the predicted bounding box coordinates based on the image
    # dimensions
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    
    # Draw the predicted bounding box on the image
    cv2.rectangle(image, (startX, startY), (endX, endY),
        (0, 255, 0), 2)
    
    # Show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)

