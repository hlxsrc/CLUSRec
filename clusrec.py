# USAGE:
# python clusrec.py --input path/to/input --config config

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

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
        help="Path to input image/text file of image paths")
ap.add_argument("-hm", "--humanmodel", required=True,
        help="Path to human model")
ap.add_argument("-hl", "--humanlbin", required=True,
        help="Path to human label binarizer")
ap.add_argument("-cm", "--clothesmodel", required=True,
        help="Path to clothes model")
ap.add_argument("-cl", "--clotheslbin", required=True,
        help="Path to clothes label binarizer")
args = vars(ap.parse_args())

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
        imagePaths.append(str(f))

# Create variables to store input paths
HUMAN_MODEL_PATH = args["humanmodel"]
HUMAN_LBIN_PATH = args["humanlbin"]
CLOTHES_MODEL_PATH = args["clothesmodel"]
CLOTHES_LBIN_PATH = args["clotheslbin"]

# Store images dimension of each model
HUMAN_IMG_DIM = (224, 224, 3)
CLOTHES_IMG_DIM = (96, 96, 3)

# Load the human body object detector and label binarizer from disk
print("[INFO] Loading human body object detector...")
human_model = load_model(HUMAN_MODEL_PATH)
human_lb = pickle.loads(open(HUMAN_LBIN_PATH, "rb").read())

# Load the clothes object detector and label binarizer from disk
print("[INFO] Loading clothes object detector...")
clothes_model = load_model(CLOTHES_MODEL_PATH)
clothes_lb = pickle.loads(open(CLOTHES_LBIN_PATH, "rb").read())

# Loop over the images 
for imagePath in imagePaths:

    # Load the input image (OpenCV format)
    image = cv2.imread(imagePath)
    # Resize the image
    image = imutils.resize(image, width=600)
    # Grab image dimensions
    (h, w) = image.shape[:2]

    # PREDICT HUMAN SILHOUTTE
    print("[INFO] Detecting human silhoutte...")

    # Load the input image (Keras format)
    print("[INFO] image: ", imagePath)
    image_k = load_img(imagePath, 
            target_size=(HUMAN_IMG_DIM[1], HUMAN_IMG_DIM[0]))
    # Preprocess image scaling the pixel intensities to the range [0, 1]
    image_k = img_to_array(image_k) / 255.0
    image_k = np.expand_dims(image_k, axis=0)

    # Predict the bounding box of the object along with the class
    # label
    (boxPreds, labelPreds) = human_model.predict(image_k)
    (startX, startY, endX, endY) = boxPreds[0]

    # Determine the class label with the largest predicted
    # probability
    i = np.argmax(labelPreds, axis=1)
    label = human_lb.classes_[i][0]

    # Get human silhoutte ROI
    if label == 'human':
    
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
        cv2.imshow("Human Silhoutte", image)
        cv2.waitKey(0)

        # Extract the ROI from the image
        roi = image[int(startY):int(endY), int(startX):int(endX)]

        # Show the region of interest
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)

        # PREDICT CLOTHES
        print("[INFO] Detecting clothes...")

        # Get top prediction
        top = roi[int(roi.shape[0]-roi.shape[0]):int(roi.shape[0]*.60)]

        # Show top
        cv2.imshow("Top", top)
        cv2.waitKey(0)

        # Get bottom prediction
        bottom = roi[int(roi.shape[0]*.40):roi.shape[0]]

        # Show bottom
        cv2.imshow("Bottom", bottom)
        cv2.waitKey(0)
