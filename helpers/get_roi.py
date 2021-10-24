# Python script to extract the roi of an image 
# given the coordinates in a csv file
# This scrips read CSV files using the following format
#   <image_name>,<start_x>,<start_y>,<end_x>,<end_y>,<label>
# USAGE:
#   python get_roi.py

# Import the necessary packages 
from pathlib import Path
from imutils import paths
import imutils
import argparse
import cv2
import os

# Define the base path to the input dataset
DATASET = "test"
BASE_PATH = "../../datasets/" + DATASET

# Add dir to complete base path
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

# Configure output dir
BASE_OUTPUT = "output"
Path(BASE_OUTPUT).mkdir(parents=True, exist_ok=True)

print("[INFO] getting region of interest...")

# Loop over all CSV files in the annotations directory
for csvPath in paths.list_files(ANNOTS_PATH, validExts=(".csv")):

    # Load the contents of the current CSV annotations file
    rows = open(csvPath).read().strip().split("\n")

    # Loop over the rows
    for row in rows:

        # Break the row into the filename, bounding box coordinates,
        # and class label
        row = row.split(",")
        (filename, startX, startY, endX, endY, label) = row
        
        # Derive the path to the input image 
        imagePath = os.path.sep.join([IMAGES_PATH, label,
            filename])

        print("\n")
        print("[INFO] loading image: ", filename)
        print("[INFO]  image path: ", imagePath)

        # Load the image
        image = cv2.imread(imagePath)

        # Extract roi 
        #roi = image[int(startX):int(endX), int(startY):int(endY)]
        roi = image[int(startY):int(endY), int(startX):int(endX)]

        # Show the region of interest
        cv2.imshow("ROI", roi)
        cv2.waitKey(0)
        
        outputPath = BASE_OUTPUT + "/" + filename
        print("[INFO] saving image as: ", outputPath)
        cv2.imwrite(outputPath, roi)
