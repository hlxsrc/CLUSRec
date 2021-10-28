# This scripts creates a CNN model using VGGNet to predict 
# the class of images using bouding box regression
# USAGE
# python train.py --config config

# Enable/disable debugging logs (0,1,2,3)
# 0 -> all, 3 -> none
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from cnn.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from pathlib import Path
from imutils import paths
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib
import argparse
import random
import pickle
import cv2

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True,
   help="Path to configuration file")
args = vars(ap.parse_args())

# Handle configuration file (simple way)
if not args["config"]:
    config_file = "config"
else:
    config_file = args["config"]

# Import configuration file
cf = getattr(__import__("configuration", fromlist=[config_file]), config_file)

# Configure output dir
Path(cf.BASE_OUTPUT).mkdir(parents=True, exist_ok=True)

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Initialize data (images), targets (bounding boxes) and filenames (images)
data = []
targets = []
imageNames = []

# Loop over all CSV files in the annotations directory
print("[INFO] loading dataset...")
for csvPath in paths.list_files(cf.ANNOTS_PATH, validExts=(".csv")):

    # Load the contents of the current CSV annotations file
    rows = open(csvPath).read().strip().split("\n")

    # Loop over the rows
    for row in rows:

        # Break the row into the filename and bounding box coordinates
        row = row.split(",")
        (imageName, startX, startY, endX, endY, label) = row

        # Derive the path to the input image
        imagePath = os.path.sep.join([cf.IMAGES_PATH, label, 
            imageName])
        # Load the image (in OpenCV format)
        image = cv2.imread(imagePath)
        # Grab its dimensions
        (h, w) = image.shape[:2]

        # Scale the bounding box coordinates relative to the spatial
        # dimensions of the input image
        startX = float(startX) / w
        startY = float(startY) / h
        endX = float(endX) / w
        endY = float(endY) / h

        # Load the image and preprocess it
        image = load_img(imagePath, target_size=(cf.IMAGE_DIMS[1], cf.IMAGE_DIMS[0]))
        image = img_to_array(image)

        # Update our list of data, targets, and filenames
        data.append(image)
        targets.append((startX, startY, endX, endY))
        imageNames.append(imageName)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# Partition the data into training and testing splits 
split = train_test_split(data, targets,
    imageNames, test_size=cf.TEST_SPLIT, random_state=42)

# Unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainPaths, testPaths) = split[4:]

# Write testing image paths to disk
print("[INFO] saving testing image paths...")
f = open(cf.TEST_PATH, "w")
f.write("\n".join(testPaths))
f.close()

# Initialize the model
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
    width=cf.IMAGE_DIMS[1], height=cf.IMAGE_DIMS[0],
    depth=cf.IMAGE_DIMS[2], numCoordinates=4)

# Initialize the optimizer (SGD)
#opt = Adam(lr=cf.LR, decay=cf.LR / cf.EPOCHS)
opt = Adam(lr=cf.LR)

# Compile the model using categorical cross-entropy
# model.compile(loss="categorical_crossentropy", optimizer=opt,
#    metrics=["accuracy"])
model.compile(loss="mse", optimizer=opt,
    metrics=["accuracy"])

# Print summary of the network
model.summary()

# Train the network
print("[INFO] training bounding box regressor...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=cf.BS,
    epochs=cf.EPOCHS, 
    verbose=10)

# Save the model to disk
print("[INFO] serializing network...")
model.save(cf.MODEL_PATH, save_format="h5")

# Plot the model training history
N = cf.EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(cf.PLOT_PATH)
