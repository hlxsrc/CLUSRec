# USAGE
# python train.py

# Enable/disable debugging logs (0,1,2,3)
# 0 -> all, 3 -> none
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import confiuration file
from configuration import config

# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from cnn.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import tensorflow as tf
import seaborn as sns
import numpy as np
import matplotlib
import argparse
import random
import pickle
import cv2

# Construct the argument parse
#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#   help="path to input dataset (directory of images)")
#args = vars(ap.parse_args())

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Load the contents of the CSV annotations file
print("[INFO] loading dataset...")
rows = open(config.ANNOTS_PATH).read().strip().split("\n")

# Initialize data (images), targets (bounding boxes) and filenames (images)
data = []
targets = []
imageNames = []

# Loop over the rows
for row in rows:

    # Break the row into the filename and bounding box coordinates
    row = row.split(",")
    (imageName, startX, startY, endX, endY) = row

    # Derive the path to the input image
    imagePath = os.path.sep.join([config.IMAGES_PATH, imageName])
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
    image = load_img(imagePath, target_size=(config.IMAGE_DIMS[1], config.IMAGE_DIMS[0]))
    image = img_to_array(image)

    # Update our list of data, targets, and filenames
    data.append(image)
    targets.append((startX, startY, endX, endY))
    imageNames.append(imageName)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float32") / 255.0
targets = np.array(targets, dtype="float32")

# Partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, targets,
    imageNames, test_size=0.2, random_state=42)

# Unpack the data split
(trainImages, testImages) = split[:2]
(trainTargets, testTargets) = split[2:4]
(trainPaths, testPaths) = split[4:]

# Write testing image paths to disk
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

# Construct the image generator for data augmentation
#aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
#    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
#    horizontal_flip=True, fill_mode="nearest")

# Initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
# since it is single label we don't need number of classes
# neither the finalAct
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
    width=config.IMAGE_DIMS[1], height=config.IMAGE_DIMS[0],
    depth=config.IMAGE_DIMS[2], classes=1,
    finalAct="sigmoid")

# Initialize the optimizer (SGD)
opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR / config.EPOCHS)

# Compile the model using categorical cross-entropy
#model.compile(loss="categorical_crossentropy", optimizer=opt,
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
    batch_size=config.BS,
    epochs=config.EPOCHS, 
    verbose=1)

# Save the model to disk
print("[INFO] serializing network...")
model.save(config.MODEL_PATH, save_format="h5")

# Plot the model training history
N = config.EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.title("Bounding Box Regression Loss on Training Set")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
