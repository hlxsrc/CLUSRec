# This scripts creates a CNN model using VGGNet to predict classes of images using bouding box regression.
# USAGE
# python train.py --configname <configuration>

# Enable/disable debugging logs (0,1,2,3)
# 0 -> all, 3 -> none
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from cnn.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from imutils import paths
import tensorflow as tf
import seaborn as sns
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# Import configuration file
from configuration import config

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-cn", "--configname", required=True,
   help="Name of desired configuration")
ap.add_argument("-cf", "--configfile", required=False,
   help="Name of configuration file")
args = vars(ap.parse_args())

# Get selected configuration as dictionary
config_dict = config.read_configuration(config_file=args["configfile"],
        config=args["configname"])

# Create output paths
paths_dict = config.create_paths(args["configname"], config_dict)

# Configure output dir
Path(paths_dict["output"]).mkdir(parents=True, exist_ok=True)

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Initialize list of data, class labels, bounding box coordinates
# and images path
data = []
labels = []
bboxes = []
imagePaths = []

# Get image dimensions
IMAGE_DIMS = config_dict["imageDimensions"]

print("[INFO] loading dataset...")

# Loop over all CSV files in the annotations directory
for csvPath in paths.list_files(paths_dict["annotations"], validExts=(".csv")):

    # Load the contents of the current CSV annotations file
    rows = open(csvPath).read().strip().split("\n")

    # Loop over the rows
    for row in rows:

        # Break the row into the filename, bounding box coordinates,
        # and class label
        row = row.split(",")
        (filename, startX, startY, endX, endY, label) = row

        # Derive the path to the input image 
        imagePath = os.path.sep.join([paths_dict["images"], label,
            filename])
        # Load the image (OpenCV format)
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
        image = load_img(imagePath, target_size=(IMAGE_DIMS[1], IMAGE_DIMS[0]))
        image = img_to_array(image)

        # Extract set of class labels from the image path and update the
        # labels list
        # l = label = imagePath.split(os.path.sep)[-2].split("_")
        # labels.append(l)

        # Update our list of data, labels, targets, and filenames
        data.append(image)
        labels.append(label)
        bboxes.append((startX, startY, endX, endY))
        imagePaths.append(imagePath)

# Convert data, class labels, bounding boxes, and image paths to
# NumPy arrays
# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagesPaths = np.array(imagePaths)

print("[INFO] data matrix: {} images ({:.2f}MB)".format(
    len(imagePaths), data.nbytes / (1024 * 1000.0)))

# Binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# If there are only two labels in the dataset, then we need to use
# Keras/TensorFlow's utility function as well
if len(lb.classes_) == 2:
    labels = to_categorical(labels)

# Loop over each of the possible class labels and show them
for (i, label) in enumerate(lb.classes_):
    print("{}. {}".format(i + 1, label))

# Partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
split = train_test_split(data, labels, bboxes, imagePaths, 
        test_size=config_dict["testSplit"], random_state=27)

# Unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths, testPaths) = split[6:]

# Write the testing image paths to disk 
print("[INFO] saving testing image paths...")
f = open(paths_dict["test"], "w")
f.write("\n".join(testPaths))
f.close()

# Initialize the model 
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], numCoordinates=4, numClasses=len(lb.classes_))

# Define a dictionary to set the loss methods -- categorical
# cross-entropy for the class label head and mean absolute error
# for the bounding box head
losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "mean_squared_error",
}

# Define a dictionary that specifies the weights per loss (both the
# class label and bounding box outputs will receive equal weight)
lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

# Initialize the optimizer (SGD is sufficient)
opt = Adam(learning_rate=config_dict["learningRate"], 
        decay=config_dict["learningRate"] / config_dict["epochs"])

# Compile the model 
model.compile(loss=losses, optimizer=opt, metrics=["accuracy"],
        loss_weights=lossWeights)

# Print summary
print(model.summary())
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Construct a dictionary for our target training outputs
trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}

# Construct a second dictionary, this one for our target testing
# outputs
testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

# Train the network
print("[INFO] training network...")
H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=config_dict["batchSize"],
    epochs=config_dict["epochs"],
    verbose=10)

# Save the model to disk
print("[INFO] serializing network...")
model.save(paths_dict["model"], save_format="h5")

# Save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(paths_dict["binarizer"], "wb")
f.write(pickle.dumps(lb))
f.close()

# Evaluate the network
print("\n")
print("[INFO] Evaluating network...")
(boxPreds, labelPreds) = model.predict(testImages, batch_size=config_dict["batchSize"])
print(classification_report(testLabels.argmax(axis=1), labelPreds.argmax(axis=1), target_names=["human", "other"]))

# Plot the total loss, label loss, and bounding box loss
lossNames = ["loss", "class_label_loss", "bounding_box_loss"]
N = np.arange(0, config_dict["epochs"])
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(13, 13))

# Loop over the loss names
for (i, l) in enumerate(lossNames):

    # Plot the loss for both the training and validation data
    title = "Loss for {}".format(l) if l != "loss" else "Total loss"
    ax[i].set_title(title)
    ax[i].set_xlabel("Epoch #")
    ax[i].set_ylabel("Loss")
    ax[i].plot(N, H.history[l], label=l)
    ax[i].plot(N, H.history["val_" + l], label="val_" + l)
    ax[i].legend()

# Save the losses figure and create a new figure for the accuracies
plt.tight_layout()
plt.savefig(paths_dict["loss"])
plt.close()

# Create a new figure for the accuracies
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["class_label_accuracy"],
    label="class_label_train_acc")
plt.plot(N, H.history["val_class_label_accuracy"],
    label="val_class_label_acc")
plt.title("Class Label Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Accuracy")
plt.legend(loc="lower left")

# Save the accuracies plot
plt.savefig(paths_dict["accuracy"])

