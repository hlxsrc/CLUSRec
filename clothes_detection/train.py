# This scripts creates a CNN model using VGGNet to predict classes of images.
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
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MultiLabelBinarizer
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

# Grab the image paths and randomly shuffle them
print("[INFO] path: ", paths_dict["images"])
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(paths_dict["images"])))
random.seed(42)
random.shuffle(imagePaths)

# Initialize the data and labels
data = []
imageNames = []
labels = []
classes = []

# Get image dimensions
IMAGE_DIMS = config_dict["imageDimensions"]

# Loop over the input images
for imagePath in imagePaths:
    
    # Load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    # Pre-process the image
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)
    # Update data list
    data.append(image)

    # Append image path (name)
    imageNames.append(imagePath)

    # Extract set of class labels from the image path 
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    # Update the labels list
    labels.append(l)

# Scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
imageNames = np.array(imageNames)
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
    len(imagePaths), data.nbytes / (1024 * 1000.0)))

# Binarize the labels 
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# Loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
    classes.append(label)
    print("{}. {}".format(i + 1, label))

print(classes)

# Partition the data into training and testing splits
split = train_test_split(data, labels,
    imageNames, test_size=config_dict["testSplit"], random_state=27)

# Unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainPaths, testPaths) = split[4:]

# Write the testing image paths to disk
print("[INFO] saving testing image paths...")
f = open(paths_dict["test"], "w")
f.write("\n".join(testPaths))
f.close()

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# Initialize the model 
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
    width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
    finalAct="sigmoid")

# Initialize the optimizer (SGD is sufficient)
opt = Adam(learning_rate=config_dict["learningRate"], 
        decay=config_dict["learningRate"] / config_dict["epochs"])

# Compile the model
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

# Print summary
print(model.summary())
plot_model(model, to_file='clothes_model_plot.png', show_shapes=True, show_layer_names=True)

# Train the network
print("[INFO] training network...")
H = model.fit(
    x=aug.flow(trainImages, trainLabels, 
        batch_size=config_dict["batchSize"]),
    validation_data=(testImages, testLabels),
    steps_per_epoch=len(trainImages) // config_dict["batchSize"],
    epochs=config_dict["epochs"], verbose=10)

# Save the model to disk
print("[INFO] serializing network...")
model.save(paths_dict["model"], save_format="h5")

# Save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(paths_dict["binarizer"], "wb")
f.write(pickle.dumps(mlb))
f.close()

# Evaluate the network
print("\n")
print("[INFO] Evaluating network...")
predictions = model.predict(testImages, batch_size=config_dict["batchSize"])
print(classification_report(testLabels.argmax(axis=1), 
    predictions.argmax(axis=1), target_names=classes))

# Create Confusion Matrix
print("\n")
print("[INFO] Confusion Matrix...")
cm = confusion_matrix(testLabels.argmax(axis=1), predictions.argmax(axis=1))
print(cm)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Greens')  # annot=True to annotate cells

# Set labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(classes)
ax.yaxis.set_ticklabels(classes)
plt.savefig(paths_dict["matrix"])

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = config_dict["epochs"]
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(paths_dict["plot"])
#plt.show()
