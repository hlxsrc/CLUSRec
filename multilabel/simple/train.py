# This scripts creates a CNN model using VGGNet to predict classes of images.
# USAGE
# python train.py --config <configuration_file>

# Enable/disable debugging logs (0,1,2,3)
# 0 -> all, 3 -> none
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from cnn.smallervggnet import SmallerVGGNet
import matplotlib.pyplot as plt
import matplotlib
from imutils import paths
import tensorflow as tf
import seaborn as sns
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", required=True,
   help="Path to configuration file")
args = vars(ap.parse_args())

# List to store configuration file name
cf = []

# Get configuration file
if not args["config"]: 
    cf[0] = "config"
else:
    cf[0] = args["config"]

# Import configuration file
from configuration import cf[0]

# Disable eager execution
tf.compat.v1.disable_eager_execution()

# Grab the image paths and randomly shuffle them
print("[INFO] path: ", cf[0].IMAGES_PATH)
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(cf[0].IMAGES_PATH)))
random.seed(42)
random.shuffle(imagePaths)

# Initialize the data and labels
data = []
imageNames = []
labels = []

# Loop over the input images
for imagePath in imagePaths:
	
    # Load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    # Pre-process the image
    image = cv2.resize(image, (cf[0].IMAGE_DIMS[1], cf[0].IMAGE_DIMS[0]))
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
	print("{}. {}".format(i + 1, label))

# Partition the data into training and testing splits
split = train_test_split(data, labels,
    imageNames, test_size=0.2, random_state=42)

# Unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainPaths, testPaths) = split[4:]

# Write the testing image paths to disk
print("[INFO] saving testing image paths...")
f = open(cf[0].TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

# Construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# Initialize the model 
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=cf[0].IMAGE_DIMS[1], height=cf[0].IMAGE_DIMS[0],
	depth=cf[0].IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# Initialize the optimizer (SGD is sufficient)
opt = Adam(lr=cf[0].INIT_LR, decay=cf[0].INIT_LR / cf[0].EPOCHS)

# Compile the model using binary cross-entropy
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# Train the network
print("[INFO] training network...")
H = model.fit(
	x=aug.flow(trainImages, trainLabels, batch_size=cf[0].BS),
	validation_data=(testImages, testLabels),
	steps_per_epoch=len(trainImages) // cf[0].BS,
	epochs=cf[0].EPOCHS, verbose=10)

# Save the model to disk
print("[INFO] serializing network...")
model.save(cf[0].MODEL_PATH, save_format="h5")

# Save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(cf[0].LB_PATH, "wb")
f.write(pickle.dumps(mlb))
f.close()

# Evaluate the network
print("\n")
print("[INFO] Evaluating network...")
predictions = model.predict(testImages, batch_size=32)
print(classification_report(testLabels.argmax(axis=1), predictions.argmax(axis=1), target_names=["human", "other"]))

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
ax.xaxis.set_ticklabels(['human', 'other'])
ax.yaxis.set_ticklabels(['human', 'other'])

# Plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = cf[0].EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(cf[0].PLOT_PATH)
plt.show()
