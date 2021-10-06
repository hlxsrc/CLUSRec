# USAGE
# python train.py --dataset dataset --model human.model --labelbin human.pickle

# set the matplotlib backend so figures can be saved in the background
import matplotlib
#matplotlib.use("Agg")

# import confiuration file
from configuration import config

# import the necessary packages
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
from imutils import paths
import tensorflow as tf
import seaborn as sns
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required=True,
#	help="path to input dataset (directory of images)")
#ap.add_argument("-m", "--model", required=True,
#	help="path to output model")
#ap.add_argument("-l", "--labelbin", required=True,
#	help="path to output label binarizer")
#ap.add_argument("-p", "--plot", type=str, default="plot.png",
#	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# disable eager execution
tf.compat.v1.disable_eager_execution()

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
print("[INFO] path:", config.IMAGES_PATH)
imagePaths = sorted(list(paths.list_images(config.IMAGES_PATH)))
random.seed(42)
random.shuffle(imagePaths)

# initialize the data and labels
data = []
imageNames = []
labels = []

# loop over the input images
for imagePath in imagePaths:
	
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (config.IMAGE_DIMS[1], config.IMAGE_DIMS[0]))
    image = img_to_array(image)
    data.append(image)

    # append image path (name)
    imageNames.append(imagePath)

    # extract set of class labels from the image path and update the
    # labels list
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    labels.append(l)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
imageNames = np.array(imageNames)
labels = np.array(labels)
print("[INFO] data matrix: {} images ({:.2f}MB)".format(
    len(imagePaths), data.nbytes / (1024 * 1000.0)))

# binarize the labels using scikit-learn's special multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
#(trainX, testX, trainY, testY) = train_test_split(data,
#    labels, test_size=0.2, random_state=42)
split = train_test_split(data, labels,
    imageNames, test_size=0.2, random_state=42)

# unpack the data split
(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainPaths, testPaths) = split[4:]

# write the testing image paths to disk so that we can use then
# when evaluating/testing our object detector
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testPaths))
f.close()

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallerVGGNet.build(
	width=config.IMAGE_DIMS[1], height=config.IMAGE_DIMS[0],
	depth=config.IMAGE_DIMS[2], classes=len(mlb.classes_),
	finalAct="sigmoid")

# initialize the optimizer (SGD is sufficient)
opt = Adam(lr=config.INIT_LR, decay=config.INIT_LR / config.EPOCHS)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(
	x=aug.flow(trainImages, trainLabels, batch_size=config.BS),
	validation_data=(testImages, testLabels),
	steps_per_epoch=len(trainImages) // config.BS,
	epochs=config.EPOCHS, verbose=10)

# save the model to disk
print("[INFO] serializing network...")
model.save(config.MODEL_PATH, save_format="h5")

# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open(config.LB_PATH, "wb")
f.write(pickle.dumps(mlb))
f.close()

# Evaluate the network
print("\n")
print("[INFO] Evaluating network...")
predictions = model.predict(testImages, batch_size=32)
print(classification_report(testLabels.argmax(axis=1), predictions.argmax(axis=1), target_names=["human", "other"]))

# Multilabel Confusion Matrix
print("\n")
print("[INFO] Multilabel Confusion Matrix...")
cm = multilabel_confusion_matrix(testLabels.argmax(axis=1), predictions.argmax(axis=1))
print(cm)

# Confusion Matrix
print("\n")
print("[INFO] Confusion Matrix...")
cm = confusion_matrix(testLabels.argmax(axis=1), predictions.argmax(axis=1))
print(cm)

ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g', cmap='Greens')  # annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['human', 'other'])
ax.yaxis.set_ticklabels(['human', 'other'])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = config.EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(config.PLOT_PATH)
plt.show()
