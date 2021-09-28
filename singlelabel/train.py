# USAGE
# python train.py --dataset dataset --model model.hdf5

# import the necessary packages

# scikit-learn Libraries
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

# cnn libraries
from cnn.smallervggnet import SmallerVGGNet
from cnn.preprocessing import ImageToArrayPreprocessor
from cnn.preprocessing import SimplePreprocessor
from cnn.datasets import SimpleDatasetLoader

# keras library
from tensorflow.keras.optimizers import SGD

# numpy library
import numpy as np

# aux libraries
from imutils import paths
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(learning_rate=0.005)
model = SmallerVGGNet.build(width=32, height=32, depth=3, classes=2, finalAct="sigmoid")

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
#print("[INFO] training network...")
#H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=100, verbose=10)

BS = 32
EPOCHS = 100
# train the network
print("[INFO] training network...")
H = model.fit(
	trainX, trainY, batch_size=BS,
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS, verbose=1)

# save the network to disk
print("[INFO] serializing network...")
model.save(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["human", "other"]))

# Starts cross validation
print("\nCross Validation...")

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
cvscores = []

i = 1

for train, test in kfold.split(data, labels):

    # Fold title
    print("\nFold: ", i)
    i = i + 1

    # initialize the optimizer and model
    print("[INFO] compiling model...")
    opt = SGD(lr=0.005)
    model = SmallerVGGNet.build(width=32, height=32, depth=3, classes=2, finalAct="sigmoid")
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # convert the labels from integers to vectors
    labels = LabelBinarizer().fit_transform(labels)

    # train the network
    print("[INFO] training network...")
    model.fit(data[train], labels[train], batch_size=32, epochs=100, verbose=0)

    # evaluate the model
    print("[INFO] evaluating model...")
    scores = model.evaluate(data[test], labels[test], verbose=0)
    print("Fold accuracy: %.2f%%" % (scores[1] * 100))
    cvscores.append(scores[1] * 100)

print("\nAccuracy %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Confusion Matrix
print("\nConfusion Matrix...")
cm = confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
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
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
