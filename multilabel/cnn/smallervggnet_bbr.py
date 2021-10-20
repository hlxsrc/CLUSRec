# Based on the VGGNet network introduced by Simonyan and Zisserman 
# in Very Deep Convolutional Networks for Large Scale Image Recognition.

# VGGNet-like architectures are characterized by:
# - Using only 3×3 convolutional layers stacked on top of each other 
#   in increasing depth
# - Fully-connected layers at the end of the network prior to 
#   a softmax classifier
# - Reducing volume size by max pooling

# As seen on Pyimagesearch

# import the necessary packages
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

# create SmallerVGGNet class
class SmallerVGGNet:

    @staticmethod
    def build_bb_branch(inputShape, numCoordinates, chanDim=-1):

        # CONV => RELU => POOL
        # CONV layer with 32 filters with a 3x3 kernel
        bboxHead = Conv2D(32, (3, 3), padding="same")(inputShape)
        # RELU activation
        bboxHead = Activation("relu")(bboxHead)
        # apply batch normalization
        bboxHead = BatchNormalization(axis=chanDim)(bboxHead)
        # apply max pooling
        bboxHead = MaxPooling2D(pool_size=(3, 3))(bboxHead)
        # apply dropout to reduce overfitting
        bboxHead = Dropout(0.25)(bboxHead)

        # Two sets of (CONV => RELU) * 2 => POOL
        # reduce spatial size but increasing depth

        # (CONV => RELU) * 2 => POOL
        bboxHead = Conv2D(64, (3, 3), padding="same")(bboxHead)
        bboxHead = Activation("relu")(bboxHead)
        bboxHead = BatchNormalization(axis=chanDim)(bboxHead)
        bboxHead = Conv2D(64, (3, 3), padding="same")(bboxHead)
        bboxHead = Activation("relu")(bboxHead)
        bboxHead = BatchNormalization(axis=chanDim)(bboxHead)
        bboxHead = MaxPooling2D(pool_size=(2, 2))(bboxHead)
        bboxHead = Dropout(0.25)(bboxHead)

        # (CONV => RELU) * 2 => POOL
        bboxHead = Conv2D(128, (3, 3), padding="same")(bboxHead)
        bboxHead = Activation("relu")(bboxHead)
        bboxHead = BatchNormalization(axis=chanDim)(bboxHead)
        bboxHead = Conv2D(128, (3, 3), padding="same")(bboxHead)
        bboxHead = Activation("relu")(bboxHead)
        bboxHead = BatchNormalization(axis=chanDim)(bboxHead)
        bboxHead = MaxPooling2D(pool_size=(2, 2))(bboxHead)
        bboxHead = Dropout(0.25)(bboxHead)

        # Construct a fully-connected layer
        # to output the predicted bounding box coordinates
        bboxHead = Flatten()(bboxHead)
        bboxHead = Dense(128)(bboxHead)
        bboxHead = Activation("relu")(bboxHead)
        bboxHead = Dense(64)(bboxHead)
        bboxHead = Activation("relu")(bboxHead)
        bboxHead = Dense(32)(bboxHead)
        bboxHead = Activation("relu")(bboxHead)
        bboxHead = Dense(numCoordinates)(bboxHead)
        bboxHead = Activation("sigmoid", name="bounding_box")(bboxHead)

        # Return the bounding box prediction sub-network
        return bboxHead 

    @staticmethod
    def build_classes_branch(inputShape, numClasses, chanDim=-1):

        # CONV => RELU => POOL
        # CONV layer with 32 filters with a 3x3 kernel
        softmaxHead = Conv2D(32, (3, 3), padding="same")(inputShape)
        # RELU activation
        softmaxHead = Activation("relu")(softmaxHead)
        # apply batch normalization
        softmaxHead = BatchNormalization(axis=chanDim)(softmaxHead)
        # apply max pooling
        softmaxHead = MaxPooling2D(pool_size=(3, 3))(softmaxHead)
        # apply dropout to reduce overfitting
        softmaxHead = Dropout(0.25)(softmaxHead)

        # Two sets of (CONV => RELU) * 2 => POOL
        # reduce spatial size but increasing depth

        # (CONV => RELU) * 2 => POOL
        softmaxHead = Conv2D(64, (3, 3), padding="same")(softmaxHead)
        softmaxHead = Activation("relu")(softmaxHead)
        softmaxHead = BatchNormalization(axis=chanDim)(softmaxHead)
        softmaxHead = Conv2D(64, (3, 3), padding="same")(softmaxHead)
        softmaxHead = Activation("relu")(softmaxHead)
        softmaxHead = BatchNormalization(axis=chanDim)(softmaxHead)
        softmaxHead = MaxPooling2D(pool_size=(2, 2))(softmaxHead)
        softmaxHead = Dropout(0.25)(softmaxHead)

        # (CONV => RELU) * 2 => POOL
        softmaxHead = Conv2D(128, (3, 3), padding="same")(softmaxHead)
        softmaxHead = Activation("relu")(softmaxHead)
        softmaxHead = BatchNormalization(axis=chanDim)(softmaxHead)
        softmaxHead = Conv2D(128, (3, 3), padding="same")(softmaxHead)
        softmaxHead = Activation("relu")(softmaxHead)
        softmaxHead = BatchNormalization(axis=chanDim)(softmaxHead)
        softmaxHead = MaxPooling2D(pool_size=(2, 2))(softmaxHead)
        softmaxHead = Dropout(0.25)(softmaxHead)

        # Construct a second fully-connected layer head
        # to predict the class label
        softmaxHead = Flatten()(softmaxHead)
        softmaxHead = Dense(512)(softmaxHead)
        softmaxHead = Activation("relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(512)(softmaxHead)
        softmaxHead = Activation("relu")(softmaxHead)
        softmaxHead = Dropout(0.5)(softmaxHead)
        softmaxHead = Dense(numClasses)(softmaxHead)
        softmaxHead = Activation("softmax", name="class_label")(softmaxHead)

        return softmaxHead

    # Define build function 
    # Receives: width, height, depth, classes and
    #   the activation function
    @staticmethod
    def build(width, height, depth, numCoordinates, numClasses):
                
        # Initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        chanDim = -1

        # If we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Construct both the "bb" and "classes" sub-networks
        inputs = Input(shape=inputShape)
        bbBranch = SmallerVGGNet.build_bb_branch(inputs,
            numCoordinates, chanDim=chanDim)
        classesBranch = SmallerVGGNet.build_classes_branch(inputs,
            numClasses, chanDim=chanDim)

        # Create the model using our input (the batch of images) and
        # two separate outputs -- one for the bounding box
        # branch and another for the classes branch, respectively
        model = Model(
            inputs=inputs,
            outputs=[bbBranch, classesBranch],
            name="krbynet")

        # Return the constructed network architecture
        return model
