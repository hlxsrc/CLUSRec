# import the necessary packages
import os

# define the base path to the input dataset and then use it to derive
# the path to the images directory
BASE_PATH = "../../datasets/bb"
IMAGES_PATH = os.path.sep.join([BASE_PATH, "human"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "human_single.csv"])

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training plot,
# and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "detector.h5"])
LB_PATH = os.path.sep.join([BASE_OUTPUT, "detector.pickle"])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, "plot.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])
TEST_FILENAMES = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])

# Initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
# EPOCHS -> in order to learn patterns using backpropagation (100 or 125)
EPOCHS = 100
# INIT_LR -> initial learning rate
INIT_LR = 1e-4
# BS -> batch size
BS = 32
# IMAGE_DIMS -> image dimesions
IMAGE_DIMS = (96, 96, 3)
