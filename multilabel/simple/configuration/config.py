# Import the necessary packages
import os

# Initialize variables
# TEST counter
TEST = 1
# DATASET name
DATASET = "roi_v2"
# IMAGE_DIMS -> image dimesions (pre-processing)
IMAGE_DIMS = (96, 96, 3)
# EPOCHS -> in order to learn patterns using backpropagation (100 or 125)
EPOCHS = 100
# LR -> initial learning rate
LR = 1e-3
# BS -> batch size
BS = 32
# Size of test in train_test_split
TEST_SPLIT = 0.2

# Define the base path to the input dataset
BASE_PATH = "../../../datasets/" + DATASET

# Add dir to complete base path
IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

# Define the path to the base output directory
BASE_OUTPUT = "output/" + DATASET

# Create string to store name with the next notation
# NOTATION: TEST # + DESCRIPTION + DATASET + IMAGE_DIMENSIONS + EPOCHS +
#           LEARNING RATE + BATCH SIZE + TEST SPLIT
model_path = str(TEST) + "_model_" + DATASET + "_" + str(IMAGE_DIMS[0]) + "_" + str(EPOCHS) + "_" + str(LR) + "_" + str(BS) + "_" + str(int(TEST_SPLIT*100)) + ".h5"
lbin_path = str(TEST) + "_lbin_" + DATASET + "_" + str(IMAGE_DIMS[0]) + "_" + str(EPOCHS) + "_" + str(LR) + "_" + str(BS) + "_" + str(int(TEST_SPLIT*100)) + ".pickle"
plot_path = str(TEST) + "_plot_" + DATASET + "_" + str(IMAGE_DIMS[0]) + "_" + str(EPOCHS) + "_" + str(LR) + "_" + str(BS) + "_" + str(int(TEST_SPLIT*100)) + ".png"
cmat_path = str(TEST) + "_cmat_" + DATASET + "_" + str(IMAGE_DIMS[0]) + "_" + str(EPOCHS) + "_" + str(LR) + "_" + str(BS) + "_" + str(int(TEST_SPLIT*100)) + ".png"
accs_path = str(TEST) + "_accs_" + DATASET + "_" + str(IMAGE_DIMS[0]) + "_" + str(EPOCHS) + "_" + str(LR) + "_" + str(BS) + "_" + str(int(TEST_SPLIT*100)) + ".png"
loss_path = str(TEST) + "_loss_" + DATASET + "_" + str(IMAGE_DIMS[0]) + "_" + str(EPOCHS) + "_" + str(LR) + "_" + str(BS) + "_" + str(int(TEST_SPLIT*100)) + ".png"
test_path = str(TEST) + "_test_" + DATASET + "_" + str(IMAGE_DIMS[0]) + "_" + str(EPOCHS) + "_" + str(LR) + "_" + str(BS) + "_" + str(int(TEST_SPLIT*100)) + ".txt"

# Define the path to the:
#     output serialized model, labe binarizer, model training plots,
#     and testing image filenames
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, model_path])
LBIN_PATH = os.path.sep.join([BASE_OUTPUT, lbin_path])
PLOT_PATH = os.path.sep.join([BASE_OUTPUT, plot_path])
CMAT_PATH = os.path.sep.join([BASE_OUTPUT, cmat_path])
ACCS_PATH = os.path.sep.join([BASE_OUTPUT, accs_path])
LOSS_PATH = os.path.sep.join([BASE_OUTPUT, loss_path])
TEST_PATH = os.path.sep.join([BASE_OUTPUT, test_path])
