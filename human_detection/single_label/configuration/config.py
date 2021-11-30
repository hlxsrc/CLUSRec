# Import the necessary packages
import os
import yaml


# Read configuration file
# Receives: 
#   config_file: name of the configuration file
#                default value is config.yaml
#   config: name of specific configuration
#           default is 'base' configuration
# Returns:
#   configuration: content of selected configuration
#                  from the configuration yaml
# This function assumes the configuration file 
# is in the same directory level
def read_configuration(config_file=None, 
        config='base'):

    if config_file is None:
        config_file = 'configuration/config.yaml'

    # Read configuration file
    stream = open(config_file, 'r')
    configurations = yaml.safe_load(stream)

    # Get specific configuration
    configuration = configurations['configuration'][config]

    # Return configuration as dictionary
    return configuration


# Create output paths using selected configuration
# Receives:
#   config_name: name of selected configuration
#   config_dict: configuration dictionary of 
#                  the selected configuration
# Returns:
#   paths: dictionary with output paths
def create_paths(config_name, config_dict):

    # Define the base path to the input dataset
    BASE_PATH = config_dict['dataset']

    # Add dir to complete base path
    IMAGES_PATH = os.path.sep.join([BASE_PATH, "images"])
    ANNOTS_PATH = os.path.sep.join([BASE_PATH, "annotations"])

    # Get basename from base path
    DATASET = os.path.basename(BASE_PATH)

    # Define the path to the base output directory
    BASE_OUTPUT = os.path.sep.join(["output", DATASET])

    # Create string to store name with the next notation
    # NOTATION: TEST # + DESCRIPTION + EXTENSION
    model_path = config_name + "_model.h5"
    lbin_path = config_name + "_lbin.pickle"
    plot_path = config_name + "_plot.png"
    cmat_path = config_name + "_cmat.png"
    accs_path = config_name + "_accs.png"
    loss_path = config_name + "_loss.png"
    test_path = config_name + "_test.txt"

    # Define the path to the:
    #   Trained model, binarized classes, plots
    #   and images used in the test split
    MODEL_PATH = os.path.sep.join([BASE_OUTPUT, model_path])
    LBIN_PATH = os.path.sep.join([BASE_OUTPUT, lbin_path])
    PLOT_PATH = os.path.sep.join([BASE_OUTPUT, plot_path])
    CMAT_PATH = os.path.sep.join([BASE_OUTPUT, cmat_path])
    ACCS_PATH = os.path.sep.join([BASE_OUTPUT, accs_path])
    LOSS_PATH = os.path.sep.join([BASE_OUTPUT, loss_path])
    TEST_PATH = os.path.sep.join([BASE_OUTPUT, test_path])

    # Create dictionary with final paths
    paths = {
            "output": BASE_OUTPUT,
            "images": IMAGES_PATH,
            "annotations": ANNOTS_PATH,
            "model": MODEL_PATH,
            "binarizer": LBIN_PATH,
            "plot": PLOT_PATH,
            "matrix": CMAT_PATH,
            "accuracy": ACCS_PATH,
            "loss": LOSS_PATH,
            "test": TEST_PATH
            }

    return paths
