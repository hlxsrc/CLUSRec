# CLUSRec

CLothes and hUman Silhoutte Recognition

This is the repository containing all the code for my final Uni project / thesis. 

# Project Structure


```
|-- CLUSRec/
|   |-- helpers/
|   |-- clothes_detection/
|   |   |-- cnn/
|   |   |   |-- smallervggnet.py
|   |   |-- configuration/
|   |   |-- classify.py
|   |   |-- train.py
|   |-- human_detection/
|   |   |-- bounding_box_regression/
|   |   |   |-- cnn/
|   |   |   |   |-- smallervggnet.py
|   |   |   |-- configuration/
|   |   |   |-- predict.py
|   |   |   |-- train.py
|   |   |-- single_label/
|   |   |   |-- cnn/
|   |   |   |   |-- smallervggnet.py
|   |   |   |-- configuration/
|   |   |   |-- predict.py
|   |   |   |-- train.py
|   |-- requirements/
|   |   |-- install.sh
|   |   |-- apt.txt
|   |   |-- pip.txt
```

> Work in progress...

# Getting started

## Prerequisites

In order to run the Python scripts it is necessary to install certain packages.

It is recommended to create a Python virtual environment to avoid conflicts between packages.

### Automatic Installation

If you want to install all the packages automatically inside a virtual environment you can use the included script `requirements/install.sh`
The script was tested using Ubuntu 20.04 and Ubuntu 21.04. To use the installation script follow the next steps: 

> This script is still under development, use with caution.

1. Go to the requirements dir:

```sh
cd requirements
```

2. Make the script executable:

```sh
chmod +x install.sh
```

3. Execute the script using the following command:

```sh
. ./install.sh
```

4. The script will run and it will ask for your password in order to use `apt`.

5. At the end the script will have started the virtual environment automatically, if it does not use the following command:

```sh
workon clusrec
```

### Manual Installation

Alternatively you can install the requirements in `apt.txt` and `pip.txt` manually. To install the `apt` requirements listed in `apt.txt`:

1. Update your system

```sh
sudo apt update
```

2. Install the packages using the `apt.txt` file:

```sh
xargs -a apt.txt sudo apt install -y
```

To install the `pip` requirements listed in `pip.txt`:

1. Check if pip is installed

```sh
pip -V
```

2. If pip is not installed use the following commands to install it:

```sh
wget https://bootstrap.pypa.io/get-pip.py
```

and 

```sh
python3 get-pip.py
```

3. Install `virtualenv` and `virtualenvwrapper` using `pip`:

```sh
pip3 install virtualenv virtualenvwrapper
```

or

```sh
$HOME/.local/bin/pip3 install virtualenv virtualenvwrapper
```

4. Add the next lines to your `bash` profile (`~/.bashrc`) according to your installation paths:

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv
source $HOME/.local/bin/virtualenvwrapper.sh 
```

or

```
# virtualenv and virtualenvwrapper   
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
source /usr/local/bin/virtualenvwrapper.sh
```

5. Update your `bash` profile:

```sh
source ~/.bashrc
```

6. Create the CLUSREC virtual environment

```sh
mkvirtualenv clusrec -p python3
```

7. Install the `pip` requirements listed in `pip.txt`

```sh
$HOME/.local/bin/.virtualenvs/clusrec/bin/pip install -r pip.txt
```

# Using CLUSRec

Once the virtual environment was created you can start running the Python scripts included in this repository.

- Make sure the `virtualenv` is activated:

```sh
workon clusrec
```

## Before Starting

All of the Python scripts in this repository were created with the idea of being easy and straightforward to use. For this reason the basic usage is almost the same in every training and classifying script:

```sh
python train.py --configname <configuration>
```

and

```sh
python predict.py --input <path/to/file> --configname <configuration>
```

Where the configuration is stored in a YAML file inside each of the main directories in the `configuration/` directory. 

The configuration file is named `config.yaml` and the content of the file is shown below:

```yaml
configuration:
  base:
    dataset: "/home/<user>/datasets/human"
    imageDimensions: [96, 96, 3]
    epochs: 100
    learningRate: 0.0001
    batchSize: 32
    testSplit: 0.2
```

You can add a new configuration appending the desired configuration inside the file `config.yaml` below the `base` configuration:

```yaml
configuration:
  base:
    dataset: "/home/<user>/datasets/human"
    imageDimensions: [96, 96, 3]
    epochs: 100
    learningRate: 0.0001
    batchSize: 32
    testSplit: 0.2
  a:
    dataset: "/home/<user>/datasets/human"
    imageDimensions: [96, 96, 3]
    epochs: 200
    learningRate: 0.0001
    batchSize: 32
    testSplit: 0.2
```

In this example the new configuration is named `a` and it is using 200 epochs and a learning rate of 0.0001. You can add as many configuration as you want as long as you are using the correct convention which is:

```yaml
configuration:
  ...
  <configuration_name>:
    dataset: <path/to/dataset>
    imageDimensions: <input_image_dimensions>
    epochs: <epochs>
    learningRate: <learning_rate>
    batchSize: <batch_size>
    testSplit: <test_split>
  ...
```

Alternatively you can create a new configuration file and pass the path to new file to the training script with the flag `--configfile`

## Clothes

### Training

Usage:

```sh
python train.py --configname <configuration>
```

Example:

```sh
python train.py --configname base
```

### Classifying

```sh 
python classify.py --input <path/to/file> --configname <configuration>
```

Example:

```sh
python --input tests/test_01.jpg --configname base
```

or

```sh
python --input test.txt --configname base
```

## Human Silhoutte

### Training (With Bounding Box Regression)

Usage: 

```sh
python train.py --configname <configuration> 
```

Example:

```sh
python train.py --configname base
```

### Predicting (With Bounding Box Regression)

Usage:

```sh
python predict.py --input <path/to/file> --configname <configuration>
```

Example:

```sh
python predict.py --input tests/test_01.jpg --configname base
```

or

```sh
python predict.py --input test.txt --configname base
```

### Training (Single Label)

Usage:

```sh
python train.py --configname base
```

Example:

```sh
python train.py --configname base
```

### Predicting (Single Label)

Usage:

```sh
python predict.py --input <path/to/file> --configname <configuration>
```

Example:

```sh
python predict.py --input tests/test_01.jpg --configname base
```

or

```sh
python predict.py --input test.txt --configname base
```

## CLUSRec

This script is the final wrapper to join the created models together.

Usage:

```sh
python clusrec.py -i <path> -hm <file> -hl <file> -cm <file> -cl <file>
```

Example: 

```sh
python clusrec.py \
  -i test.txt \
  -hm human_detection/multi_label/output/human/base_model.h5 \
  -hl human_detection/multi_label/output/human/base_lbin.pickle \
  -cm clothes_detection/output/clothes/base_model.h5 \
  -cl clothes_detection/output/clothes/base_lbin.pickle
```

or

```sh
python clusrec.py \
  -i test.txt \
  -hm human_detection/single_label/output/human/base_model.h5 \
  -hl human_detection/single_label/output/human/base_lbin.pickle \
  -cm clothes_detection/output/clothes/base_model.h5 \
  -cl clothes_detection/output/clothes/base_lbin.pickle
```

