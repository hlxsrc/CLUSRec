# CLUSRec

CLothes and hUman Silhoutte Recognition

This is the repository containing all the code for my final Uni project / thesis. 

# Project Structure


```
|-- CLUSRec/
|   |-- helpers/
|   |-- multilabel/
|   |   |-- simple/
|   |   |   |-- cnn/
|   |   |   |   |-- smallervggnet.py
|   |   |   |-- configuration/
|   |   |   |-- predict.py
|   |   |   |-- train.py
|   |   |-- bounding_box_regression/
|   |   |   |-- cnn/
|   |   |   |   |-- smallervggnet.py
|   |   |   |-- configuration/
|   |   |   |-- predict.py
|   |   |   |-- train.py
|   |-- singlelabel/
|   |   |-- cnn/
|   |   |   |-- smallervggnet.py
|   |   |-- predict.py
|   |   |-- train.py
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
The script was tested using Ubuntu 20.04 and Ubuntu 21.04.
To use the installation script follow the next steps

> This script is still under development, use with caution.

1. Go to the requirements dir:

`cd requirements`

2. Make the script executable:

`chmod +x install.sh`

3. Execute the script using the following command:

`. ./install.sh`

4. The script will run and it will ask for your password in order to use `apt`.

5. At the end the script will have started the virtual environment automatically, if it does not use the following command:

`workon clusrec`

### Manual Installation

Alternatively you can install the requirements in `apt.txt` and `pip.txt` manually. 

To install the `apt` requirements listed in `apt.txt`:

1. Update your system

`sudo apt update`

2. Install the packages using the `apt.txt` file:

`xargs -a apt.txt sudo apt install -y`

To install the `pip` requirements listed in `pip.txt`:

1. Check if pip is installed

`pip -V`

2. If pip is not installed use the following commands to install it:

`wget https://bootstrap.pypa.io/get-pip.py`

and 

`python3 get-pip.py`

3. Install `virtualenv` and `virtualenvwrapper` using `pip`:

`pip3 install virtualenv virtualenvwrapper`

or

`$HOME/.local/bin/pip3 install virtualenv virtualenvwrapper`

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

`source ~/.bashrc`

6. Create the CLUSREC virtual environment

`mkvirtualenv clusrec -p python3`

7. Install the `pip` requirements listed in `pip.txt`

`$HOME/.local/bin/.virtualenvs/clusrec/bin/pip install -r pip.txt`

# Using CLUSRec

Once the virtual environment was created you can start running the Python scripts included in this repository.

- Make sure the `virtualenv` is activated:

`workon clusrec`


