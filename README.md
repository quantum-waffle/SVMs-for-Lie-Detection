# SVMs-for-Lie-Detection

 ![PyPI - Python Version](https://img.shields.io/badge/python-3.6.5-blue.svg) ![PyPI - Library](https://img.shields.io/badge/library-tensorflow-blue.svg)  
This proyect provides a Support Vector Machine implementation for lie detection, based on EGG metrics. 
It also contains several test files that helped determine the best model for the data. 
It relies on Tensorflow, you can get it [here](https://www.tensorflow.org/).

## Installation
### Before Installation
It is recommended to use a `python virtual enviroment`. Create it with:
``` sh
python3 -m venv MP
```
and run it with:
``` sh
source MP/bin/activate
```
### General Instructions
#### python
First, install all dependencies using `pip`:
``` sh
sudo pip install -r requirements.txt
```

## Run
Assuming you run it from a terminal within the main folder:
``` sh
python tensor_nonlinear_gaussian_rbf_SVM.py
```
