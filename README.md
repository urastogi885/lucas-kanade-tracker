# Lucas-Kanade-Tracker
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/urastogi885/lucas-kanade-tracker/blob/master/LICENSE)

## Overview

This project implements object tracking using Lucas-Kanade template tracking. Object tracking was implemented to track 
3 things: a car, face of a human baby, and a human. 3 different dataset were used for each of them. You can find these 
dataset [here](https://drive.google.com/open?id=1gHAVRtSSuB_yo6xt2TIQl84hBLBkNz3E).

The output videos can be seen [*here*](https://drive.google.com/open?id=1LI2pedrUU_xVriF7smvxJ_RnW34GoKPo).

## Dependencies

- Python3
- Python3-tk
- Python3 Libraries: Numpy, OpenCV-Python, Scipy

## Install Dependencies

- Install Python3, Python3-tk, and the necessary libraries: (if not already installed)

```
sudo apt install python3 python3-tk
sudo apt install python3-pip
pip3 install numpy opencv-python scipy
```

- Check if your system successfully installed all the dependencies
- Open terminal using ```Ctrl+Alt+T``` and enter ```python3```.
- The terminal should now present a new area represented by ```>>>``` to enter python commands
- Now use the following commands to check libraries: (Exit python window using ```Ctrl + Z``` if an error pops up while
running the below commands)

```
import tkinter
import numpy
import cv2
import scipy
```

## Run

- Download each of the dataset mentioned in the [*Overview Section*](https://github.com/urastogi885/lucas-kanade-tracker#overview).
- It is recommended that you save the dataset within outer-most directory-level of the project otherwise it will become 
too cumbersome for you to reference the correct location of the file.
- Using the terminal, clone this repository and go into the project directory, and run the main program:

```
https://github.com/urastogi885/lucas-kanade-tracker
cd lucas-kanade-tracker/Code
python3 main.py dataset dataset_location output_location select_roi
```

- If you have a compressed version of the project, extract it, go into project directory, open the terminal by 
right-clicking on an empty space, and type:

```
cd Code/
python3 main.py dataset dataset_location output_location select_roi
```
- For instance:
```
python3 main.py baby ../DragonBaby/img/ ../DragonBaby/output.avi 0
```
- Choose select_roi as "0" to use saved ROI points and "1" to select ROI region yourself.
- Use the following to define the dataset-parameter in the input arguments:
	- car - Car Dataset
	- bolt - Bolt Dataset
	- baby - Dragon Baby Dataset
- For further documentation on the input arguments, refer 
[*main.py*](https://github.com/urastogi885/lucas-kanade-tracker/blob/master/Code/main.py)
