# lucas-kanade-tracker
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://github.com/urastogi885/lucas-kanade-tracker/blob/master/LICENSE)

## Overview

This project implements object tracking using Lucas-Kanade template tracking. Object tracking was implemented to track 
3 things: a car, face of a human baby, and a human. 3 different dataset were used for each of them. You can find these 
dataset [here](https://drive.google.com/open?id=1gHAVRtSSuB_yo6xt2TIQl84hBLBkNz3E).

## Dependencies

- Python3
- Python3-tk
- Python3 Libraries: Numpy, OpenCV-Python

## Install Dependencies

- Install Python3, Python3-tk, and the necessary libraries: (if not already installed)

```
sudo apt install python3 python3-tk
sudo apt install python3-pip
pip3 install numpy opencv-python
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
```

## Run

- Download each of the dataset mentioned in the [*Overview Section*](https://github.com/urastogi885/lucas-kanade-tracker#overview).
- It is recommended that you save the dataset within outer-most directory-level of the project otherwise it will become 
too cumbersome for you to reference the correct location of the file.
- Using the terminal, clone this repository and go into the project directory, and run the main program:

```
https://github.com/urastogi885/lucas-kanade-tracker
cd lucas-kanade-tracker/Code
python3 utils/lucas-kanade-tracker.py dataset_location
```

- For instance:
```
python3 utils/lucas-kanade-tracker.py ../Bolt2/img
```
