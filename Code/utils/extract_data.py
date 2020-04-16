import cv2
from sys import argv
from glob import glob

script, dataset_location = argv

if __name__ == '__main__':
    filename_array = []
    for filename in glob(str(dataset_location) + '*.jpg'):
        filename_array.append(filename)
    filename_array.sort()

    for file in filename_array:
        img = cv2.imread(file)
        cv2.imshow("image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
