import cv2
from glob import glob


def extract_locations(dataset_location):
    """
    Get locations of all JPEG images from the given folder
    :param dataset_location: location of the folder
    :return: a list containing proper location of all JPEG images
    """
    filename_array = []
    for filename in glob(str(dataset_location) + '*.jpg'):
        filename_array.append(filename)

    return filename_array.sort()

