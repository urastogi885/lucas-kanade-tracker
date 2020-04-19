import cv2
from glob import glob


def extract_locations(dataset_location):
    """
    Get locations of all JPEG images from the given folder
    :param dataset_location: location of the folder
    :return: a list containing proper location of all JPEG images
    """
    # Define an empty list to store file locations
    filename_array = []
    # Add file locations to the list
    for filename in glob(str(dataset_location) + '*.jpg'):
        filename_array.append(filename)
    # Sort the list and return it
    filename_array.sort()
    return filename_array

