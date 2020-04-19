from sys import argv
from utils import lk_tracker

"""
:param dataset_location: Path of dataset for tracking relative to the project folder
"""
script, dataset_location = argv


if __name__ == '__main__':
    # Get full image-locations from the given path
    img_locations = lk_tracker.extract_locations(dataset_location)
