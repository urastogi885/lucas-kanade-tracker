import cv2
from sys import argv
import numpy as np
from numpy import zeros
from utils import lk_tracker

"""
:param dataset: Dataset to be used for tracking
:param dataset_location: Path of dataset for tracking relative to the project folder
:param output_location: Path of the output video relative to the project folder
:param select_roi: 1 to select a new roi and 0 to use saved roi points 
"""
script, dataset, dataset_location, output_location, select_roi = argv

if __name__ == '__main__':
    # Get full image-locations from the given path
    img_locations = lk_tracker.extract_locations(dataset_location)
    template_image = cv2.imread(img_locations[0])
    width, height, _ = template_image.shape
    # Define parameters for recording output video
    video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_output = cv2.VideoWriter(str(output_location), video_format, 30.0, (height, width))
    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    # Add total least squares method for robustness
    template_image = cv2.equalizeHist(template_image) / 255
    # Apply gamma correction
    template_image = np.array(255*(template_image / 255) ** 0.2, dtype='uint8')
    warp_prev = zeros(2)
    # Retrieve saved roi points for best results
    roi_points = lk_tracker.get_roi_points(dataset)
    if roi_points is None:
        select_roi = 1
    if int(select_roi):
        roi_points = list(cv2.selectROI('ROI', template_image))
    prev_box = roi_points[0], roi_points[1]
    count = 0
    for img_location in img_locations:
        count += 1
        img_frame = cv2.imread(img_location)
        gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        # Add total least squares method for robustness
        gray = cv2.equalizeHist(gray) / 255
        # Apply gamma correction
        gray = np.array(255*(gray / 255) ** 0.2, dtype='uint8')
        warp_prev = lk_tracker.affine_lk_tracker(gray, template_image, roi_points, warp_prev)
        bounding_box = roi_points[0] + int(warp_prev[0]), roi_points[1] + int(warp_prev[1])
        if 0 > bounding_box[0] or bounding_box[0] >= width or 0 > bounding_box[1] or bounding_box[1] >= height:
            bounding_box = prev_box
        cv2.rectangle(img_frame, bounding_box,
                      (bounding_box[0] + roi_points[2], bounding_box[1] + roi_points[3]),
                      (0, 255, 0), 3)
        prev_box = bounding_box
        video_output.write(img_frame)
    video_output.release()
    cv2.destroyAllWindows()
