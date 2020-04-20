import cv2
from numpy import zeros
from sys import argv
from utils import lk_tracker

"""
:param dataset_location: Path of dataset for tracking relative to the project folder
"""
script, dataset_location, output_location, select_roi = argv


if __name__ == '__main__':
    video_format = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_output = cv2.VideoWriter(str(output_location), video_format, 10.0, (640, 360))
    # Get full image-locations from the given path
    img_locations = lk_tracker.extract_locations(dataset_location)
    template_image = cv2.imread(img_locations[0])
    template_image = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    template_image = cv2.equalizeHist(template_image) / 255
    warp_prev = zeros(2)
    # TODO: Define best points here
    roi_points = [155, 75, 60, 77]
    if int(select_roi):
        roi_points = list(cv2.selectROI('ROI', template_image))
    for img_location in img_locations:
        img_frame = cv2.imread(img_location)
        gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray) / 255
        warp_prev = lk_tracker.affine_lk_tracker(gray, template_image, roi_points, warp_prev)
        bounding_box = roi_points[0] + int(warp_prev[0]), roi_points[1] + int(warp_prev[1])
        cv2.rectangle(img_frame, bounding_box,
                      (bounding_box[0] + roi_points[2], bounding_box[1] + roi_points[3]), (0, 255, 0), 3)
        cv2.imshow("output", img_frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        video_output.write(img_frame)
    video_output.release()
    cv2.destroyAllWindows()
