import numpy as np
from glob import glob
from ast import literal_eval
from scipy.interpolate import RectBivariateSpline as rbs


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


def get_roi_points(dataset):
    """
    Retrieve saved roi points for best results
    :param dataset: a string representing the dataset user wants to run
    :return: a list containing ROI points
    """
    dataset = str(dataset).lower()
    with open('utils/roi.txt', 'r') as roi:
        template = roi.readlines()
        if dataset == 'car':
            template = template[0].split(':')
        elif dataset == 'baby':
            template = template[1].split(':')
        elif dataset == 'bolt':
            template = template[2].split(':')
        else:
            print('Error: No ROI points for the given dataset! Choose ROI')
            return None
    return list(literal_eval(template[1]))


def affine_lk_tracker(img, tmp, rect, warp_prev):
    """
    Method to track a template frame in a new image frame using lucas kanade tracking algorithm
    :param img: current image frame to track template
    :param tmp: template image frame
    :param rect: bonding box coordinates marking the template region in template image frame
    :param warp_prev: warping parameters from warping of previous image frame
    :return: warping parameters of current image frame
    """
    # Get start and end point of the bounding rectangle in the template image
    start_point = rect[0], rect[1]
    opp_corner_point = rect[0] + rect[2], rect[1] + rect[3]
    # Define initial error and convergence threshold
    error = 1
    convergence_threshold = 0.001
    # Get x and y values
    x = np.arange(0, tmp.shape[0], 1)
    y = np.arange(0, tmp.shape[1], 1)
    # Interpolate points from top-left to bottom-right corner of the bounding box
    a = np.linspace(start_point[0], opp_corner_point[0], 87)
    b = np.linspace(start_point[1], opp_corner_point[1], 36)
    # Get a mesh grid from these interpolated points
    mesh_a, mesh_b = np.meshgrid(a, b)
    # Get bivariate spline of template image and evaluate intensities over interpolated points
    spline_tmp = rbs(x, y, tmp)
    intensities_tmp = spline_tmp.ev(mesh_b, mesh_a)
    # Adjust brightness scale
    # Uncomment to scale brightness in the roi region
    # img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = adjust_brightness(img, tmp, rect)
    # Get bivariate spline and gradient of current image frame
    spline_img = rbs(x, y, img)
    grad_y, grad_x = np.gradient(img)
    # Get bivariate spine of gradient
    spline_grad_x = rbs(x, y, grad_x)
    spline_grad_y = rbs(x, y, grad_y)
    # Define a 4x4 jacobian
    jacobian = np.array([[1, 0], [0, 1]])
    count = 0
    # Iterate until convergence
    while np.linalg.norm(error) > convergence_threshold:
        # Interpolate warping points from top-left to bottom-right corner
        warp_a = np.linspace(start_point[0] + warp_prev[0], opp_corner_point[0] + warp_prev[0], 87)
        warp_b = np.linspace(start_point[1] + warp_prev[1], opp_corner_point[1] + warp_prev[1], 36)
        # Get mesh from these interpolated points
        mesh_warp_a, mesh_warp_b = np.meshgrid(warp_a, warp_b)
        # Evaluate intensities over interpolated points in x,y direction for gradients
        intensities_grad_x = spline_grad_x.ev(mesh_warp_b, mesh_warp_a)
        intensities_grad_y = spline_grad_y.ev(mesh_warp_b, mesh_warp_a)
        # Stack the intensities from both the gradients
        intensities_grad = np.vstack((intensities_grad_x.ravel(), intensities_grad_y.ravel())).T
        # Evaluate intensities over the interpolated points for current image frame
        intensities_img = spline_img.ev(mesh_warp_b, mesh_warp_a)
        # Calculate the hessian matrix
        hessian = intensities_grad @ jacobian
        hess = hessian.T @ hessian
        # Calculate the change in intensities from the template image to the current image frame
        change = (intensities_tmp - intensities_img).reshape(-1, 1)
        # Uncomment to add huber loss implementation
        change = get_huber_loss(change)
        # Evaluate errors
        try:
            error = np.linalg.inv(hess) @ hessian.T @ change
            warp_prev[0] += error[0, 0]
            warp_prev[1] += error[1, 0]
        except np.linalg.LinAlgError:
            break
        # Increment iteration count
        count += 1
        # Terminate convergence after 200 iterations
        if count > 200:
            break

    return warp_prev


def adjust_brightness(img, tmp, roi):
    """
    Scale brightness to maintain similar brightness in the tracking region
    :param img: current image frame to track template
    :param tmp: template image frame
    :param roi: bounding box for the ROI in open-cv rect format
    :return: current image frame with brightness adjustment
    """
    tmp = tmp[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    img = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
    tmp_mean = np.full(img.shape, np.mean(tmp))
    img_mean = np.full(img.shape, np.mean(img))
    std_ = np.std(img)
    z_score = np.true_divide((img - tmp_mean), std_)
    d_mean = np.mean(img) - np.mean(tmp)
    if d_mean < 0.1:
        shifted_img = (z_score * std_) + img_mean
    else:
        shifted_img = -(z_score * std_) + img_mean

    return shifted_img.astype(dtype=np.uint8)


def get_huber_loss(change):
    """
    Implement a huber-loss method
    :param change: difference between template and image
    :return:
    """
    huber_thresh = 0.004
    return (huber_thresh ** 2) * (np.sqrt(1 + (change / huber_thresh) ** 2) - 1)
