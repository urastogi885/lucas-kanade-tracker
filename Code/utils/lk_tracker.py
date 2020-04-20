import numpy as np
from glob import glob
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
    # Get bivariate spline and gradient of current image frame
    spline_img = rbs(x, y, img)
    grad_y, grad_x = np.gradient(img)
    # Get bivariate spine of gradient
    spline_grad_x = rbs(x, y, grad_x)
    spline_grad_y = rbs(x, y, grad_y)
    # Define a 4x4 jacobian
    jacobian = np.array([[1, 0], [0, 1]])
    # Iterate until convergence
    while np.square(error).sum() > convergence_threshold:
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
        # Evaluate errors
        error = np.linalg.inv(hess) @ hessian.T @ change
        warp_prev[0] += error[0, 0]
        warp_prev[1] += error[1, 0]

    return warp_prev
