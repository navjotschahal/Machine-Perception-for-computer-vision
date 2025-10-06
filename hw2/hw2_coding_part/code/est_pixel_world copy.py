import numpy as np

def est_pixel_world(pixels, R_wc, t_wc, K):
    """
    Estimate the world coordinates of a point given a set of pixel coordinates.
    The points are assumed to lie on the x-y plane in the world.
    Input:
        pixels: N x 2 coordiantes of pixels
        R_wc: (3, 3) Rotation of camera in world
        t_wc: (3, ) translation from world to camera
        K: 3 x 3 camara intrinsics
    Returns:
        Pw: N x 3 points, the world coordinates of pixels
    """

    ##### STUDENT CODE START #####
    # The camera pose (R_wc, t_wc) transforms points from camera to world.
    # For projection, we need the inverse: from world to camera.
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc

    # We can form a homography H that maps a point on the Z=0 world plane
    # to a pixel in the image.
    # The columns of the rotation matrix correspond to the axes.
    # We use the first two columns (for X and Y) and the translation vector.
    H = K @ np.hstack((R_cw[:, 0:1], R_cw[:, 1:2], t_cw.reshape(3, 1)))
    
    # To go from a pixel to a world coordinate, we need the inverse of H.
    H_inv = np.linalg.inv(H)
    
    Pw = []
    for pixel in pixels:
        # Convert the pixel to homogeneous coordinates
        p_img_hom = np.array([pixel[0], pixel[1], 1.0])
        
        # Apply the inverse homography
        p_world_hom = H_inv @ p_img_hom
        
        # De-homogenize to get the 2D point on the Z=0 plane
        p_world_2d = p_world_hom / p_world_hom[2]
        
        # Append the 3D world coordinate (X, Y, 0)
        Pw.append([p_world_2d[0], p_world_2d[1], 0])
        
    Pw = np.array(Pw)       
    ##### STUDENT CODE END #####
    
    return Pw
