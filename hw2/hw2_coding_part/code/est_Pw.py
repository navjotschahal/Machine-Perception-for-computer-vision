import numpy as np

def est_Pw(s):
    """
    Estimate the world coordinates of the April tag corners, assuming the world origin
    is at the center of the tag, and that the xy plane is in the plane of the April
    tag with the z axis in the tag's facing direction. See world_setup.jpg for details.
    Input:
        s: side length of the April tag

    Returns:
        Pw: 4x3 numpy array describing the world coordinates of the April tag corners
            in the order of a, b, c, d for row order. See world_setup.jpg for details.

    """

    ##### STUDENT CODE START #####
    half_s = s / 2.0
    Pw = np.array([
        [-half_s, -half_s, 0],  # Corner a
        [half_s,  -half_s, 0],  # Corner b
        [ half_s,  half_s, 0],  # Corner c
        [ -half_s, half_s, 0]   # Corner d
    ])
    ##### STUDENT CODE END #####

    return Pw
