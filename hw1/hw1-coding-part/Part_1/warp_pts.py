import numpy as np
from est_homography import est_homography


def warp_pts(X, Y, interior_pts):
    """
    First compute homography from video_pts to logo_pts using X and Y,
    and then use this homography to warp all points inside the soccer goal

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
        interior_pts: Nx2 matrix of points inside goal
    Returns:
        warped_pts: Nx2 matrix containing new coordinates for interior_pts.
        These coordinate describe where a point inside the goal will be warped
        to inside the penn logo. For this assignment, you can keep these new
        coordinates as float numbers.

    """

    # You should Complete est_homography first!
    H = est_homography(X, Y)

    ##### STUDENT CODE START #####

    H = np.asarray(H, dtype=float)
    interior_pts = np.asarray(interior_pts, dtype=float)

    M = interior_pts.shape[0]
    if H.shape != (3, 3):
        raise ValueError("H must be 3x3")
    if interior_pts.ndim != 2 or interior_pts.shape[1] != 2:
        raise ValueError(f"interior_pts must be ({M},2)")

    ones = np.ones((M, 1), dtype=float)
    ph = np.hstack([interior_pts, ones])                 
    qh = (H @ ph.T).T                                 
    w = qh[:, 2:3]
    safe = np.abs(w) > 1e-12
    pts_logo = np.full((M, 2), np.nan, dtype=float)
    pts_logo[safe[:, 0]] = (qh[safe[:, 0], :2] / w[safe[:, 0]]).reshape(-1, 2)
    
    ##### STUDENT CODE END #####

    return pts_logo
