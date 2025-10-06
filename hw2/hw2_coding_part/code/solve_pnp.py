from est_homography import est_homography
import numpy as np

def PnP(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-N-Point problem with collineation assumption, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3, ) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Homography Approach
    # Following slides: Pose from Projective Transformation

    # The homography maps from the world plane (z=0) to the image plane.
    # Pw has 3 columns (X,Y,Z), but for a planar homography, we only use the first two (X,Y).
    H = est_homography(Pw[:, :2], Pc)

    # Decompose the homography to get rotation and translation.
    # H = K * [r1, r2, t]
    # K_inv * H = [r1, r2, t]
    K_inv = np.linalg.inv(K)
    H_prime = K_inv @ H

    # Extract the column vectors
    h1 = H_prime[:, 0]
    h2 = H_prime[:, 1]
    h3 = H_prime[:, 2]

    # Normalize to enforce that r1 and r2 are unit vectors.
    # The scaling factor lambda is 1/||h1|| (or 1/||h2||).
    norm_factor = np.linalg.norm(h1)
    r1 = h1 / norm_factor
    r2 = h2 / norm_factor
    t_wc = h3 / norm_factor # This is t_wc (world-to-camera)

    # r1 and r2 must be orthogonal. r3 is their cross product.
    r3 = np.cross(r1, r2)
    
    # Assemble the rotation matrix R_wc (world-to-camera)
    R_wc = np.stack((r1, r2, r3), axis=1)

    # The computed R_wc might not be a perfect rotation matrix due to noise.
    # Using SVD to find the closest valid rotation matrix.
    U, _, Vt = np.linalg.svd(R_wc)
    R_wc = U @ Vt

    # We have the transformation from world to camera (R_wc, t_wc).
    # The renderer needs the camera's pose in the world (R_cw, t_cw), which is the inverse.
    # R_cw = R_wc.T
    # t_cw = -R_wc.T @ t_wc
    R = R_wc.T
    t = -R_wc.T @ t_wc
    ##### STUDENT CODE END #####

    return R, t
