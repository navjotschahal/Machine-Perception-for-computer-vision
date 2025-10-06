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

    # referring 6DoF Pose from ProjectiveTransformations page 12 by Kostas Daniilidis : https://sites.google.com/seas.upenn.edu/cis5800f2025/lectures?authuser=0

    # Homography Approach
    # Following slides: Pose from Projective Transformation

    # The homography maps from the world plane (z=0) to the image plane.
    # Pw has 3 columns (X,Y,Z), but for a planar homography, we only use the first two (X,Y).
    H = est_homography(Pw[:, :2], Pc)

    # Decompose the homography to get rotation and translation. Normalize by intrinsics: H' = K^{-1} H with columns (a b c)
    # H = K * [r1, r2, t]
    # K_inv * H = [r1, r2, t]
    K_inv = np.linalg.inv(K)
    H_prime = K_inv @ H

    # Extract the column vectors a b c from H'
    a = H_prime[:, 0]
    b = H_prime[:, 1]
    c = H_prime[:, 2]

    # SVD on (a b) to enforce orthonormality (sheet step 3)
    #    (a b) = U_{3x2} diag(s1,s2) V_{2x2}^T
    M = np.column_stack([a, b])                  # 3x2
    U, S, Vt = np.linalg.svd(M, full_matrices=False)  # U:3x2, S:(2,), Vt:2x2
    s1, s2 = float(S[0]), float(S[1])

    # (r1 r2) = U V^T  and  λ = (s1 + s2)/2
    R12 = U @ Vt                                 
    r1 = R12[:, 0]
    r2 = R12[:, 1]
    lambda_ = 0.5 * (s1 + s2)

    # t_cw = c / λ,  r3 = r1 × r2,  R_cw = [r1 r2 r3] see sheet step 4
    t_cw = c / (lambda_ + 1e-12) # adding small regularizer value to avoid division by zero
    r3 = np.cross(r1, r2)
    R_cw = np.column_stack([r1, r2, r3])

    # checking for det(R_cw) = +1 (proper rotation)
    if np.linalg.det(R_cw) < 0:
        R_cw[:, 2] *= -1.0   # flip r3

    # Convert from camera-to-world (R_cw, t_cw) to world-to-camera (R_wc, t_wc)
    R = R_cw.T
    t = -R @ t_cw
    ##### STUDENT CODE END #####

    return R, t
