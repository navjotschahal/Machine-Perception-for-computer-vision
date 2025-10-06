import numpy as np


def est_homography(X, Y):
    """
    Calculates the homography of two planes, from the plane defined by X
    to the plane defined by Y. In this assignment, X are the coordinates of the
    four corners of the soccer goal while Y are the four corners of the penn logo

    Input:
        X: 4x2 matrix of (x,y) coordinates of goal corners in video frame
        Y: 4x2 matrix of (x,y) coordinates of logo corners in penn logo
    Returns:
        H: 3x3 homogeneours transformation matrix s.t. Y ~ H*X

    """

    ##### STUDENT CODE START #####

    X = np.asarray(X, dtype=float)
    Y  = np.asarray(Y,  dtype=float)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("pts X and pts Y must both be (N,2) with the same N")
    if X.shape[0] < 4:
        raise ValueError("At least 4 point correspondences are required")
    else:
        T_src = np.eye(3)
        T_dst = np.eye(3)
        src_n = X
        dst_n = Y

    N = src_n.shape[0]
    A = np.zeros((2 * N, 9), dtype=float)
    x, y = src_n[:, 0], src_n[:, 1]
    xp, yp = dst_n[:, 0], dst_n[:, 1]

    A[0::2, 0:3] = -np.stack([x, y, np.ones_like(x)], axis=1)
    A[0::2, 6:9] =  np.stack([x * xp, y * xp, xp], axis=1)
    A[1::2, 3:6] = -np.stack([x, y, np.ones_like(x)], axis=1)
    A[1::2, 6:9] =  np.stack([x * yp, y * yp, yp], axis=1)

    # [ U, S , V] = svd (A) ;
    # The vector h will then be the last column of V , and you can then construct the 3x3
    # homography matrix by reshaping the 9x1 h vector.
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H_tilde = h.reshape(3, 3)

    H = np.linalg.inv(T_dst) @ H_tilde @ T_src

    # Fix scale so H[2,2] = 1 (if possible)
    if np.abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    else:
        H = H / (np.linalg.norm(H) + 1e-12)

    ##### STUDENT CODE END #####

    return H


if __name__ == "__main__":
    # You could run this file to test out your est_homography implementation
    #   $ python est_homography.py
    # Here is an example to test your code, 
    # but you need to work out the solution H yourself.
    X = np.array([[0, 0],[0, 10], [5, 0], [5, 10]])
    Y = np.array([[3, 4], [4, 11],[8, 5], [9, 12]])
    H = est_homography(X, Y)
    print(H)
    