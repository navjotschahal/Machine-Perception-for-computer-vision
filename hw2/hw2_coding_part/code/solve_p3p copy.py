from re import X
import numpy as np

def P3P(Pc, Pw, K=np.eye(3)):
    """
    Solve Perspective-3-Point problem, given correspondence and intrinsic

    Input:
        Pc: 4x2 numpy array of pixel coordinate of the April tag corners in (x,y) format
        Pw: 4x3 numpy array of world coordinate of the April tag corners in (x,y,z) format
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """

    ##### STUDENT CODE START #####

    # Invoke Procrustes function to find R, t
    # You may need to select the R and t that could transoform all 4 points correctly. 
    # R,t = Procrustes(Pc_3d, Pw[1:4])
    
    # referring “Problem Definition” section in Haralick et al. (Grunert)s paper : https://haralick-org.torahcode.us/journals/three_point_perspective.pdf
    # computing a,b,c,cosα,cosβ,cosγ
    # let the unknown positions of the 3 points of the known triangle be
    # choosing 3 of the 4 corners
    i1, i2, i3 = 1, 2, 3
    Pw1, Pw2, Pw3 = Pw[i1], Pw[i2], Pw[i3]

    # the known side lengths of the triangle
    a = np.linalg.norm(Pw2 - Pw3)  # length of side opposite to Pw1
    b = np.linalg.norm(Pw1 - Pw3)  # length of side opposite to Pw2
    c = np.linalg.norm(Pw1 - Pw2)  # length of side opposite to Pw3

    # u, v from Pc
    u = Pc[:, 0]
    v = Pc[:, 1]

    K_inv = np.linalg.inv(K)

    def ray(u, v):
        """Compute the ray direction in camera coordinates for pixel (u,v)"""
        q = K_inv @ np.array([u, v, 1.0])
        return q / np.linalg.norm(q)
    
    # unit vectors 
    j1 = ray(u[i1], v[i1])
    j2 = ray(u[i2], v[i2])
    j3 = ray(u[i3], v[i3])

    # dot products
    cos_alpha = np.dot(j2, j3.T)  # angle at j1
    cos_beta  = np.dot(j1, j3.T)  # angle at j2
    cos_gamma = np.dot(j1, j2.T)  # angle at j3

    # Grunert's  equations (1)(2)(3) in Haralick et al. (Grunert)s paper
    A0 = (1 + (a**2 - c**2) / (b**2))**2 - (4 * a**2 / (b**2)) * (cos_gamma**2)
    A1 = 4 * (-(a**2 - c**2) / (b**2) * (1 + (a**2 - c**2) / (b**2)) * cos_beta 
         + (2 * a**2) / (b**2) * cos_gamma**2 * cos_beta
         - (1 - (a**2 + c**2) / (b**2)) * cos_alpha * cos_gamma)

    A2 = 2 * (((a**2 - c**2) / (b**2))**2 - 1 + 2 * ((a**2 - c**2) / (b**2))**2 * cos_beta**2
            + 2 * ((b**2 - c**2) / (b**2)) * cos_alpha**2
            - 4 * ((a**2 + c**2) / (b**2)) * cos_alpha * cos_beta * cos_gamma
            + 2 * ((b**2 - a**2) / (b**2)) * cos_gamma**2)

    A3 = 4 * ((a**2 - c**2) / (b**2) * (1 - (a**2 - c**2) / (b**2)) * cos_beta
            - (1 - (a**2 + c**2) / (b**2)) * cos_alpha * cos_gamma
            + 2 * (c**2) / (b**2) * cos_alpha**2 * cos_beta)

    A4 = (((a**2 - c**2) / (b**2)) - 1)**2 - (4 * c**2) / (b**2) * cos_alpha**2

    print("A0,A1,A2,A3,A4:", A0, A1, A2, A3, A4)

    # Solve quartic equation A4*x^4 + A3*x^3 + A2*x^2 + A1*x + A0 = 0
    coeffs = [A4, A3, A2, A1, A0]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    positive_real_roots = real_roots[real_roots > 0]
    print("positive_real_roots:", positive_real_roots)



    # Find valid solutions
    

    R, t = None, None
    best_R, best_t = None, None
    error = float('inf')
    for x in positive_real_roots:
        #         This fourth order polynomial equation can
        # have as many as four real roots. By equa-
        # tion (8), to every solution for v there will be a
        # corresponding solution for u. Then having val-
        # ues for u and v it is an easy matter to determine
        # a value for Sl from equation (5). The values
        # for s2 and s3 are immediately determined from
        # equation (4). Most of the time it gives two
        # solutions (Wolfe et al. 1991).
        u = ((-1 + (a**2 - c**2)/(b**2)) * x**2 - 2 * ((a**2 - c**2)/(b**2)) * cos_beta * x + 1 + (a**2 - c**2)/(b**2)) / (2 * (cos_gamma - x * cos_alpha))
        s1 = np.sqrt((b**2) / (1 + x**2 - 2 * x * cos_beta))
        s2 = u * s1 # equation 4
        s3 = x * s1 # equation 4

        # compute the 3D points in camera coordinates
        Pc1 = s1 * j1
        Pc2 = s2 * j2
        Pc3 = s3 * j3

        # compute R, t using Procrustes
        Pc_3d = np.vstack((Pc1, Pc2, Pc3))
        R, t = Procrustes(Pw[i1:], Pc_3d)

        P = np.dot(K, (R @ Pw.T + t.reshape(3, 1)))
        P_last = P[-1]
        P_norm = P / P_last
        Pc_proj = P_norm[:-1].T  # Nx2

        finite_diff_error = np.linalg.norm(Pc - Pc_proj)
        if finite_diff_error < error:
            error = finite_diff_error
            best_R, best_t = R, t   
    

    # Convert from camera-to-world (R_cw, t_cw) to world-to-camera (R_wc, t_wc)
    R, t = best_R.T, -best_R.T @ best_t

    ##### STUDENT CODE END #####

    return R, t



def Procrustes(X, Y):
    """
    Solves the Procrustes problem to find the optimal rotation and translation
    that aligns point set X to point set Y, minimizing ||Y - (RX + t)||^2.

    The transformation is defined as: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in the source coordinate system (e.g., camera coordinates).
        Y: Nx3 numpy array of N points in the target coordinate system (e.g., world coordinates).
    Returns:
        R: 3x3 numpy array, the optimal rotation matrix.
        t: (3,) numpy array, the optimal translation vector.
    """
    # Ensure inputs are numpy arrays
    A = np.asarray(X, dtype=float)
    B = np.asarray(Y, dtype=float)
    assert A.shape == B.shape and A.shape[1] == 3 and A.shape[0] >= 3

    # Calculate centroids
    A_bar = A.mean(axis=0)
    B_bar = B.mean(axis=0)

    # Center the point sets
    A0 = A - A_bar
    B0 = B - B_bar

    # Calculate the covariance matrix (3 x N) @ (N x 3) -> 3 x 3 matrix.
    abt = A0.T @ B0

    # Perform Singular Value Decomposition (SVD) on H
    U, S, Vt = np.linalg.svd(abt)
    V = Vt.T

    # Calculate the optimal rotation matrix R
    R = V @ U.T

    # The reflection case to ensure a proper rotation
    # If det(R) is -1, it's a reflection, not a rotation. We must flip it.
    # This is done by flipping the sign of the last column of V and re-calculating R.
    if np.linalg.det(R) < 0:
        # Create a new V with the last column negated
        V_prime = V.copy()
        V_prime[:, 2] *= -1
        R = V_prime @ U.T

    # Optimal translation vector t
    t = B_bar - R @ A_bar

    return R, t

