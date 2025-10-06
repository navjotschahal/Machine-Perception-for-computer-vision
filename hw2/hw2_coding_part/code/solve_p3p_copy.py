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
    raise NotImplementedError("P3P() not implemented!")
    ##### STUDENT CODE END #####

    return R, t

def Procrustes(X, Y):
    """
    Solve Procrustes: Y = RX + t

    Input:
        X: Nx3 numpy array of N points in camera coordinate (returned by your P3P)
        Y: Nx3 numpy array of N points in world coordinate
    Returns:
        R: 3x3 numpy array describing camera orientation in the world (R_wc)
        t: (3,) numpy array describing camera translation in the world (t_wc)

    """
    ##### STUDENT CODE START #####
    raise NotImplementedError("Procrustes() not implemened!")
    ##### STUDENT CODE END #####

    return R, t
