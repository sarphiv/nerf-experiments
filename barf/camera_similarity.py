import torch as th

def align_rotation(P, Q):
    """
    optimize ||P@R - Q||^2 using SVD
    """
    H = P.T@Q
    U, S, V = th.linalg.svd(H)
    d = th.linalg.det(V@U.T)
    K = th.eye(len(S))
    K[-1,-1] = d
    R = U@K@V.T
    return R

def align_paired_point_clouds(P, Q):
    """
    align paired point clouds P and Q
    by translating and rotating P into Q
    """
    # translate P to origin
    cP = th.mean(P, dim=0, keepdim=True)
    cQ = th.mean(Q, dim=0, keepdim=True)
    # rotate P to Q
    R = align_rotation(P-cP, Q-cQ)
    Qhat = (P - cP)@R + cQ
    return Qhat, R, cQ@R - cP # t = cQ@R - cP


def expected_unit_vector_error(R1,R2):
    """
    compute the mean squared error between two rotation matrices
    
    That is, let v ~ uniform on the unit sphere in R^3
    then the mean unit vector error is
    E[ ||R1@v - R2@v||^2 ] = 1/3 * trace((R1-R2)@(R1-R2).T)

    Returns: E_{v} ( ||R1@v - R2@v||^2 )
    """

    return th.sqrt(th.trace((R1-R2)@(R1-R2).T)/3)

def max_unit_vector_error(R1,R2):
    """
    compute the maximal squared error between two rotation matrices

    That is, let v ~ uniform on the unit sphere in R^3
    then the max unit vector error is

    maximal eigenvalue of (R1-R2)@(R1-R2).T

    Returns: max_{v} ||R1@v - R2@v||^2

    """

    U, S, V = th.linalg.svd(R1-R2)
    return S[0]**2
