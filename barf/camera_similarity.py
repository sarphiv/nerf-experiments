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

