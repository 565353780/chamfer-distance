from chamfer_distance.Method.chamfer_torch import chamfer_torch


def sided_torch(a, b):
    dist1, dist2, idx1, idx2 = chamfer_torch(a, b)
    return dist1, idx1
