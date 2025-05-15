from chamfer_distance.Method.chamfer_cpu import chamfer_cpu


def sided_torch(a, b):
    dist1, dist2, idx1, idx2 = chamfer_cpu(a, b)
    return dist1, idx1


def chamfer_torch(a, b):
    return chamfer_cpu(a, b)
