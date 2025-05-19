from chamfer_distance.Method.forwards import sided_forward_func


def sided_torch(xyz1, xyz2):
    return sided_forward_func("torch", xyz1, xyz2)
