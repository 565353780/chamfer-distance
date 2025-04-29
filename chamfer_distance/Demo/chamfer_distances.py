from chamfer_distance.Module.chamfer_distances import ChamferDistances


def demo():
    xyz1_shape = [1, 4000, 3]
    xyz2_shape = [1, 4000, 3]

    ChamferDistances.check(xyz1_shape, xyz2_shape)
    return True
