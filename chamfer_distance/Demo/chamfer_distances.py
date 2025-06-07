from chamfer_distance.Module.chamfer_distances import ChamferDistances


def demo_check_chamfer():
    xyz1_shape = [1, 4000, 3]
    xyz2_shape = [1, 4000, 3]

    ChamferDistances.check(xyz1_shape, xyz2_shape, True, True)
    ChamferDistances.check(xyz1_shape, xyz2_shape, True, False)
    ChamferDistances.check(xyz1_shape, xyz2_shape, False, True)
    return True
