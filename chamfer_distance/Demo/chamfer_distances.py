from chamfer_distance.Module.sided_distances import SidedDistances
from chamfer_distance.Module.chamfer_distances import ChamferDistances


def demo_check_sided():
    xyz1_shape = [1, 4000, 3]
    xyz2_shape = [1, 4000, 3]

    SidedDistances.check(xyz1_shape, xyz2_shape)
    return True


def demo_check_chamfer():
    xyz1_shape = [1, 4000, 3]
    xyz2_shape = [1, 4000, 3]

    ChamferDistances.check(xyz1_shape, xyz2_shape)
    return True
