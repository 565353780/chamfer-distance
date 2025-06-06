import torch

from chamfer_distance.Method.nm_dist import nm_dist


def sided_triton(xyz1: torch.Tensor, xyz2: torch.Tensor):
    dist1_list = []
    idx1_list = []

    for i in range(xyz1.shape[0]):
        dist1, idx1 = nm_dist(xyz1[i], xyz2[i])
        dist1_list.append(dist1)
        idx1_list.append(idx1)

    dists1 = torch.vstack(dist1_list)
    idxs1 = torch.vstack(idx1_list)

    return dists1, idxs1
