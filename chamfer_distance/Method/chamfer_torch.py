import torch


def batched_pairwise_dist(x, y):
    bs = x.size(0)
    num_points_x = x.size(1)
    num_points_y = y.size(1)

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)

    if num_points_x < num_points_y:
        zz = torch.bmm(2.0 * x, y.transpose(2, 1))
    else:
        zz = torch.bmm(x, (2.0 * y).transpose(2, 1))

    rx = xx.unsqueeze(2).expand(bs, num_points_x, num_points_y)
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y)

    P = rx + ry - zz
    return P


def chamfer_torch(a, b):
    P = batched_pairwise_dist(a, b)

    dists1, idxs1 = torch.min(P, 2)
    dists2, idxs2 = torch.min(P, 1)

    valid_dists1 = torch.nan_to_num(dists1, 0.0)
    valid_dists2 = torch.nan_to_num(dists2, 0.0)

    return valid_dists1, valid_dists2, idxs1, idxs2
