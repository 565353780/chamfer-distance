import numpy as np
from scipy.spatial.distance import cdist


def chamfer_sp(ref_pts, query_pts):
    dist_mat = cdist(ref_pts, query_pts, metric='sqeuclidean')
    ref_closest_index_sp = np.argmin(dist_mat, axis=1)
    ref_closest_dist_sp = np.min(dist_mat, axis=1)
    query_closest_index_sp = np.argmin(dist_mat, axis=0)
    query_closest_dist_sp = np.min(dist_mat, axis=0)

    return ref_closest_dist_sp, ref_closest_index_sp, query_closest_dist_sp, query_closest_index_sp
