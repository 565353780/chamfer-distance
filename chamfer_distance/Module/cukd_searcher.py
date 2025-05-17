import torch

import chamfer_cpp

from chamfer_distance.Module.base_searcher import BaseSearcher


class CUKDSearcher(BaseSearcher):
    def __init__(self):
        super().__init__()
        self.searcher = chamfer_cpp.CUKDSearcher()
        return
