import torch

import chamfer_cpp

from chamfer_distance.Module.base_searcher import BaseSearcher


class FAISSSearcher(BaseSearcher):
    def __init__(self):
        self.searcher = chamfer_cpp.FAISSSearcher()
        return
