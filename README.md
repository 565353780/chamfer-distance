# Chamfer Distance

## Setup

```bash
conda create -n cd python=3.10
conda activate cd
./setup.sh
```

## Run

```bash
from chamfer_distance.Module.chamfer_distances import ChamferDistances
algo_name = 'default' or 'cuda' or 'triton' or 'cukd'
dists1, dists2, idxs1, idxs2 = ChamferDistances.namedAlgo(algo_name)(xyz1, xyz2)
```

## Enjoy it~
