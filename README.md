# Chamfer Distance

## Setup

```bash
conda create -n cd python=3.10
conda activate cd
./setup.sh
```

## Algo Balance Test

```bash
python test_balance.py
```

## Algo Speed Test

```bash
# 3D Bars
python test_best_speed.py
# Curve
python test_best_speed_curve.py
```

and you can get the curve like

![Algo FPS Curve](https://github.com/565353780/chamfer-distance/blob/master/asset/algo_fps_curve.png)

Just use like

```bash
from chamfer_distance.Module.chamfer_distances import ChamferDistances
# fusion is the best choice in most cases on your GPU
algo_name = 'cpu' or 'cuda' or 'triton' or 'cukd'
dists1, dists2, idxs1, idxs2 = ChamferDistances.namedAlgo(algo_name)(xyz1, xyz2)
```

## Enjoy it~
