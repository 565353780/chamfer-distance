import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.patches import Patch

from chamfer_distance.Method.map import createDataMapDict, mapData


def renderAlgoFPSMapDict(algo_fps_map_dict: dict, free_width: float = 0.1) -> bool:
    algo_num = len(algo_fps_map_dict.keys())
    algo_bar_width = (1.0 - free_width) / algo_num
    algo_bar_start = -0.5 + 0.5 * free_width + 0.5 * algo_bar_width

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    dx = algo_bar_width
    dy = 0.5

    colors = plt.cm.tab10.colors

    legend_patches = []

    algo_idx = 0
    for algo_name, algo_fps_map in algo_fps_map_dict.items():
        x, y, fps = algo_fps_map.toXYFPS()

        delta_x = algo_bar_start + algo_idx * algo_bar_width

        x_map = createDataMapDict(x)
        y_map = createDataMapDict(y)

        mapped_x = mapData(x, x_map)
        mapped_y = mapData(y, y_map)

        mapped_x = mapped_x.astype(float) + delta_x

        color = colors[algo_idx % len(colors)]

        ax.bar3d(
            mapped_x, mapped_y, np.zeros_like(fps),
            dx, dy, fps,
            shade=True,
            color=color,
            alpha=0.8,
        )

        legend_patches.append(Patch(color=color, label=algo_name))

        algo_idx += 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('FPS')

    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    plt.show()
    return True

def renderBestAlgoFPSMapDict(algo_fps_map_dict: dict, free_width: float = 0.5) -> bool:
    algo_bar_width = 1.0 - free_width

    xy_all = []

    for algo_name, algo_fps_map in algo_fps_map_dict.items():
        xy = algo_fps_map.toXY()

        xy_all.append(xy)

    xy_all = np.hstack(xy_all)
    xy_all = np.unique(xy_all, axis=1)

    x_all = xy_all[0]
    y_all = xy_all[1]

    x_map = createDataMapDict(x_all)
    y_map = createDataMapDict(y_all)

    mapped_x = mapData(x_all, x_map)
    mapped_y = mapData(y_all, y_map)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    dx = dy = algo_bar_width

    colors = plt.cm.tab10.colors

    for i in range(x_all.shape[0]):
        query_xy = (x_all[i], y_all[i])

        best_fps = 0.0
        best_algo_idx = -1

        algo_idx = 0
        for algo_fps_map in algo_fps_map_dict.values():
            algo_fps = algo_fps_map.fps_dict[query_xy]
            if algo_fps > best_fps:
                best_fps = algo_fps
                best_algo_idx = algo_idx

            algo_idx += 1

        ax.bar3d(
            mapped_x[i], mapped_y[i], 0.0,
            dx, dy, best_fps,
            shade=True,
            color=colors[best_algo_idx % len(colors)],
            alpha=0.8,
        )

    legend_patches = []

    algo_idx = 0
    for algo_name, algo_fps_map in algo_fps_map_dict.items():
        color = colors[algo_idx % len(colors)]
        legend_patches.append(Patch(color=color, label=algo_name))
        algo_idx += 1

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('FPS')

    ax.legend(handles=legend_patches, loc='upper right')

    plt.tight_layout()
    plt.show()
    return True

def renderAlgoFPSMapDictCurve(algo_fps_map_dict: dict) -> bool:
    plt.figure(figsize=(10, 6))

    cmap = get_cmap("tab10")

    algo_idx = 0
    for algo_name, algo_fps_map in algo_fps_map_dict.items():
        x, y, fps = algo_fps_map.toXYFPS()
        xy = x * y

        xy_map = createDataMapDict(xy)

        mapped_xy = mapData(xy, xy_map)

        color = cmap(algo_idx)

        plt.plot(mapped_xy, fps, label=algo_name, color=color)

        algo_idx += 1

    plt.xlabel('XY')
    plt.ylabel('FPS')
    plt.title('FPS Curve')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return True
