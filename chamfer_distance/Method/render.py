import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from chamfer_distance.Method.map import createDataMapDict, mapData


def renderAlgoFPSMapDict(algo_fps_map_dict: dict, free_width: float = 0.1) -> bool:
    algo_num = len(algo_fps_map_dict.keys())
    algo_bar_width = (1.0 - free_width) / algo_num
    algo_bar_start = -0.5 + 0.5 * free_width

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
