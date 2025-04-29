import numpy as np
import matplotlib.pyplot as plt

from chamfer_distance.Method.map import createDataMapDict, mapData

class FPSMap(object):
    def __init__(self) -> None:
        self.fps_dict = {}
        return

    def reset(self) -> bool:
        self.fps_dict = {}
        return True

    def getFPS(self, x: int, y: int) -> float:
        if (x, y) not in self.fps_dict.keys():
            print('[WARN][FPSMap::getFPS]')
            print('\t this fps not found! will return 0.0!')
            print('\t ( x , y ): (', x, ',', y, ')')
            return 0.0

        return self.fps_dict[(x, y)]

    def addFPS(
        self,
        x: int,
        y: int,
        fps: float = 0.0,
        is_exchange_xy: bool = True,
    ) -> bool:
        if (x, y) in self.fps_dict.keys():
            print('[WARN][FPSMap::addFPS]')
            print('\t already saved this fps value!')
            return True

        self.fps_dict[(x, y)] = fps

        if is_exchange_xy:
            self.fps_dict[(y, x)] = fps

        return True

    def toXYFPS(self) -> np.ndarray:
        x_list = []
        y_list = []
        fps_list = []
        for position, fps in self.fps_dict.items():
            x_list.append(position[0])
            y_list.append(position[1])
            fps_list.append(fps)

        xyfps = np.vstack([x_list, y_list, fps_list])
        return xyfps

    def render(self) -> bool:
        x, y, fps = self.toXYFPS()

        x_map = createDataMapDict(x)
        y_map = createDataMapDict(y)

        mapped_x = mapData(x, x_map)
        mapped_y = mapData(y, y_map)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        dx = dy = 0.5
        dz = fps

        ax.bar3d(mapped_x, mapped_y, np.zeros_like(fps), dx, dy, dz, shade=True, color='skyblue')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('FPS')

        plt.show()
        return True
