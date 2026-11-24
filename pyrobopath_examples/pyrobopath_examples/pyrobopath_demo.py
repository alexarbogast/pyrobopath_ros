#!/usr/bin/env python3

import numpy as np
import rclpy

from pyrobopath.toolpath import Toolpath, Contour
from pyrobopath.toolpath_scheduling import animate_multi_agent_toolpath_full

from pyrobopath_execution import ScheduleExecution

NAME = "pyrobopath_demo"

LIMITS = ((-1.5, 1.5), (-1.5, 1.5))


def raster_rect(p, h, spacing, n):
    pi = np.array(p)
    raster = [pi.copy()]
    dir = 1.0
    for _ in range(n):
        pi[1] = pi[1] + (h * dir)
        raster.append(pi.copy())
        pi[0] = pi[0] + spacing
        dir *= -1
        raster.append(pi.copy())
    pi[1] = pi[1] + (h * dir)
    raster.append(pi.copy())
    return raster


def rotate_pathZ(path, about, rad):
    about = np.array(about)
    new_path = [p - about for p in path]
    s, c = np.sin(rad), np.cos(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    new_path = [(R @ p) + about for p in new_path]
    return new_path


class PyrobopathDemo(ScheduleExecution):
    def __init__(self):
        super().__init__(NAME)

    def run(self):
        toolpath = self.create_toolpath()
        self.move_home()

        # schedule toolpath
        self.schedule_toolpath(toolpath)

        animate_multi_agent_toolpath_full(
            toolpath,
            self.schedule,
            self.agent_models,
            limits=LIMITS,
        )

        # execute schedule
        self.execute_schedule()

    def create_toolpath(self) -> Toolpath:
        self.get_logger().info("Creating demo toolpath")

        # Layer 1
        path1 = raster_rect([-0.2, -0.2, 0.0], 0.5, 0.05, 1)
        # path2 = raster_rect([0.10, -1.5, 0.0], 3.0, 0.2, 7)

        # Layer 2
        path3 = raster_rect([-1.5, -1.5, 0.003], 1.4, 0.2, 7)
        path4 = raster_rect([-1.5, 1.5, 0.003], 3.0, 0.2, 7)
        path4 = rotate_pathZ(path4, [-1.5, 1.5, 0.003], -np.pi / 2)
        path5 = raster_rect([0.1, -1.5, 0.003], 1.4, 0.2, 7)

        # Layer 3
        path6 = raster_rect([-1.5, 1.5, 0.006], -2.0, 0.2, 5)
        path7 = raster_rect([-0.030, 1.5, 0.006], -2.0, 0.2, 5)
        path8 = raster_rect([-2.3, -1.5, 0.006], 2.2, 0.2, 4)
        path8 = rotate_pathZ(path8, [-1.5, -1.5, 0.006], -np.pi / 2)
        path9 = raster_rect([0.9, -1.5, 0.006], 3.0, 0.2, 3)

        c1 = Contour(path1, tool=0)
        # c2 = Contour(path2, tool=1)
        # c3 = Contour(path3, tool=1)
        # c4 = Contour(path4, tool=0)
        # c5 = Contour(path5, tool=0)
        # c6 = Contour(path6, tool=1)
        # c7 = Contour(path7, tool=0)
        # c8 = Contour(path8, tool=0)
        # c9 = Contour(path9, tool=1)

        return Toolpath([c1])


def main(args=None):
    rclpy.init(args=args)

    node = PyrobopathDemo()
    node.run()
    rclpy.spin(node)
    node.destroy_node()

    rclpy.shutdown()


if __name__ == "__main__":
    main()
