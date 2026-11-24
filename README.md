# Pyrobopath ROS

Ros interfaces and execution clients for [pyrobopath](https://github.com/alexarbogast/pyrobopath).

## Installation

Install the pyrobopath Python package.
```sh
pip install pyrobopath
```

Create a colcon workspace and install the package and its dependencies.
```sh
mkdir -p pyrobopath_ws/src && cd pyrobopath_ws/src
vcs import . < pyrobopath_ros/pyrobopath_ros.repos
cd ../
colcon build
```

## Examples

The `pyrobopath_examples` package provides a simple two-robot example to illustrate system setup and demonstrate the core package functionality.

**Running the Demo**

1) Bringup the two-robot system:

```shell
ros2 launch pyrobopath_examples bringup.launch.py
```

2) Launch the demo in a separate terminal:

```shell
ros2 launch pyrobopath_examples pyrobopath_demo.launch.py
```

**Visualizing Collision Geometry**

```shell
ros2 launch pyrobopath_examples visualize_collision_geometry.launch.py
```

## Documentation

Checkout the [Pyrobopath Documentation](https://pyrobopath.readthedocs.io/en/latest/) for installation help, examples, and API reference.
