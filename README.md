# Pyrobopath ROS

Ros utilities and execution clients for
[pyrobopath](https://github.com/alexarbogast/pyrobopath).

## Installation

First install pyrobopath.
```sh
pip install pyrobopath
```

Create a catkin workspace
```sh
mkdir -p pyrobopath_ws/src && cd pyrobopath_ws/src
```

The Pyrobopath ROS interface depends on the cartesian_planning package for
executing toolpath schedules.

To use the package, clone the
[cartesian_planning](https://github.com/alexarbogast/cartesian_planning)
repository and Pyrobopath ROS into your catkin workspace and build the
packages.

```sh
git clone git@github.com:alexarbogast/cartesian_planning.git
git clone git@github.com:alexarbogast/pyrobopath_ros.git
cd ../
catkin build
```

## Documentation
Checkout the [Pyrobopath
Documentation](https://pyrobopath.readthedocs.io/en/latest/) for installation
help, examples, and API reference. 
