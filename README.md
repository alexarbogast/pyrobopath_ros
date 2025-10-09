# Pyrobopath ROS

Ros interfaces and execution clients for
[pyrobopath](https://github.com/alexarbogast/pyrobopath).

## Installation

Install the pyrobopath Python package.
```sh
pip install pyrobopath
```

Create a colcon workspace and install the package
```sh
mkdir -p pyrobopath_ws/src && cd pyrobopath_ws/src
```

The `pyrobopath_ros` package requires the
[cartesian_planning](https://github.com/alexarbogast/cartesian_planning)
library. Clone the packages and build the colcon workspace.
```sh
git clone git@github.com:alexarbogast/cartesian_planning.git
git clone git@github.com:alexarbogast/pyrobopath_ros.git
cd ../
colcon build
```

## Documentation
Checkout the [Pyrobopath
Documentation](https://pyrobopath.readthedocs.io/en/latest/) for installation
help, examples, and API reference. 
