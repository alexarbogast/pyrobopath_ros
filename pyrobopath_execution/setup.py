from setuptools import find_packages, setup

package_name = "pyrobopath_execution"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "gcodeparser",
        "numpy",
        "numpy-quaternion",
        "pyrobopath",
    ],
    zip_safe=True,
    maintainer="Alex Arbogast",
    maintainer_email="arbogastaw@gmail.com",
    description="Python ROS2 interfaces for pyrobopath schedule execution",
    license="MIT",
    entry_points={
        "console_scripts": [],
    },
)
