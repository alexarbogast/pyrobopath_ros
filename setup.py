from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

package_name = "pyrobopath_ros"

setup_args = generate_distutils_setup(
    packages=[package_name],
    package_dir={'': 'src'},
    requires=[
        "gcodeparser",
        "numpy",
        "numpy-quaternion",
        "pyrobopath",
    ],
)

setup(**setup_args)
