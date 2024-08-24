from setuptools import setup
from setuptools import setup, find_packages

setup(
    name="gym_wumpus",
    version="0.0.1",
    packages=find_packages(),
    install_requires=['gym'],
    include_package_data=True,
)
