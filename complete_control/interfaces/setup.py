"""Setup script for interfaces package."""

from setuptools import find_packages, setup

setup(
    name="controller-interfaces",
    version="1.0.0",
    description="Interface definitions for controller submodule implementations",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
