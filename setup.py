#/usr/bin/env python

from setuptools import setup, find_packages
import re

## parse version from init.py
with open("simcat/__init__.py") as init:
    CUR_VERSION = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                    init.read(),
                    re.M).group(1)

## run setup script
setup(
    name="simcat",
    version=CUR_VERSION,
    url="https://github.com/pmckenz1/intro-ml",
    author="Patrick McKenzie",
    author_email="patrick.mckenzie@columbia.edu",
    description="simulation and machine learning algorithms for admixture inference",
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[],
    entry_points={},
    data_files=[],
    license='GPLv3',
    classifiers=[
        'Programming Language :: Python',
    ],
)
