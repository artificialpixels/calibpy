#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# TODO: add install requirements
requirements = ['numpy', 'Pillow']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

# TODO: add console scripts
entry_points = {
    'console_scripts': [],
}

setup(
    author='artificialpixels',
    classifiers=[
        'Programming Language :: Python :: 3.9',
    ],
    description='Calibration Toolbox',
    entry_points=entry_points,
    python_requires='>=3.9',
    install_requires=requirements,
    license='Not open source',
    long_description=readme,
    keywords='calibpy',
    name='calibpy',
    packages=find_packages(exclude=['tests']),
    setup_requires=setup_requirements,
    test_suite='test',
    tests_require=test_requirements,
    url='https://github.com/artificialpixels/calibpy',
    version='0.0.1',
    zip_safe=False,
    package_dir={'calibpy': 'calibpy'},
)
