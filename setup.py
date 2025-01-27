# setup.py

from setuptools import setup, find_packages

setup(
    name='pytorch_project',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'scipy',
        'matplotlib',
        'optuna',
        'PyYAML',
        # Añade otras dependencias necesarias
    ],
)
