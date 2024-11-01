from setuptools import setup, find_packages
import os

with open("README.md","r") as fh:
    long_description = fh.read()

VERSION = '0.0.2'
DESCRIPTION = 'Decomposition using shapley values'

# Setting up
setup(
    name="shapley_decomposition",
    version=VERSION,
    url="https://github.com/canitez01/shapley_decomposition",
    author="Can Itez",
    author_email="<canitez01@hotmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    license = "MIT",
    packages = find_packages(),
    install_requires = ['pandas>=1.1', 'numpy>=1.3', 'scikit-learn>=0.24'],
    extras_require = {"dev":["pytest>=6.2.4"]},
    keywords=['python', 'data analysis', 'descriptive analysis', 'shapley values', 'owen values', 'decomposition'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
