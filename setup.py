#!/usr/bin/env python
"""
Setup script for ALPUD: Active Learning from Positive and Unlabeled Data
"""

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="alpud",
    version="1.0.0",
    author="Alireza Ghasemi",
    description="Active Learning from Positive and Unlabeled Data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aghasemi/alpud",
    py_modules=["alpud"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
)
